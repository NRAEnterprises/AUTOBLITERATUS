"""Transactional in-memory model checkpointing for mutation pipelines."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def fingerprint_model(model: Any) -> str:
    """Hash ordered state-dict names, shapes, dtypes, and tensor bytes."""
    state_dict = model.state_dict()
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        shape = tuple(getattr(value, "shape", ()))
        digest.update(repr(shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(getattr(value, "dtype", "unknown")).encode("ascii"))
        digest.update(b"\0")
        try:
            payload = value.detach().to(device="cpu").contiguous().numpy().tobytes()
        except AttributeError:
            payload = repr(value).encode("utf-8")
        digest.update(payload)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass
class MutationTransaction:
    """Hold an exact pre-mutation model snapshot and durable checkpoint."""

    adapter: Any
    checkpoint_dir: Path
    stage: str = "mutation"
    active: bool = False
    committed: bool = False
    rolled_back: bool = False
    before_fingerprint: str = ""
    after_fingerprint: str | None = None
    _original_model: Any = None
    _state_snapshot: Any = None

    @classmethod
    def begin(
        cls,
        adapter: Any,
        work_dir: str | os.PathLike[str],
        *,
        stage: str = "mutation",
        model_revision: str | None = None,
    ) -> MutationTransaction:
        model = adapter.handle.model
        root = Path(work_dir).expanduser().resolve() / "mutation_checkpoints"
        root.mkdir(parents=True, exist_ok=True)
        checkpoint_id = uuid.uuid4().hex
        staging = Path(tempfile.mkdtemp(prefix=f".{stage}-{checkpoint_id}.", dir=str(root)))
        target = root / f"{stage}-{checkpoint_id}"
        transaction = cls(
            adapter=adapter,
            checkpoint_dir=staging,
            stage=stage,
            active=True,
            before_fingerprint=fingerprint_model(model),
            _original_model=model,
            _state_snapshot=copy.deepcopy(model.state_dict()),
        )
        manifest = {
            "schema_version": 1,
            "stage": stage,
            "status": "armed",
            "model_revision": model_revision,
            "before_fingerprint": transaction.before_fingerprint,
        }
        try:
            import torch

            with (staging / "state.pt").open("wb") as stream:
                torch.save(transaction._state_snapshot, stream)
                stream.flush()
                os.fsync(stream.fileno())
            _atomic_json(staging / "checkpoint.json", manifest)
            os.replace(staging, target)
            transaction.checkpoint_dir = target
        except Exception:
            transaction.checkpoint_dir = staging
            transaction._cleanup()
            raise
        return transaction

    def rollback(self, reason: str = "stage_failure") -> dict[str, Any]:
        if not self.active:
            return {
                "status": "not_active",
                "rolled_back": self.rolled_back,
                "before_fingerprint": self.before_fingerprint,
                "after_fingerprint": self.after_fingerprint,
            }
        original_model = self._original_model
        self.adapter.handle.model = original_model
        modules = getattr(self.adapter, "modules", None)
        if modules is not None and hasattr(modules, "model"):
            modules.model = original_model
        original_model.load_state_dict(self._state_snapshot, strict=True)
        restored = fingerprint_model(original_model)
        if restored != self.before_fingerprint:
            raise RuntimeError(
                "Mutation rollback fingerprint mismatch: "
                f"expected {self.before_fingerprint}, restored {restored}."
            )
        self.after_fingerprint = restored
        self.active = False
        self.rolled_back = True
        _atomic_json(
            self.checkpoint_dir / "checkpoint.json",
            {
                "schema_version": 1,
                "stage": self.stage,
                "status": "rolled_back",
                "reason": reason,
                "before_fingerprint": self.before_fingerprint,
                "restored_fingerprint": restored,
            },
        )
        return {
            "status": "rolled_back",
            "rolled_back": True,
            "reason": reason,
            "before_fingerprint": self.before_fingerprint,
            "restored_fingerprint": restored,
        }

    def commit(self) -> dict[str, Any]:
        if not self.active:
            if self.committed:
                return {"status": "committed", "after_fingerprint": self.after_fingerprint}
            raise RuntimeError("Cannot commit an inactive mutation transaction.")
        self.after_fingerprint = fingerprint_model(self.adapter.handle.model)
        if self.after_fingerprint == self.before_fingerprint:
            raise RuntimeError("Mutation transaction produced no state change; refusing to commit.")
        self.active = False
        self.committed = True
        _atomic_json(
            self.checkpoint_dir / "checkpoint.json",
            {
                "schema_version": 1,
                "stage": self.stage,
                "status": "committed",
                "before_fingerprint": self.before_fingerprint,
                "after_fingerprint": self.after_fingerprint,
            },
        )
        return {
            "status": "committed",
            "before_fingerprint": self.before_fingerprint,
            "after_fingerprint": self.after_fingerprint,
        }

    def _cleanup(self) -> None:
        for path in sorted(self.checkpoint_dir.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        self.checkpoint_dir.rmdir()
