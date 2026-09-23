"""Tuning type 35: collaborative auto tuning.

Multiple researchers adjust parameters in real time. The system merges
their inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


TOGGLE = {"id": "tuning.collaborative_auto", "default": False}


@dataclass
class CollaborativeSession:
    session_id: str
    participants: list = field(default_factory=list)
    state: dict = field(default_factory=dict)


class CollaborativeAutoTuning:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.session: CollaborativeSession | None = None

    def open(self, session_id: str) -> CollaborativeSession:
        if not self.enabled:
            return CollaborativeSession(session_id=session_id)
        self.session = CollaborativeSession(session_id=session_id)
        return self.session

    def apply(self, participant: str, change: dict) -> dict:
        if not self.enabled or self.session is None:
            return {}
        self.session.state.update(change)
        return self.session.state
