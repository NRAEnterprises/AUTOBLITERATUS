    refusal_direction_norm: float
    projection_strength: float
    target_component: str
    norm_preserve: bool = True


@dataclass
class AblationPlan:
    model_path: str
    architecture_family: str
    architecture_class: str
    reasoning_class: str
    param_bucket: str
    alignment_method: str
    defense_type: str
    refusal_geometry: str
    direction_count: int
    component_split: dict
    layer_plans: list = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)
    quantization: str = "none"
    target_frameworks: list = field(default_factory=list)
    deployment_target: str = "server"
    seeds: dict = field(default_factory=dict)
    derived_at: str = ""


class YamlConfigAutoFigurer:

    FAMILY_TABLE = {
        "llama":        ("dense_transformer", "dense"),
        "mistral":      ("dense_transformer", "dense"),
        "qwen2":        ("dense_transformer", "dense"),
        "qwen3":        ("dense_transformer", "dense"),
        "mixtral":      ("moe_sparse", "moe"),
        "deepseek_v2":  ("moe_shared_routed", "moe"),
        "deepseek_v3":  ("moe_shared_routed", "moe"),
        "mamba":        ("ssm", "recurrent"),
        "rwkv":         ("rwkv", "recurrent"),
        "retnet":       ("retnet", "recurrent"),
        "jamba":        ("hybrid_attn_ssm", "hybrid"),
    }

    def __init__(self, model_path: str, work_dir: str = "./abliterate_workspace"):
        self.model_path = model_path
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.work_dir / "auto_config.yaml"

    def build(self, capability_probe: dict, refusal_probe: dict, telemetry: dict) -> AblationPlan:
        cfg = self._read_config()
        family, klass = self._classify(cfg)
        plan = AblationPlan(
            model_path=self.model_path,
            architecture_family=family,
            architecture_class=klass,
            reasoning_class=capability_probe.get("reasoning_class", "standard"),
            param_bucket=self._param_bucket(cfg),
            alignment_method=capability_probe.get("alignment_method", "unknown"),
            defense_type=refusal_probe.get("defense_type", "none"),
            refusal_geometry=refusal_probe.get("geometry", "linear"),
            direction_count=refusal_probe.get("direction_count", 1),
            component_split=self._component_split(family),
            thresholds=self._thresholds(capability_probe),
            quantization="none",
            target_frameworks=["transformers"],
            deployment_target="server",
            seeds={"python": 0, "numpy": 0, "torch": 0},
            layer_plans=self._layer_plans(refusal_probe),
            derived_at=datetime.now(timezone.utc).isoformat(),
        )
        self._write(plan)
        return plan

    def _read_config(self) -> dict:
        p = Path(self.model_path) / "config.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _classify(self, cfg: dict) -> tuple:
        model_type = (cfg.get("model_type") or "").lower()
        return self.FAMILY_TABLE.get(model_type, ("unknown", "unknown"))
