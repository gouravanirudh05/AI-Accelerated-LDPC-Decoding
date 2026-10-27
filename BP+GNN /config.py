"""YAML + argparse configuration.

The repo's Phase 0-2 trainer is argparse-driven, so this stays plain
YAML-plus-argparse rather than pulling in Hydra.  A run is::

    python -m ldpc_corrector.train --config ldpc_corrector/configs/gcn.yaml \\
        --set train.steps=2000 loss.hard_mining=true

``--set`` takes dotted keys so every ablation in the spec is reachable without
adding a flag per knob.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml


@dataclass
class CodeConfig:
    code_type: str = "LDPC"
    n: int = 49
    k: int = 24


@dataclass
class BPConfig:
    iterations: int = 5
    damping: float = 0.0
    freeze: bool = True  # no gradient through BP unless explicitly enabled
    # BP iterations resumed after the correction (see CorrectorDecoder); 0 = off.
    resume_iterations: int = 0


@dataclass
class ModelConfig:
    corrector: str = "gcn"
    hidden_dim: int = 32
    dropout: float = 0.0
    # Bound on the per-bit residual. 8.0 rather than the legacy per-edge 2.0:
    # see the note in configs/*.yaml for the measurement behind it.
    correction_scale: float = 8.0
    use_gate: bool = False
    fixed_gate: float = 1.0
    # Architecture-specific knobs; unknown keys are passed to the corrector.
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    steps: int = 3000
    batch_size: int = 256
    lr: float = 2.0e-4
    min_lr: float = 1.0e-6
    weight_decay: float = 1.0e-5
    grad_clip: float = 1.0
    log_every: int = 50
    eval_every: int = 500
    ebn0_min: float = 2.0
    ebn0_max: float = 6.0
    # "uniform" draws a continuous Eb/N0 per frame; "discrete" draws from
    # eval_ebn0, matching the SNR points the baselines were measured at.
    ebn0_distribution: str = "uniform"
    zero_codeword: bool = False


@dataclass
class EvalConfig:
    ebn0: List[float] = field(default_factory=lambda: [2.0, 3.0, 4.0, 5.0, 6.0])
    frames: int = 51200
    batch_size: int = 2048
    # Stop early at an SNR point once this many frame errors are seen; keeps
    # the high-SNR tail affordable on CPU.
    min_frame_errors: int = 200
    timing_batches: int = 5


@dataclass
class RunConfig:
    name: str = "gcn"
    seed: int = 42
    validation_seed: int = 43
    deterministic: bool = True
    device: str = ""
    # BP's check update is a Python loop over many small per-check tensors, so
    # intra-op threading costs more than it saves: measured on this code,
    # 1 thread runs at 146 ms/step vs 244 ms/step at 4 threads. 0 leaves
    # torch's default alone.
    torch_threads: int = 1
    save_dir: str = "Results_Corrector"
    code: CodeConfig = field(default_factory=CodeConfig)
    bp: BPConfig = field(default_factory=BPConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    loss: Dict[str, Any] = field(default_factory=lambda: {"bce": True})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


_SECTIONS = {
    "code": CodeConfig,
    "bp": BPConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
}


def _coerce(value: str) -> Any:
    """Parse a ``--set`` value as YAML so types come out right."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _apply_override(tree: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node: Any = tree
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _build_section(name: str, payload: Mapping[str, Any]) -> Any:
    cls = _SECTIONS[name]
    known = set(cls.__dataclass_fields__)
    supplied = dict(payload)
    if name == "model":
        # Anything the shared ModelConfig does not know is an
        # architecture-specific knob and goes to the corrector verbatim.
        extra = dict(supplied.pop("extra", {}) or {})
        for key in list(supplied):
            if key not in known:
                extra[key] = supplied.pop(key)
        supplied["extra"] = extra
    else:
        unknown = set(supplied) - known
        if unknown:
            raise KeyError(f"Unknown keys in config section {name!r}: {sorted(unknown)}")
    return cls(**supplied)


def load_config(
    path: Optional[str | Path] = None,
    overrides: Optional[Sequence[str]] = None,
) -> RunConfig:
    """Load a YAML config and apply ``key.subkey=value`` overrides."""
    tree: Dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a YAML mapping.")
        tree = copy.deepcopy(loaded)

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(
                f"Override {override!r} must be of the form key.subkey=value."
            )
        key, raw_value = override.split("=", 1)
        _apply_override(tree, key.strip(), _coerce(raw_value.strip()))

    sections = {
        name: _build_section(name, tree.pop(name, {}) or {}) for name in _SECTIONS
    }
    loss = tree.pop("loss", None) or {"bce": True}
    if not isinstance(loss, dict):
        raise ValueError("The loss section must be a mapping.")

    top_level = set(RunConfig.__dataclass_fields__) - set(_SECTIONS) - {"loss"}
    unknown = set(tree) - top_level
    if unknown:
        raise KeyError(f"Unknown top-level config keys: {sorted(unknown)}")

    return RunConfig(loss=loss, **sections, **tree)


def add_config_arguments(parser) -> None:
    """Attach ``--config`` / ``--set`` to an ``argparse`` parser."""
    parser.add_argument("--config", default=None, help="Path to a YAML config file.")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="Dotted config overrides, e.g. train.steps=2000 loss.focal=true",
    )
