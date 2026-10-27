from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ldpc_corrector.ablate import DEFAULT_SWEEP, load_sweep, resolve_variant
from ldpc_corrector.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SHIPPED_CONFIGS = ["gcn", "gat", "local_gt", "sparse_global_gt"]


def test_defaults_load_without_a_file() -> None:
    config = load_config(None, [])
    assert config.model.corrector == "gcn"
    assert config.bp.iterations == 5
    assert config.bp.freeze is True  # no gradient through BP by default
    assert config.loss == {"bce": True}


@pytest.mark.parametrize("name", SHIPPED_CONFIGS)
def test_every_shipped_config_builds_a_working_model(name, ldpc49_code) -> None:
    from ldpc_corrector.train import build_model

    _, parity_check = ldpc49_code
    config = load_config(CONFIG_DIR / f"{name}.yaml")
    assert config.name == name
    model = build_model(config, parity_check)
    assert model.corrector_name == config.model.corrector
    assert model.corrector.num_parameters() > 0
    assert model.estimate_flops_per_frame() > 0


def test_dotted_overrides_reach_nested_sections() -> None:
    config = load_config(
        None,
        [
            "train.steps=7",
            "bp.iterations=9",
            "model.hidden_dim=64",
            "eval.ebn0=[1.0, 2.5]",
            "loss.focal=true",
            "loss.focal_gamma=3.0",
        ],
    )
    assert config.train.steps == 7
    assert config.bp.iterations == 9
    assert config.model.hidden_dim == 64
    assert config.eval.ebn0 == [1.0, 2.5]
    assert config.loss == {"focal": True, "focal_gamma": 3.0}


def test_unknown_model_keys_become_architecture_specific_extras() -> None:
    """ModelConfig stays shared; per-architecture knobs pass straight through."""
    config = load_config(None, ["model.top_k=12", "model.hop_count=2"])
    assert config.model.extra == {"top_k": 12, "hop_count": 2}


def test_unknown_keys_elsewhere_are_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown keys in config section 'train'"):
        load_config(None, ["train.nonsense=1"])
    with pytest.raises(KeyError, match="Unknown top-level config keys"):
        load_config(None, ["nonsense=1"])
    with pytest.raises(ValueError, match="key.subkey=value"):
        load_config(None, ["train.steps"])


def test_config_round_trips_through_json() -> None:
    import json

    config = load_config(CONFIG_DIR / "gat.yaml")
    restored = json.loads(config.to_json())
    assert restored["model"]["corrector"] == "gat"
    assert restored["bp"]["iterations"] == 5


def test_shipped_sweep_resolves_every_variant() -> None:
    sweep = load_sweep(DEFAULT_SWEEP)
    common = sweep.get("common") or []
    names = set()
    for variant in sweep["variants"]:
        config = resolve_variant(variant, common, DEFAULT_SWEEP.parent)
        assert config.name == variant["name"]
        assert config.name not in names, "duplicate variant name"
        names.add(config.name)
        # The variant's own overrides must win over the common block.
        assert config.train.steps == 1500
    assert len(names) == len(sweep["variants"])


def test_sweep_covers_every_ablation_axis_in_the_spec() -> None:
    sweep = load_sweep(DEFAULT_SWEEP)
    overrides = " ".join(
        " ".join(v.get("overrides") or []) for v in sweep["variants"]
    )
    for axis in (
        "model.edge_exclusion=true",
        "model.edge_exclusion=false",
        "model.use_gate=true",
        "model.fixed_gate=",
        "model.hop_count=1",
        "model.hop_count=2",
        "model.top_k=",
        "model.threshold=",
        "loss.hard_mining=true",
        "loss.block_soft_bler=true",
    ):
        assert axis in overrides, f"sweep is missing the {axis} ablation"


def test_unknown_variant_name_is_rejected() -> None:
    from ldpc_corrector.ablate import run_sweep

    with pytest.raises(KeyError, match="Unknown variant names"):
        run_sweep(DEFAULT_SWEEP, only=["does_not_exist"], dry_run=True)


def test_sweep_without_variants_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"common": []}))
    with pytest.raises(ValueError, match="must define a 'variants' list"):
        load_sweep(path)


def test_only_runs_variants_in_the_order_given() -> None:
    """--only is a priority list: a truncated sweep must still cover the
    variants the caller put first."""
    from ldpc_corrector.ablate import load_sweep, resolve_variant

    sweep = load_sweep(DEFAULT_SWEEP)
    common = sweep.get("common") or []
    by_name = {v["name"]: v for v in sweep["variants"]}

    # Deliberately reversed relative to the file order.
    file_order = [v["name"] for v in sweep["variants"]]
    requested = [file_order[-1], file_order[0]]
    assert requested != sorted(requested, key=file_order.index)

    ordered, seen = [], set()
    for name in requested:
        if name not in seen:
            seen.add(name)
            ordered.append(by_name[name])
    assert [v["name"] for v in ordered] == requested
    # And each still resolves to a usable config.
    for variant in ordered:
        assert resolve_variant(variant, common, DEFAULT_SWEEP.parent).name == variant["name"]


def test_only_deduplicates_repeated_names() -> None:
    from ldpc_corrector.ablate import run_sweep

    # Should not raise, and should not run a variant twice.
    assert run_sweep(DEFAULT_SWEEP, only=["edge_gat", "edge_gat"], dry_run=True) is None
