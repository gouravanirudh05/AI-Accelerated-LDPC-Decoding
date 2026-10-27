"""Package a corrector experiment as a Kaggle GPU kernel.

    python ldpc_corrector/kaggle/build_kernel.py --username <kaggle-user> \
        [--experiment cnn|sparse] [--token-file ~/.kaggle/access_token_gpu] --push

``--experiment`` picks the job: ``cnn`` (``run_cnn_corrector.py``, the ResNet
correctors) or ``sparse`` (``run_sparse_followup.py``, gate / FER loss /
resumed BP on the sparse-global corrector). Bundles ``ldpc_corrector`` (minus
tests), the four ``GAT+BP`` files the BP decoder needs, and the ECCT/AECCT
reference CSV into that script, writes ``kernel-metadata.json`` next to it
(private, GPU, internet for ``pip install torch_geometric``), and optionally
pushes it. Credentials are the Kaggle CLI's own (``~/.kaggle/access_token``
or ``KAGGLE_API_TOKEN``); ``--token-file`` selects another account.

Then::

    kaggle kernels status <user>/<slug>
    kaggle kernels output <user>/<slug> -p <dir>
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "ldpc_corrector"
# experiment -> (runner script, kernel title, kernel slug)
EXPERIMENTS = {
    "cnn": ("run_cnn_corrector.py", "LDPC ResNet Corrector", "ldpc-resnet-corrector"),
    "sparse": ("run_sparse_followup.py", "LDPC Sparse Corrector Followup",
               "ldpc-sparse-corrector-followup"),
}
GAT_BP_FILES = ("gat_bp.py", "Codes.py", "train_gat_bp.py", "Codes_DB/LDPC_N49_K24.alist")
REFERENCE_CSV = "comparison/data/test_results.csv"


def gat_bp_dir() -> Path:
    """Same lookup as ``ldpc_corrector._compat``."""
    return Path(os.environ.get("LDPC_GAT_BP_DIR", REPO_ROOT / "GAT+BP"))


def reference_csv() -> Optional[bytes]:
    """The ECCT/AECCT overlay, from the working tree or else from git HEAD."""
    path = REPO_ROOT / REFERENCE_CSV
    if path.is_file():
        return path.read_bytes()
    result = subprocess.run(
        ["git", "show", f"HEAD:{REFERENCE_CSV}"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else None


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def build_bundle() -> bytes:
    source_dir = gat_bp_dir()
    missing = [f for f in GAT_BP_FILES if not (source_dir / f).is_file()]
    if missing:
        raise SystemExit(
            f"{source_dir} lacks {missing}. Point LDPC_GAT_BP_DIR at a GAT+BP "
            "copy whose gat_bp.py supports return_trace."
        )
    if "return_trace" not in (source_dir / "gat_bp.py").read_text():
        raise SystemExit(
            f"{source_dir}/gat_bp.py predates return_trace; ldpc_corrector "
            "cannot run on it."
        )

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted(PACKAGE.rglob("*")):
            relative = path.relative_to(PACKAGE)
            if not path.is_file() or relative.parts[0] in ("tests", "kaggle"):
                continue
            if "__pycache__" in relative.parts or path.suffix not in (".py", ".yaml"):
                continue
            tar.add(path, arcname=f"ldpc_corrector/{relative}")
        for name in GAT_BP_FILES:
            tar.add(source_dir / name, arcname=f"GAT+BP/{name}")
        csv = reference_csv()
        if csv is not None:
            _add_bytes(tar, "comparison/test_results.csv", csv)
        else:
            print("warning: no ECCT/AECCT reference CSV; the overlay is skipped")
    return buffer.getvalue()


def build(out_dir: Path, username: str, experiment: str = "cnn") -> Path:
    runner, title, slug = EXPERIMENTS[experiment]
    template = Path(__file__).with_name(runner)
    bundle = build_bundle()
    script = template.read_text().replace(
        '"@@BUNDLE@@"', repr(base64.b64encode(bundle).decode("ascii")), 1
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / template.name).write_text(script)
    metadata = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": template.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    print(
        f"bundle {len(bundle) / 1024:.0f} KiB -> {out_dir / template.name} "
        f"({len(script) / 1024:.0f} KiB)"
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--username",
        default=os.environ.get("KAGGLE_USERNAME"),
        help="Kaggle username that owns the kernel (or set KAGGLE_USERNAME).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "Results_Corrector" / "kaggle_kernel"),
    )
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="cnn")
    parser.add_argument("--push", action="store_true", help="Run `kaggle kernels push`.")
    parser.add_argument("--token-file", default=None,
                        help="Kaggle access-token file for --push (selects the account).")
    parser.add_argument(
        "--accelerator",
        default=None,
        help="Override the GPU type on push, e.g. NvidiaTeslaT4 or NvidiaTeslaP100.",
    )
    args = parser.parse_args()
    if not args.username:
        parser.error("--username (or KAGGLE_USERNAME) is required for the kernel id.")

    out_dir = build(Path(args.out_dir), args.username, args.experiment)
    slug = EXPERIMENTS[args.experiment][2]
    if args.push:
        env = dict(os.environ)
        if args.token_file:
            env["KAGGLE_API_TOKEN"] = str(Path(args.token_file).expanduser())
        kaggle = Path(sys.executable).with_name("kaggle")
        command = [str(kaggle if kaggle.exists() else "kaggle"), "kernels", "push", "-p", str(out_dir)]
        if args.accelerator:
            command += ["--accelerator", args.accelerator]
        subprocess.run(command, env=env, check=True)
        print(f"\nstatus: kaggle kernels status {args.username}/{slug}")
        print(f"output: kaggle kernels output {args.username}/{slug} -p <dir>")


if __name__ == "__main__":
    main()
