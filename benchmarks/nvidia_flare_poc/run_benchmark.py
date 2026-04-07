#!/usr/bin/env python3
"""NVFlare POC benchmark runner.

This runner mirrors the structure used by the simulation benchmarks: it loads
the shared project configuration, prints the benchmark settings, prepares the
runtime workspace, and saves a small manifest for later analysis.

Unlike the simulation runners, the actual federated execution happens through
the official NVFlare POC workflow. The role of this script is to keep the POC
workspace inside the repository and expose the exact commands needed to start,
stop, and clean the services.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_WORKSPACE = PROJECT_ROOT / "nvflare_poc_workspace"
DEFAULT_JOBS_DIR = Path(__file__).resolve().parent / "jobs"
DEFAULT_MANIFEST_PATH = RESULTS_DIR / "nvidia_flare_poc_manifest.json"


def load_project_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load the shared benchmark configuration."""

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_nvflare_cli() -> str:
    """Return the `nvflare` executable path or fail with a clear message."""

    cli = shutil.which("nvflare")
    if cli:
        return cli

    raise RuntimeError(
        "Comando 'nvflare' non trovato nel PATH. Attiva il virtualenv corretto prima di eseguire il benchmark."
    )


def run_command(command: list[str], cwd: Path | None = None) -> None:
    """Run a subprocess command after printing it for traceability."""

    print("$", " ".join(command))
    subprocess.run(command, cwd=str(cwd or PROJECT_ROOT), check=True)


def print_banner(config: dict, workspace: Path) -> None:
    """Print the benchmark header in the same style as the simulation runners."""

    dataset = config.get("dataset", {})
    federated = config.get("federated", {})
    xgboost = config.get("xgboost", {})
    hardware = config.get("hardware", {})

    print("\n" + "=" * 70)
    print("NVIDIA FLARE POC BENCHMARK")
    print("NECSTLab - Polimi LS2")
    print("=" * 70)
    print(f"Workspace: {workspace}")
    print(f"Clients: {dataset.get('num_clients', 9)}")
    print(f"Rounds: {federated.get('num_rounds', 10)}")
    print(f"Test fraction: {dataset.get('test_fraction', 0.2)}")
    print(f"Tree method: {xgboost.get('tree_method', 'hist')}")
    print(f"Use GPU: {hardware.get('use_gpu', False)}")
    print()


def ensure_directories(workspace: Path) -> None:
    """Create the local workspace and results directories if needed."""

    workspace.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def prepare_workspace(workspace: Path, jobs_dir: Path, num_clients: int) -> None:
    """Prepare the NVFlare POC workspace through the official CLI."""

    os.environ["NVFLARE_POC_WORKSPACE"] = str(workspace)

    cli = resolve_nvflare_cli()
    run_command([cli, "config", "-pw", str(workspace)])
    run_command([cli, "poc", "prepare", "-n", str(num_clients)])

    if jobs_dir.exists() and any(jobs_dir.iterdir()):
        run_command([cli, "poc", "prepare-jobs-dir", "-j", str(jobs_dir)])
    else:
        print(f"Jobs directory non pronta: {jobs_dir}")
        print("Aggiungi i job NVFlare in questa cartella e rilancia prepare-jobs-dir.")


def start_poc(service: str | None = None, exclude: str | None = None, gpu: list[str] | None = None) -> None:
    """Start NVFlare POC services."""

    command = [resolve_nvflare_cli(), "poc", "start"]
    if service:
        command += ["-p", service]
    if exclude:
        command += ["-ex", exclude]
    if gpu:
        command += ["-gpu", *gpu]
    run_command(command)


def stop_poc(service: str | None = None, exclude: str | None = None) -> None:
    """Stop NVFlare POC services."""

    command = [resolve_nvflare_cli(), "poc", "stop"]
    if service:
        command += ["-p", service]
    if exclude:
        command += ["-ex", exclude]
    run_command(command)


def clean_poc() -> None:
    """Clean the POC workspace."""

    run_command([resolve_nvflare_cli(), "poc", "clean"])


def write_manifest(config: dict, workspace: Path, jobs_dir: Path, elapsed: float, manifest_path: Path) -> Path:
    """Write a small JSON manifest describing the prepared run."""

    dataset = config.get("dataset", {})
    federated = config.get("federated", {})
    xgboost = config.get("xgboost", {})
    hardware = config.get("hardware", {})

    manifest = {
        "framework": "nvidia_flare",
        "mode": "poc",
        "workspace": str(workspace),
        "jobs_dir": str(jobs_dir),
        "prepared_in_seconds": elapsed,
        "dataset": {
            "name": dataset.get("name", "garmin_sleep_quality"),
            "num_clients": int(dataset.get("num_clients", 9)),
            "test_fraction": float(dataset.get("test_fraction", 0.2)),
        },
        "federated": {
            "num_rounds": int(federated.get("num_rounds", 10)),
            "local_epochs": int(federated.get("local_epochs", 5)),
        },
        "xgboost": {
            "objective": xgboost.get("objective", "reg:squarederror"),
            "max_depth": int(xgboost.get("max_depth", 6)),
            "learning_rate": float(xgboost.get("learning_rate", 0.1)),
            "subsample": float(xgboost.get("subsample", 0.8)),
            "colsample_bytree": float(xgboost.get("colsample_bytree", 0.8)),
        },
        "hardware": {
            "use_gpu": bool(hardware.get("use_gpu", False)),
            "num_cpus": int(hardware.get("num_cpus", 8)),
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def print_launch_commands(num_clients: int) -> None:
    """Print the POC commands that can be used to launch the benchmark."""

    print("Comandi NVFlare POC da eseguire dopo prepare:")
    print("  nvflare poc start -p server")
    for index in range(1, num_clients + 1):
        print(f"  nvflare poc start -p site-{index}")
    print("  nvflare poc start -p admin@nvidia.com")
    print()
    print("Per lanciare tutto in una volta:")
    print("  nvflare poc start")
    print("Per fermare i servizi:")
    print("  nvflare poc stop")
    print("Per pulire il workspace:")
    print("  nvflare poc clean")
    print()


def run_benchmark(workspace: Path, jobs_dir: Path, manifest_path: Path, config: dict) -> Path:
    """Prepare the NVFlare POC workspace and save the manifest."""

    num_clients = int(config.get("dataset", {}).get("num_clients", 9))

    print_banner(config, workspace)

    start_total = time.time()
    ensure_directories(workspace)
    prepare_workspace(workspace, jobs_dir, num_clients)
    elapsed = time.time() - start_total

    manifest = write_manifest(config, workspace, jobs_dir, elapsed, manifest_path)

    print("\n" + "=" * 70)
    print("POC PREPARATO")
    print("=" * 70)
    print(f"Tempo preparazione: {elapsed:.2f}s")
    print(f"Manifest: {manifest}")
    print()
    print_launch_commands(num_clients)

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="NVFlare POC benchmark runner")
    parser.add_argument(
        "command",
        nargs="?",
        default="benchmark",
        choices=["benchmark", "prepare", "start", "stop", "clean"],
        help="Operazione da eseguire",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help="Percorso del workspace POC locale",
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=DEFAULT_JOBS_DIR,
        help="Directory contenente i job NVFlare da collegare al POC",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="File JSON in cui salvare il riepilogo del run",
    )
    parser.add_argument(
        "--service",
        type=str,
        default=None,
        help="Participant/servizio specifico per start/stop, ad esempio server o admin@nvidia.com",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Servizio da escludere in start/stop, ad esempio admin",
    )
    parser.add_argument(
        "--gpu",
        nargs="*",
        default=None,
        help="ID GPU da passare a nvflare poc start",
    )

    args = parser.parse_args()

    try:
        config = load_project_config()
        if args.command == "benchmark":
            run_benchmark(args.workspace, args.jobs_dir, args.manifest, config)
        elif args.command == "prepare":
            ensure_directories(args.workspace)
            prepare_workspace(args.workspace, args.jobs_dir, int(config.get("dataset", {}).get("num_clients", 9)))
            write_manifest(config, args.workspace, args.jobs_dir, 0.0, args.manifest)
        elif args.command == "start":
            start_poc(args.service, args.exclude, args.gpu)
        elif args.command == "stop":
            stop_poc(args.service, args.exclude)
        elif args.command == "clean":
            clean_poc()
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"Comando NVFlare fallito con exit code {exc.returncode}")
        return exc.returncode
    except Exception as exc:
        print(f"Errore: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())