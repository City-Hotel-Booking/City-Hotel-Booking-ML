"""Shared command entry point. Task owners implement the unfinished runners."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RUNNERS = {
    "validate": "src.data.validate",
    "analyze": "src.data.eda",
    "prepare": "src.data.split",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="City Hotel H2 project: Milestone 1 command interface."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Configuration path relative to the project."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="Check the H2 source data (not implemented).")
    commands.add_parser("analyze", help="Produce EDA summaries and plots (not implemented).")
    commands.add_parser("prepare", help="Create the agreed split manifest (not implemented).")
    baseline = commands.add_parser("baseline", help="Run one baseline (not implemented).")
    baseline.add_argument(
        "--task", required=True, choices=("regression", "classification", "clustering")
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    from src.settings import load_config

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    try:
        config = load_config(config_path)
        module_name = (
            f"src.{args.task}.baseline" if args.command == "baseline" else RUNNERS[args.command]
        )
        import_module(module_name).run(config, PROJECT_ROOT)
    except (NotImplementedError, FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

