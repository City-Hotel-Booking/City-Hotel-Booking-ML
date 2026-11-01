"""Shared command entry point. Task owners implement the unfinished runners."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RUNNERS = {
    "validate": "src.data.validate",
    "analyze": "src.data.eda",
    "prepare": "src.data.prepare",
}
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="City Hotel H2 project: Milestone 1 command interface."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Configuration path relative to the project."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="Check H2 source data and save validation evidence.")
    commands.add_parser("analyze", help="Produce EDA tables, plots and unusual-record review.")
    commands.add_parser("prepare", help="Create the grouped, stratified split manifest.")
    cluster = commands.add_parser("cluster", help="Run/view complete clustering analysis or assign bookings.")
    cluster.add_argument("arguments", nargs=argparse.REMAINDER)
    baseline = commands.add_parser("baseline", help="Run the task baseline.")
    baseline.add_argument(
        "--task", required=True, choices=("regression", "classification", "clustering")
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "cluster":
        from src.clustering.complete import main as clustering_main
        try:
            clustering_main(["--config", args.config, *args.arguments])
        except (ValueError, FileNotFoundError) as exc:
            parser.exit(2, f"{exc}\n")
        return 0
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
