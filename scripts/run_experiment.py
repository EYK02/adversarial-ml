import argparse

from src.utils.config import load_experiment, resolve_root_paths
from src.runner.experiment_builders import build_experiments
from src.runner.core import run_stages


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--stage", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()

    # Load config
    cfg = load_experiment(
        args.config,
        dry_run=args.dry_run,
        smoke_test=args.smoke_test,
        run_name=args.run_name,
    )
    cfg = resolve_root_paths(cfg)

    # Run full experiment pipeline
    run_stages(
        cfg,
        stage=args.stage,
        dry_run=args.dry_run,
        smoke_test=args.smoke_test,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()