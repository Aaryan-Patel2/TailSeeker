"""Training entrypoint for TailSeeker.

Usage:
    python -m tailseeker.run --config config/default.yaml --seed 42
"""

import argparse
from pathlib import Path

import yaml
from dotmap import DotMap

from tailseeker.models.model import get_model
from tailseeker.utils.utils import set_seed


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments (non-research parameters only)."""
    parser = argparse.ArgumentParser(description="TailSeeker training")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    return parser.parse_args()


def main() -> None:
    """Run training from the command line."""
    args = _parse_args()
    set_seed(args.seed)  # must be the first substantive call after arg parsing

    config_path = Path(args.config)
    assert config_path.exists(), f"Config file not found: {config_path}"
    with open(config_path) as f:
        config = DotMap(yaml.safe_load(f))

    # Always go through get_model — never instantiate a model class directly
    model = get_model(config)

    # TODO: build TailSeekerDataModule and TailSeekerModule, then call trainer.fit
    raise NotImplementedError("main() training loop not yet implemented")


if __name__ == "__main__":
    main()
