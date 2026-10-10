"""Create a separate, enabled SeaS policy with pinned backend source files."""
import argparse
from pathlib import Path
import sys
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from detected_pipeline.augmentation.bounded import validate_policy
from detected_pipeline.util import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--service", type=Path, required=True)
    p.add_argument("--reuse-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-unreviewed", action="store_true", help="Permit structurally valid pseudo masks for experiments")
    args = p.parse_args()
    config = yaml.safe_load((PROJECT / "configs/synthetic.yaml").read_text(encoding="utf-8"))
    config.update(enabled=True, command=[str(args.python.resolve()), str(args.service.resolve())],
                  reuse_root=str(args.reuse_root.resolve()), allow_structurally_valid_unreviewed=args.allow_unreviewed,
                  backend_sha256={str(f.resolve()): sha256_file(f) for f in args.service.parent.glob("*.py")})
    validate_policy(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
