"""Main entry point for autograd code generation.

Usage:
    python -m tools.autograd.gen_autograd [--yaml PATH] [--output-dir PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(yaml_path: str | Path, output_dir: str | Path) -> None:
    from .load_derivatives import load_derivatives
    from .gen_functions import gen_functions
    from .gen_variable_type import gen_variable_type
    from .gen_registration import gen_registration

    yaml_path = Path(yaml_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    infos = load_derivatives(yaml_path)
    print(f"Loaded {len(infos)} derivative entries from {yaml_path}")

    # Generate files
    files = {
        "functions.py": gen_functions(infos),
        "variable_type.py": gen_variable_type(infos),
        "registration.py": gen_registration(infos),
    }

    for name, content in files.items():
        dest = output_dir / name
        # Only write if content changed
        if dest.exists() and dest.read_text() == content:
            print(f"  {name} — unchanged")
            continue
        dest.write_text(content)
        print(f"  {name} — written ({len(content)} bytes)")

    # Write __init__.py if missing
    init = output_dir / "__init__.py"
    if not init.exists():
        init.write_text("# Auto-generated package\n")
        print("  __init__.py — created")


def cli():
    parser = argparse.ArgumentParser(description="Generate autograd code from derivatives.yaml")
    parser.add_argument(
        "--yaml",
        default=str(Path(__file__).parent / "derivatives.yaml"),
        help="Path to derivatives.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent.parent / "src" / "candle" / "_generated"),
        help="Output directory for generated files",
    )
    args = parser.parse_args()
    main(args.yaml, args.output_dir)


if __name__ == "__main__":
    cli()
