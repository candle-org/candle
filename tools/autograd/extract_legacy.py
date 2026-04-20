#!/usr/bin/env python3
"""Extract legacy (hand-edited) autograd wrappers from generated files.

This script separates variable_type.py and functions.py into:
  - Generated ops (from derivatives.yaml) -> stay in variable_type.py / functions.py
  - Legacy ops (hand-edited, NOT in yaml) -> move to variable_type_legacy.py / functions_legacy.py

Usage:
    python -m tools.autograd.extract_legacy [--generated-dir PATH]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def _get_generated_names():
    """Load derivatives.yaml and compute the sets of generated names.

    Returns:
        vt_gen_def_names: function names that the codegen produces as actual `def` blocks
        vt_gen_alias_names: function names that the codegen produces as one-line aliases
        fn_gen_class_names: class names that the codegen produces
        fn_gen_alias_names: class alias names that the codegen produces
        infos: the loaded DifferentiabilityInfo entries
    """
    from .load_derivatives import load_derivatives

    yaml_path = Path(__file__).parent / "derivatives.yaml"
    infos = load_derivatives(yaml_path)

    # Variable_type.py: names generated as actual def blocks
    vt_gen_def_names = set()
    for info in infos:
        vt_gen_def_names.add(f"{info.generated_func_stem}_autograd")
        vt_gen_def_names.add(f"{info.generated_func_stem}_autograd_post")

    # Variable_type.py: names generated as one-line aliases
    vt_gen_alias_names = set()
    seen_ops = set()
    for info in infos:
        if info.op_name in seen_ops:
            continue
        seen_ops.add(info.op_name)
        canonical = f"{info.op_name}_autograd"
        canonical_post = f"{info.op_name}_autograd_post"
        specific = f"{info.generated_func_stem}_autograd"
        specific_post = f"{info.generated_func_stem}_autograd_post"
        if canonical != specific:
            vt_gen_alias_names.add(canonical)
        if canonical_post != specific_post:
            vt_gen_alias_names.add(canonical_post)

    # Functions.py: generated class names
    fn_gen_class_names = set()
    for info in infos:
        fn_gen_class_names.add(info.backward_name)

    # Functions.py: canonical class aliases
    seen_ops = set()
    fn_gen_alias_names = set()
    for info in infos:
        if info.op_name in seen_ops:
            continue
        seen_ops.add(info.op_name)
        canonical = f"{info.op_name.capitalize()}Backward0"
        if canonical != info.backward_name:
            fn_gen_alias_names.add(canonical)

    return vt_gen_def_names, vt_gen_alias_names, fn_gen_class_names, fn_gen_alias_names, infos


def _split_variable_type(content: str, gen_def_names: set[str], gen_alias_names: set[str]):
    """Split variable_type.py content into generated and legacy parts.

    A def block is "generated" only if its name is in gen_def_names (the actual
    def blocks that the codegen emits).  A def with a name that is normally an
    alias (in gen_alias_names) but appears as a hand-written def is legacy.

    Returns (header, gen_defs, gen_aliases, legacy_defs, legacy_aliases).
    """
    lines = content.split("\n")

    # Find the header: everything before the first 'def '
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith("def "):
            header_end = i
            break

    header_lines = lines[:header_end]

    # Parse the rest into blocks: each block is a def or an alias line or a comment section
    rest = "\n".join(lines[header_end:])

    # All generated names (defs + aliases) for classifying alias lines
    all_gen_names = gen_def_names | gen_alias_names

    blocks = _split_into_def_blocks(rest)

    gen_defs = []
    legacy_defs = []
    gen_other = []  # aliases, comments for generated section
    legacy_other = []  # aliases, comments for legacy section

    in_legacy_section = False

    for block_type, block_text in blocks:
        if block_type == "def":
            # Extract function name
            m = re.match(r"def\s+(\w+)\s*\(", block_text)
            func_name = m.group(1) if m else ""
            # Only classify as generated if it's an actual def (not an alias override)
            if func_name in gen_def_names:
                gen_defs.append(block_text)
            else:
                legacy_defs.append(block_text)
        elif block_type == "alias":
            # Alias line like: `name = other_name`
            # Check if the LHS is a generated alias
            m = re.match(r"(\w+)\s*=\s*(\w+)", block_text.strip())
            if m:
                lhs = m.group(1)
                if lhs in all_gen_names:
                    gen_other.append(block_text)
                else:
                    legacy_other.append(block_text)
            else:
                # Non-alias line (comment, blank, etc.)
                if "UPSTREAM LEGACY" in block_text or "LEGACY" in block_text.upper():
                    in_legacy_section = True
                if in_legacy_section:
                    legacy_other.append(block_text)
                else:
                    gen_other.append(block_text)
        elif block_type == "comment":
            if "UPSTREAM LEGACY" in block_text or in_legacy_section:
                in_legacy_section = True
                # Don't include the legacy section markers in either file
            else:
                gen_other.append(block_text)

    return header_lines, gen_defs, gen_other, legacy_defs, legacy_other


def _split_into_def_blocks(text: str):
    """Split text into blocks of (type, content).

    type is one of: "def", "alias", "comment"
    """
    lines = text.split("\n")
    blocks = []
    current_block = []
    current_type = None
    i = 0

    while i < len(lines):
        line = lines[i]

        if line.startswith("def "):
            # Save any pending block
            if current_block:
                blocks.append((current_type, "\n".join(current_block)))
                current_block = []

            # Start a new def block
            current_type = "def"
            current_block = [line]
            i += 1
            # Continue until next top-level def, or a top-level non-indented line
            # that's not part of this function
            while i < len(lines):
                next_line = lines[i]
                # A new def at column 0 starts a new block
                if next_line.startswith("def "):
                    break
                # A top-level assignment (alias) or comment at column 0
                # that isn't indented (part of the function body)
                if next_line and not next_line[0].isspace() and not next_line.startswith("def "):
                    # Could be a decorator or continuation — check if it's an assignment
                    if re.match(r"\w+\s*=\s*\w+", next_line) or next_line.startswith("#") or next_line.startswith("class "):
                        break
                current_block.append(next_line)
                i += 1

            blocks.append(("def", "\n".join(current_block)))
            current_block = []
            current_type = None
        elif re.match(r"\w+\s*=\s*\w+", line):
            # Alias line
            if current_block and current_type != "alias":
                blocks.append((current_type, "\n".join(current_block)))
                current_block = []
            current_type = "alias"
            blocks.append(("alias", line))
            current_block = []
            current_type = None
            i += 1
        elif line.startswith("#"):
            # Comment line
            if current_block and current_type not in ("comment", None):
                blocks.append((current_type, "\n".join(current_block)))
                current_block = []
            current_type = "comment"
            current_block.append(line)
            i += 1
        elif not line.strip():
            # Blank line
            if current_block:
                blocks.append((current_type, "\n".join(current_block)))
                current_block = []
                current_type = None
            i += 1
        else:
            # Some other top-level code
            if current_block:
                blocks.append((current_type, "\n".join(current_block)))
                current_block = []
            current_type = "alias"
            blocks.append(("alias", line))
            current_block = []
            current_type = None
            i += 1

    if current_block:
        blocks.append((current_type, "\n".join(current_block)))

    return blocks


def _split_functions(content: str, gen_class_names: set[str], gen_alias_names: set[str]):
    """Split functions.py content into generated and legacy parts.

    Returns (header_text, gen_classes, gen_other, legacy_classes, legacy_other).
    header_text includes everything before the first class definition including
    all helper functions.
    """
    lines = content.split("\n")

    # Find the first class definition
    first_class_line = 0
    for i, line in enumerate(lines):
        if line.startswith("class ") and "Backward0" in line:
            first_class_line = i
            break

    # The header is everything before the first Backward0 class
    # This includes imports, helper functions, and helper classes like _ConvGradCache
    header_text = "\n".join(lines[:first_class_line])

    rest = "\n".join(lines[first_class_line:])

    # Split rest into class blocks, alias lines, and helper function blocks
    blocks = _split_into_class_blocks(rest)

    gen_classes = []
    legacy_classes = []
    gen_other = []
    legacy_other = []
    gen_helper_defs = []
    legacy_helper_defs = []

    in_legacy_section = False

    for block_type, block_text in blocks:
        if block_type == "class":
            # Extract class name
            m = re.match(r"class\s+(\w+)\s*\(", block_text)
            class_name = m.group(1) if m else ""
            if class_name in gen_class_names:
                gen_classes.append(block_text)
            else:
                legacy_classes.append(block_text)
        elif block_type == "def":
            # Helper function between classes — these are used by generated nodes
            # Include them in the header of whichever file needs them
            # For now, keep them in the generated file since they're called by
            # generated backward formulas
            if in_legacy_section:
                legacy_helper_defs.append(block_text)
            else:
                gen_helper_defs.append(block_text)
        elif block_type == "alias":
            m = re.match(r"(\w+)\s*=\s*(\w+)", block_text.strip())
            if m:
                lhs = m.group(1)
                if lhs in gen_class_names or lhs in gen_alias_names:
                    gen_other.append(block_text)
                else:
                    legacy_other.append(block_text)
            else:
                if in_legacy_section:
                    legacy_other.append(block_text)
                else:
                    gen_other.append(block_text)
        elif block_type == "comment":
            if "UPSTREAM LEGACY" in block_text:
                in_legacy_section = True
                # Don't include legacy section markers
            elif in_legacy_section:
                legacy_other.append(block_text)
            else:
                gen_other.append(block_text)

    return (header_text, gen_classes, gen_other, gen_helper_defs,
            legacy_classes, legacy_other, legacy_helper_defs)


def _split_into_class_blocks(text: str):
    """Split text into blocks of (type, content) for functions.py.

    type is one of: "class", "def", "alias", "comment"
    """
    lines = text.split("\n")
    blocks = []
    current_block = []
    current_type = None
    i = 0

    while i < len(lines):
        line = lines[i]

        if line.startswith("class "):
            # Save any pending block
            if current_block:
                blocks.append((current_type, "\n".join(current_block)))
                current_block = []

            # Start a new class block
            current_type = "class"
            current_block = [line]
            i += 1
            # Continue until next top-level class, def, or non-indented identifier
            while i < len(lines):
                next_line = lines[i]
                if next_line.startswith("class ") or next_line.startswith("def "):
                    break
                # A top-level alias/comment at column 0
                if next_line and not next_line[0].isspace() and not next_line.startswith("#"):
                    if re.match(r"\w+\s*=\s*\w+", next_line):
                        break
                    if re.match(r"# ===", next_line):
                        break
                current_block.append(next_line)
                i += 1

            blocks.append(("class", "\n".join(current_block)))
            current_block = []
            current_type = None

        elif line.startswith("def "):
            # Save any pending block
            if current_block:
                blocks.append((current_type, "\n".join(current_block)))
                current_block = []

            # Start a new def block
            current_type = "def"
            current_block = [line]
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if next_line.startswith("def ") or next_line.startswith("class "):
                    break
                if next_line and not next_line[0].isspace() and not next_line.startswith("#"):
                    if re.match(r"\w+\s*=\s*\w+", next_line) or re.match(r"# ===", next_line):
                        break
                current_block.append(next_line)
                i += 1

            blocks.append(("def", "\n".join(current_block)))
            current_block = []
            current_type = None

        elif re.match(r"\w+\s*=\s*\w+", line):
            if current_block:
                blocks.append((current_type, "\n".join(current_block)))
                current_block = []
            blocks.append(("alias", line))
            current_type = None
            i += 1

        elif line.startswith("# ===") or line.startswith("# ---") or line.startswith("# Canonical"):
            if current_block:
                blocks.append((current_type, "\n".join(current_block)))
                current_block = []
            current_type = "comment"
            current_block = [line]
            i += 1

        elif line.startswith("#"):
            if current_type == "comment":
                current_block.append(line)
            else:
                if current_block:
                    blocks.append((current_type, "\n".join(current_block)))
                current_type = "comment"
                current_block = [line]
            i += 1

        elif not line.strip():
            # Blank line
            if current_block and current_type == "comment":
                blocks.append((current_type, "\n".join(current_block)))
                current_block = []
                current_type = None
            i += 1

        else:
            if current_block:
                blocks.append((current_type, "\n".join(current_block)))
                current_block = []
            blocks.append(("alias", line))
            current_type = None
            i += 1

    if current_block:
        blocks.append((current_type, "\n".join(current_block)))

    return blocks


def _build_vt_file(header_lines, defs, other_lines) -> str:
    """Build variable_type.py content from parts."""
    parts = ["\n".join(header_lines)]
    parts.append("")  # blank line after header

    # Add def blocks
    for d in defs:
        parts.append("")
        parts.append(d)

    # Add alias section if any
    alias_lines = [line for line in other_lines if re.match(r"\w+\s*=\s*\w+", line.strip())]
    comment_lines = [line for line in other_lines if line.strip().startswith("#")]

    if alias_lines:
        parts.append("")
        parts.append("")
        if comment_lines:
            parts.extend(comment_lines)
        else:
            parts.append("# Canonical overload aliases")
        parts.extend(alias_lines)

    result = "\n".join(parts)
    # Clean up excessive blank lines
    result = re.sub(r"\n{4,}", "\n\n\n", result)
    if not result.endswith("\n"):
        result += "\n"
    return result


def _build_fn_file(header_text, classes, other_lines, helper_defs) -> str:
    """Build functions.py content from parts."""
    parts = [header_text]

    # Add helper function blocks that go between classes
    for d in helper_defs:
        parts.append("")
        parts.append(d)

    # Add class blocks
    for c in classes:
        parts.append("")
        parts.append(c)

    # Add alias section if any
    alias_lines = [line for line in other_lines if re.match(r"\w+\s*=\s*\w+", line.strip())]
    comment_lines = [line for line in other_lines if line.strip().startswith("#")]

    if alias_lines:
        parts.append("")
        parts.append("")
        if comment_lines:
            parts.extend(comment_lines)
        else:
            parts.append("# Canonical overload aliases")
        parts.extend(alias_lines)

    result = "\n".join(parts)
    result = re.sub(r"\n{4,}", "\n\n\n", result)
    if not result.endswith("\n"):
        result += "\n"
    return result


def _build_legacy_vt_file(header_lines, legacy_defs, legacy_aliases,
                          legacy_fn_class_names: set[str]) -> str:
    """Build variable_type_legacy.py with legacy forward wrappers.

    Args:
        legacy_fn_class_names: set of backward class names that live in
            functions_legacy.py (used to rewrite _F. references to _FL.).
    """
    # Use the same header but change the docstring and add legacy import
    new_header = list(header_lines)
    # Replace the first docstring
    for i, line in enumerate(new_header):
        if '"""Auto-generated forward autograd wrappers' in line:
            new_header[i] = '"""Legacy forward autograd wrappers — hand-maintained.'
            # Find closing triple-quote and update it
            for j in range(i + 1, len(new_header)):
                if 'Generated by' in new_header[j]:
                    new_header[j] = "These wrappers are NOT generated from derivatives.yaml."
                elif 'DO NOT EDIT' in new_header[j]:
                    new_header[j] = ""
                if '"""' in new_header[j] and j > i:
                    break
            break

    # Add import of functions_legacy for legacy backward nodes
    for i, line in enumerate(new_header):
        if line == "from . import functions as _F":
            new_header.insert(i + 1, "from . import functions_legacy as _FL")
            break

    parts = ["\n".join(new_header)]

    # Rewrite legacy defs: change _F.XxxBackward0 to _FL.XxxBackward0
    # for backward nodes that live in functions_legacy.py
    for d in legacy_defs:
        rewritten = d
        for cls_name in legacy_fn_class_names:
            rewritten = rewritten.replace(f"_F.{cls_name}", f"_FL.{cls_name}")
        parts.append("")
        parts.append(rewritten)

    # Legacy aliases
    alias_lines = [line for line in legacy_aliases if re.match(r"\w+\s*=\s*\w+", line.strip())]
    if alias_lines:
        parts.append("")
        parts.append("")
        parts.append("# Legacy aliases")
        parts.extend(alias_lines)

    result = "\n".join(parts)
    result = re.sub(r"\n{4,}", "\n\n\n", result)
    if not result.endswith("\n"):
        result += "\n"
    return result


def _build_legacy_fn_file(header_text, legacy_classes, legacy_aliases, legacy_helper_defs) -> str:
    """Build functions_legacy.py with legacy backward node classes."""
    # Update docstring
    new_header = header_text.replace(
        '"""Auto-generated backward Node classes \u2014 DO NOT EDIT.\n\nGenerated by tools/autograd/gen_functions.py from derivatives.yaml.\n"""',
        '"""Legacy backward Node classes \u2014 hand-maintained.\n\nThese classes are NOT generated from derivatives.yaml.\n"""'
    )

    parts = [new_header]

    for d in legacy_helper_defs:
        parts.append("")
        parts.append(d)

    for c in legacy_classes:
        parts.append("")
        parts.append(c)

    alias_lines = [line for line in legacy_aliases if re.match(r"\w+\s*=\s*\w+", line.strip())]
    if alias_lines:
        parts.append("")
        parts.append("")
        parts.append("# Legacy aliases")
        parts.extend(alias_lines)

    result = "\n".join(parts)
    result = re.sub(r"\n{4,}", "\n\n\n", result)
    if not result.endswith("\n"):
        result += "\n"
    return result


def main(generated_dir: str | Path | None = None) -> None:
    if generated_dir is None:
        generated_dir = Path(__file__).resolve().parent.parent.parent / "src" / "candle" / "_generated"
    generated_dir = Path(generated_dir)

    vt_gen_def_names, vt_gen_alias_names, fn_gen_class_names, fn_gen_alias_names, infos = _get_generated_names()

    print(f"Loaded {len(infos)} derivative entries")
    print(f"Generated VT def names: {len(vt_gen_def_names)}")
    print(f"Generated VT alias names: {len(vt_gen_alias_names)}")
    print(f"Generated FN class names: {len(fn_gen_class_names)}")
    print(f"Generated FN alias names: {len(fn_gen_alias_names)}")

    # --- Process variable_type.py ---
    vt_path = generated_dir / "variable_type.py"
    vt_content = vt_path.read_text()

    header_lines, gen_defs, gen_other, legacy_defs, legacy_other = \
        _split_variable_type(vt_content, vt_gen_def_names, vt_gen_alias_names)

    print(f"\nvariable_type.py:")
    print(f"  Generated defs: {len(gen_defs)}")
    print(f"  Legacy defs: {len(legacy_defs)}")
    print(f"  Generated other (aliases etc): {len(gen_other)}")
    print(f"  Legacy other: {len(legacy_other)}")

    # Ensure all codegen-expected aliases are present in the generated file.
    # Some aliases may have been overridden by hand-written defs in the original
    # (e.g. pow_autograd_post); those defs moved to legacy, but the alias must
    # still appear in the generated file.
    existing_alias_lhs = set()
    for line in gen_other:
        m = re.match(r"(\w+)\s*=\s*", line.strip())
        if m:
            existing_alias_lhs.add(m.group(1))
    seen_ops = set()
    for info in infos:
        if info.op_name in seen_ops:
            continue
        seen_ops.add(info.op_name)
        canonical = f"{info.op_name}_autograd"
        canonical_post = f"{info.op_name}_autograd_post"
        specific = f"{info.generated_func_stem}_autograd"
        specific_post = f"{info.generated_func_stem}_autograd_post"
        if canonical != specific and canonical not in existing_alias_lhs:
            gen_other.append(f"{canonical} = {specific}")
        if canonical_post != specific_post and canonical_post not in existing_alias_lhs:
            gen_other.append(f"{canonical_post} = {specific_post}")

    # Write slimmed variable_type.py (generated only)
    gen_vt_content = _build_vt_file(header_lines, gen_defs, gen_other)
    vt_path.write_text(gen_vt_content)
    print(f"  Wrote variable_type.py: {len(gen_vt_content)} bytes")

    # --- Process functions.py ---
    fn_path = generated_dir / "functions.py"
    fn_content = fn_path.read_text()

    (header_text, gen_classes, gen_fn_other, gen_helper_defs,
     legacy_classes, legacy_fn_other, legacy_helper_defs) = \
        _split_functions(fn_content, fn_gen_class_names, fn_gen_alias_names)

    print(f"\nfunctions.py:")
    print(f"  Generated classes: {len(gen_classes)}")
    print(f"  Legacy classes: {len(legacy_classes)}")
    print(f"  Generated other: {len(gen_fn_other)}")
    print(f"  Legacy other: {len(legacy_fn_other)}")
    print(f"  Generated helper defs: {len(gen_helper_defs)}")
    print(f"  Legacy helper defs: {len(legacy_helper_defs)}")

    # Write slimmed functions.py (generated only)
    gen_fn_content = _build_fn_file(header_text, gen_classes, gen_fn_other, gen_helper_defs)
    fn_path.write_text(gen_fn_content)
    print(f"  Wrote functions.py: {len(gen_fn_content)} bytes")

    # Write legacy functions
    legacy_fn_path = generated_dir / "functions_legacy.py"
    legacy_fn_content = _build_legacy_fn_file(
        header_text, legacy_classes, legacy_fn_other, legacy_helper_defs
    )
    legacy_fn_path.write_text(legacy_fn_content)
    print(f"  Wrote functions_legacy.py: {len(legacy_fn_content)} bytes")

    # Collect legacy backward class names for variable_type_legacy.py rewriting
    legacy_fn_class_names = set()
    for block_text in legacy_classes:
        m = re.match(r"class\s+(\w+)\s*\(", block_text)
        if m:
            legacy_fn_class_names.add(m.group(1))

    # Write legacy variable_type (deferred until after functions.py processing
    # so we know which backward node classes are in functions_legacy.py)
    legacy_vt_path = generated_dir / "variable_type_legacy.py"
    legacy_vt_content = _build_legacy_vt_file(
        header_lines, legacy_defs, legacy_other, legacy_fn_class_names
    )
    legacy_vt_path.write_text(legacy_vt_content)
    print(f"  Wrote variable_type_legacy.py: {len(legacy_vt_content)} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract legacy autograd wrappers")
    parser.add_argument(
        "--generated-dir",
        default=None,
        help="Path to _generated directory",
    )
    args = parser.parse_args()
    main(args.generated_dir)
