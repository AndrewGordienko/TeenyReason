#!/usr/bin/env python3
"""Classify tracked Python files for conservative dead-file review."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

PUBLIC_API_PATHS = {
    "main.py",
    "play_probe_policy.py",
    "run_multidomain_suite.py",
    "serve_latent_dashboard.py",
    "teenyreason/__init__.py",
    "teenyreason/models/belief_world_model.py",
    "teenyreason/models/env_belief.py",
    "teenyreason/probe/probe_data.py",
    "teenyreason/probe/probe_latent.py",
}


@dataclass(frozen=True)
class FileAudit:
    path: str
    classification: str
    inbound_non_test_refs: int
    inbound_test_refs: int
    notes: tuple[str, ...]


def repo_python_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return sorted(path for path in files if (REPO_ROOT / path).exists())


def path_to_module(path_str: str) -> str:
    path = Path(path_str)
    parts = list(path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def module_parent(module_name: str) -> str:
    if "." not in module_name:
        return ""
    return module_name.rsplit(".", 1)[0]


def resolve_import_from(
    current_module: str,
    *,
    current_is_package: bool,
    node: ast.ImportFrom,
) -> str | None:
    if node.level <= 0:
        return node.module
    package_module = current_module if current_is_package else module_parent(current_module)
    anchor = package_module.split(".") if package_module else []
    if node.level > 1:
        anchor = anchor[: max(0, len(anchor) - (node.level - 1))]
    if node.module:
        return ".".join([*anchor, node.module]) if anchor else node.module
    return ".".join(anchor)


def best_module_match(module_name: str, module_to_path: dict[str, str]) -> str | None:
    candidate = module_name
    while candidate:
        if candidate in module_to_path:
            return module_to_path[candidate]
        if "." not in candidate:
            break
        candidate = candidate.rsplit(".", 1)[0]
    return None


def load_import_graph(files: list[str]) -> tuple[dict[str, set[str]], dict[str, str]]:
    module_to_path = {path_to_module(path): path for path in files}
    inbound_refs: dict[str, set[str]] = defaultdict(set)
    for path_str in files:
        path = REPO_ROOT / path_str
        if not path.exists():
            continue
        current_is_package = path.name == "__init__.py"
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        current_module = path_to_module(path_str)
        for node in ast.walk(tree):
            target_module = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target_module = alias.name
                    match = best_module_match(target_module, module_to_path)
                    if match is not None:
                        inbound_refs[match].add(path_str)
            elif isinstance(node, ast.ImportFrom):
                target_module = resolve_import_from(
                    current_module,
                    current_is_package=current_is_package,
                    node=node,
                )
                if not target_module:
                    continue
                match = best_module_match(target_module, module_to_path)
                if match is not None:
                    inbound_refs[match].add(path_str)
    return inbound_refs, module_to_path


def file_notes(path: Path, inbound_non_test_refs: int, inbound_test_refs: int) -> tuple[str, ...]:
    notes: list[str] = []
    text = path.read_text(encoding="utf-8")
    if "Compatibility facade" in text or "compatibility shim" in text.lower():
        notes.append("explicit_compatibility_facade")
    if "if __name__ == \"__main__\":" in text:
        notes.append("has_main_guard")
    if inbound_non_test_refs > 0:
        notes.append("referenced_by_runtime")
    if inbound_test_refs > 0:
        notes.append("referenced_by_tests")
    return tuple(notes)


def is_reexport_module(text: str) -> bool:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    saw_import = False
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            if isinstance(node.value.value, str):
                continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            saw_import = True
            continue
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                continue
        return False
    return saw_import


def classify_file(
    path_str: str,
    *,
    inbound_non_test_refs: int,
    inbound_test_refs: int,
    text: str,
) -> str:
    if path_str.startswith("tests/"):
        return "test_only_module"
    if path_str in PUBLIC_API_PATHS or path_str.endswith("/__init__.py"):
        return "public_api_surface"
    if "Compatibility facade" in text or "compatibility shim" in text.lower():
        return "compatibility_shim"
    if is_reexport_module(text):
        return "compatibility_shim"
    if "if __name__ == \"__main__\":" in text:
        return "runtime_entrypoint"
    if inbound_non_test_refs > 0:
        return "active_internal_module"
    if inbound_test_refs > 0:
        return "test_only_module"
    return "orphan_candidate"


def build_audit(files: list[str]) -> list[FileAudit]:
    inbound_refs, _module_to_path = load_import_graph(files)
    audits: list[FileAudit] = []
    for path_str in files:
        inbound = inbound_refs.get(path_str, set())
        inbound_non_test_refs = sum(1 for ref in inbound if not ref.startswith("tests/"))
        inbound_test_refs = sum(1 for ref in inbound if ref.startswith("tests/"))
        path = REPO_ROOT / path_str
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        audits.append(
            FileAudit(
                path=path_str,
                classification=classify_file(
                    path_str,
                    inbound_non_test_refs=inbound_non_test_refs,
                    inbound_test_refs=inbound_test_refs,
                    text=text,
                ),
                inbound_non_test_refs=inbound_non_test_refs,
                inbound_test_refs=inbound_test_refs,
                notes=file_notes(path, inbound_non_test_refs, inbound_test_refs),
            )
        )
    return audits


def print_report(audits: list[FileAudit]) -> None:
    counts: dict[str, int] = defaultdict(int)
    for audit in audits:
        counts[audit.classification] += 1
    print("Python File Audit")
    print(f"repo={REPO_ROOT}")
    print()
    for classification in sorted(counts):
        print(f"{classification}: {counts[classification]}")
    print()
    orphan_candidates = [audit for audit in audits if audit.classification == "orphan_candidate"]
    print("Orphan Candidates")
    if not orphan_candidates:
        print("  none")
        return
    for audit in orphan_candidates:
        notes = ", ".join(audit.notes) if audit.notes else "no_notes"
        print(
            f"  {audit.path} | inbound_non_test={audit.inbound_non_test_refs} | "
            f"inbound_test={audit.inbound_test_refs} | notes={notes}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a text report")
    args = parser.parse_args()

    audits = build_audit(repo_python_files())
    if args.json:
        payload = [
            {
                "path": audit.path,
                "classification": audit.classification,
                "inbound_non_test_refs": audit.inbound_non_test_refs,
                "inbound_test_refs": audit.inbound_test_refs,
                "notes": list(audit.notes),
            }
            for audit in audits
        ]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print_report(audits)


if __name__ == "__main__":
    main()
