#!/usr/bin/env python3
"""
generate_mermaid.py

Scannt Dein funktionales Python-Projekt und schreibt ein Mermaid classDiagram:
 - jede .py als class mit multi-line Methoden-Labels
 - Ordner als package-Blöcke
 - Nur tatsächliche Imports (import & from-import) als --> Pfeile mit Label "use"

Usage:
    python generate_mermaid.py /pfad/zum/projekt --output diagram.mmd
"""
import argparse
import ast
from pathlib import Path

import pathspec


def load_gitignore(root: Path):
    gi = root / ".gitignore"
    lines = gi.read_text(encoding="utf-8", errors="ignore").splitlines() if gi.exists() else []
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def find_py(root: Path, spec):
    files = []
    for p in root.rglob("*.py"):
        rel = p.relative_to(root)
        if not spec.match_file(str(rel)):
            files.append(rel)
    return sorted(files)


def extract_methods(root: Path, rel):
    funcs = []
    try:
        src = (root / rel).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src)
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef):
                args = [a.arg for a in fn.args.args]
                funcs.append(f"{fn.name}({', '.join(args)})")
    except:
        pass
    return funcs


def extract_import_deps(root: Path, files):
    """
    Liefert nur echte Modul-Imports:
      import x.y
      from x.y import z
      from .   import a
      from ..b import c
    als (source_rel, target_rel)-Paare, wenn target_rel in files ist.
    """
    deps = set()
    file_set = set(files)
    for rel in files:
        path = root / rel
        try:
            src = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except:
            continue

        for node in ast.walk(tree):
            # import x.y -> dependency auf x/y.py
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name  # e.g. "pkg.module"
                    candidate = Path(mod.replace(".", "/")).with_suffix(".py")
                    if candidate in file_set:
                        deps.add((rel, candidate))

            # from X import Y  -> dependency auf X.py
            # from X.Y import Z -> dependency auf X/Y.py
            # relative: from .A import B -> dependency auf same/package/A.py
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    # absolute
                    if node.module:
                        base = node.module  # may include "."
                        candidate = Path(base.replace(".", "/")).with_suffix(".py")
                        if candidate in file_set:
                            deps.add((rel, candidate))
                else:
                    # relative imports
                    parent = rel.parent
                    for _ in range(node.level - 1):
                        parent = parent.parent
                    if node.module:
                        parent = parent / Path(node.module.replace(".", "/"))
                    candidate = parent.with_suffix(".py")
                    if candidate in file_set:
                        deps.add((rel, candidate))

    return deps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("project_dir", help="Projekt-Root")
    p.add_argument("-o", "--output", default="diagram.mmd", help="Mermaid-file")
    args = p.parse_args()

    root = Path(args.project_dir)
    spec = load_gitignore(root)
    files = find_py(root, spec)

    # Methoden pro Datei
    methods = {f: extract_methods(root, f) for f in files}

    # Nur Import-Abhängigkeiten
    deps = extract_import_deps(root, files)

    # Mermaid-Output
    lines = ["classDiagram"]
    # Gruppiere nach Ordner
    pkgs = {}
    for f in files:
        pkg = str(f.parent) or "root"
        pkgs.setdefault(pkg, []).append(f)

    for pkg, flist in pkgs.items():
        if pkg != "root":
            lines.append(f"  %% package {pkg}")
        for f in flist:
            cname = f.stem
            if methods[f]:
                lines.append(f"  class {cname} {{")
                for m in methods[f]:
                    lines.append(f"    + {m}")
                lines.append("  }")
            else:
                lines.append(f"  class {cname}")

    # Nur echte Import-Kanten
    for src, tgt in sorted(deps):
        lines.append(f"  {src.stem} --> {tgt.stem} : use")

    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(f"[✔] Mermaid-Datei geschrieben: {args.output}")


if __name__ == "__main__":
    main()
