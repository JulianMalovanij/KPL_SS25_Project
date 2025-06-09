#!/usr/bin/env python3
"""
generate_mermaid.py

Scannt dein funktionales Python-Projekt und schreibt ein Mermaid classDiagram:
 - jede .py als class mit multi-line Methoden-Labels
 - Ordner als package-Blöcke
 - Imports und Funktionsaufrufe als --> Pfeile mit Label "use"

Usage:
    pip install pathspec
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
    res = []
    for p in root.rglob("*.py"):
        rel = p.relative_to(root)
        if not spec.match_file(str(rel)):
            res.append(rel)
    return sorted(res)


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
    deps = set()
    for rel in files:
        try:
            src = (root / rel).read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    tgt = Path(a.name.replace(".", "/") + ".py")
                    if tgt in files and tgt != rel:
                        deps.add((rel, tgt))
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level == 0:
                    for a in node.names:
                        full = (base + "." + a.name).strip(".")
                        tgt = Path(full.replace(".", "/") + ".py")
                        if tgt in files and tgt != rel:
                            deps.add((rel, tgt))
                else:
                    parent = rel.parent
                    for _ in range(node.level - 1):
                        parent = parent.parent
                    if node.module:
                        parent = parent / Path(node.module.replace(".", "/"))
                    for a in node.names:
                        cand = parent / (a.name + ".py")
                        if cand in files and cand != rel:
                            deps.add((rel, cand))
    return deps


def extract_call_deps(root: Path, files, methods_map):
    # Build reverse map: func name -> set of files defining it
    func_to_files = {}
    for f, funcs in methods_map.items():
        for sig in funcs:
            name = sig.split("(")[0]
            func_to_files.setdefault(name, set()).add(f)

    deps = set()
    for rel in files:
        try:
            src = (root / rel).read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # direct name calls
                if isinstance(node.func, ast.Name):
                    nm = node.func.id
                    for tgt in func_to_files.get(nm, ()):
                        if tgt != rel:
                            deps.add((rel, tgt))
                # attribute calls foo.bar()
                elif isinstance(node.func, ast.Attribute):
                    nm = node.func.attr
                    for tgt in func_to_files.get(nm, ()):
                        if tgt != rel:
                            deps.add((rel, tgt))
    return deps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("project_dir", help="Projekt-Root")
    p.add_argument("-o", "--output", default="diagram.mmd", help="Mermaid-file")
    args = p.parse_args()

    root = Path(args.project_dir)
    spec = load_gitignore(root)
    files = find_py(root, spec)
    methods = {f: extract_methods(root, f) for f in files}

    import_deps = extract_import_deps(root, files)
    call_deps = extract_call_deps(root, files, methods)
    deps = import_deps.union(call_deps)

    # Mermaid-Output
    lines = ["```mermaid", "classDiagram"]
    # group by folder
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
    # dependencies
    for src, tgt in sorted(deps):
        lines.append(f"  {src.stem} --> {tgt.stem} : use")

    lines.append("```")
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(f"[✔] Mermaid-Datei geschrieben: {args.output}")


if __name__ == "__main__":
    main()
