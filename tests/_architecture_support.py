from __future__ import annotations

import ast
from pathlib import Path


def source_for(path: str) -> str:
    """Return repository source text for architecture-boundary assertions."""

    return Path(path).read_text(encoding="utf-8")


def class_body_source(path: str, class_name: str) -> str:
    """Return the source segment for one top-level class."""

    source = source_for(path)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            segment = ast.get_source_segment(source, node)
            if segment is None:
                raise AssertionError(f"Could not extract class {class_name!r}")
            return segment
    raise AssertionError(f"Class {class_name!r} not found in {path}")


def package_import_cycles(
    package_path: str,
    package_name: str,
) -> list[tuple[str, ...]]:
    """Return top-level import cycles within one package directory."""

    module_paths: dict[str, Path] = {}
    for path in Path(package_path).glob("*.py"):
        module_name = f"{package_name}.{path.stem}"
        if path.name == "__init__.py":
            module_name = package_name
        module_paths[module_name] = path

    def resolve_module(import_name: str) -> str | None:
        if import_name in module_paths:
            return import_name
        parts = import_name.split(".")
        while parts:
            candidate = ".".join(parts)
            if candidate in module_paths:
                return candidate
            parts.pop()
        return None

    graph: dict[str, set[str]] = {module_name: set() for module_name in module_paths}
    for module_name, path in module_paths.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_names = [node.module]
            else:
                continue

            for imported_name in imported_names:
                if not imported_name.startswith(package_name):
                    continue
                dependency = resolve_module(imported_name)
                if dependency is not None and dependency != module_name:
                    graph[module_name].add(dependency)

    index_by_module: dict[str, int] = {}
    lowlink_by_module: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    cycles: list[tuple[str, ...]] = []

    def visit(module_name: str) -> None:
        index_by_module[module_name] = len(index_by_module)
        lowlink_by_module[module_name] = index_by_module[module_name]
        stack.append(module_name)
        on_stack.add(module_name)

        for dependency in graph[module_name]:
            if dependency not in index_by_module:
                visit(dependency)
                lowlink_by_module[module_name] = min(
                    lowlink_by_module[module_name],
                    lowlink_by_module[dependency],
                )
            elif dependency in on_stack:
                lowlink_by_module[module_name] = min(
                    lowlink_by_module[module_name],
                    index_by_module[dependency],
                )

        if lowlink_by_module[module_name] != index_by_module[module_name]:
            return

        component: list[str] = []
        while True:
            dependency = stack.pop()
            on_stack.remove(dependency)
            component.append(dependency)
            if dependency == module_name:
                break
        if len(component) > 1:
            cycles.append(tuple(sorted(component)))

    for module_name in sorted(module_paths):
        if module_name not in index_by_module:
            visit(module_name)

    return sorted(cycles)
