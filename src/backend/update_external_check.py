"""Validate pip's proposed transaction against the entire shared environment.

Only distribution metadata is read. No runtime imports, installs or downloads.
"""
import sys
sys.path = [path for path in sys.path if path]
import importlib.metadata as metadata
import json
import re

try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import Version
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name
    from pip._vendor.packaging.version import Version


def validate(report, installed, requested):
    installed = {canonicalize_name(d.metadata["Name"]): d for d in installed}
    projected = {
        name: (d.version, d.requires or []) for name, d in installed.items()
    }
    constraints = []
    for item in report["install"]:
        package = item["metadata"]
        name, version = package["name"], package["version"]
        if not (re.fullmatch(r"[A-Za-z0-9_.-]+", name)
                and re.fullmatch(r"[A-Za-z0-9_.+!-]+", version)):
            raise ValueError("Invalid resolved package identity")
        key = canonicalize_name(name)
        old = installed.get(key)
        if old:
            direct = json.loads(old.read_text("direct_url.json") or "{}")
            if direct.get("dir_info", {}).get("editable"):
                raise ValueError("Editable package requires its own source workflow: " + name)
            if Version(version) < Version(old.version):
                raise ValueError("Refusing downgrade in external environment: " + name)
            if direct and direct.get("url") != item.get("download_info", {}).get("url"):
                raise ValueError("Refusing to replace an external source: " + name)
        projected[key] = (version, package.get("requires_dist") or [])
        constraints.append(name + "==" + version)

    # Include requested extras and propagate them through dependency edges.
    extras = {name: set() for name in projected}
    for spec in requested:
        if spec.startswith("git+"):
            # Bare VCS roots (e.g. oMLX) have no requested extras.
            continue
        req = Requirement(spec)
        extras.setdefault(canonicalize_name(req.name), set()).update(req.extras)
    problems = set()
    changed = True
    while changed:
        changed = False
        for name, (version, requirements) in projected.items():
            for spec in requirements:
                req = Requirement(spec)
                if req.marker and not any(
                    req.marker.evaluate({"extra": extra})
                    for extra in {""} | extras[name]
                ):
                    continue
                dependency = canonicalize_name(req.name)
                found = projected.get(dependency)
                if found is None or not req.specifier.contains(found[0], prereleases=True):
                    problems.add(
                        f"{name} {version} requires {req}; planned "
                        + (found[0] if found else "not installed")
                    )
                if found and not req.extras.issubset(extras[dependency]):
                    extras[dependency].update(req.extras)
                    changed = True
    if problems:
        raise ValueError("Update would break the shared environment:\n" + "\n".join(sorted(problems)))
    return "\n".join(constraints) + "\n"


def main():
    with open(sys.argv[1], encoding="utf-8") as source:
        report = json.load(source)
    constraints = validate(report, metadata.distributions(), json.loads(sys.argv[3]))
    # Write only after the entire proposed environment has passed validation.
    with open(sys.argv[2], "w", encoding="utf-8") as output:
        output.write(constraints)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
