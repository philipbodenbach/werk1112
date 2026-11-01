"""Read package metadata and PyPI release metadata; never import a runtime or run pip.

Executed with the managed interpreter's -I -B flags. Only the requested runtime
packages and common runtime dependencies are inspected, not every transitive tool.
"""
import sys
# External user installations retain their user site, but never import modules
# from the directory where werk was invoked.
sys.path = [path for path in sys.path if path]
import concurrent.futures
import importlib.metadata
import json
import sys
import urllib.error
import urllib.request


def version_type():
    try:
        from packaging.version import Version
    except ImportError:
        from pip._vendor.packaging.version import Version
    return Version


def fetch(package):
    request = urllib.request.Request(
        "https://pypi.org/pypi/" + package + "/json",
        headers={"User-Agent": "werk-update-scan", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        payload = response.read(8 * 1024 * 1024 + 1)
    if len(payload) > 8 * 1024 * 1024:
        raise ValueError("Package metadata is too large")
    return json.loads(payload)


def package_bounds(package, constraints, requirements):
    try:
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
        from pip._vendor.packaging.utils import canonicalize_name
    bounds = []
    for origin, lines in (("installer requirements", requirements),
                          ("local constraints.txt", constraints.splitlines())):
        for line in lines:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            requirement = Requirement(line)
            if (canonicalize_name(requirement.name) == canonicalize_name(package)
                    and (requirement.marker is None or requirement.marker.evaluate())):
                bounds.append((requirement, origin))
    return bounds


def release_supports_python(files):
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError:
        from pip._vendor.packaging.specifiers import SpecifierSet
    python_version = ".".join(str(part) for part in sys.version_info[:3])
    return any(not file.get("yanked", False)
               and SpecifierSet(file.get("requires_python") or "").contains(python_version)
               for file in files)


def check(package, offline, pinned, metadata=importlib.metadata, get=fetch,
          constraints="", requirements=()):
    try:
        distribution = metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        return None
    installed = distribution.version
    item = dict(name=package, installed=installed, available=None,
                status="unknown", detail="Offline; remote version not checked")
    if offline:
        return item
    try:
        # Direct source installs may be forks with an unrelated PyPI name.
        if distribution.read_text("direct_url.json") and package not in pinned:
            item["detail"] = "Direct URL/source installation; no verified release comparison"
            return item
        info = get(package)
        latest = info["info"]["version"]
        Version = version_type()
        current_version, latest_version = Version(installed), Version(latest)
        item["available"] = latest
        if latest_version.is_prerelease or latest_version.is_devrelease:
            item["detail"] = "Latest metadata refers to a pre-release; no stable comparison"
        elif not info.get("urls") or all(file.get("yanked", False) for file in info["urls"]):
            item["detail"] = "Latest release has no non-yanked distributions"
        elif latest_version > current_version:
            if package in pinned:
                item["status"] = "pinned"
                item["detail"] = "New upstream release; this runtime is pinned by Werk"
                return item
            bounds = package_bounds(package, constraints, requirements)
            releases = dict(info.get("releases") or {})
            releases.setdefault(latest, info["urls"])
            candidates = []
            for number, files in releases.items():
                try:
                    candidate = Version(number)
                except ValueError:
                    continue
                if (candidate > current_version and not candidate.is_prerelease
                        and not candidate.is_devrelease
                        and all(req.specifier.contains(candidate) for req, _ in bounds)
                        and release_supports_python(files)):
                    candidates.append((candidate, number))
            excluded = [f"{origin}: {req}" for req, origin in bounds
                        if not req.specifier.contains(latest_version)]
            if candidates:
                item["status"] = "update_available"
                item["available"] = max(candidates)[1]
                item["detail"] = "New release within the installer's version limits; dependency compatibility must still be checked before upgrading"
                if excluded:
                    item["detail"] += f". Upstream {latest} is excluded by " + "; ".join(excluded)
            elif excluded:
                item["status"] = "pinned"
                item["detail"] = f"No newer supported release; upstream {latest} is excluded by " + "; ".join(excluded)
            else:
                item["status"] = "up_to_date"
                item["detail"] = "No newer non-yanked stable release supports this Python version"
        else:
            item["status"] = "up_to_date"
            item["detail"] = "Installed version is at least the latest stable PyPI release"
    except urllib.error.HTTPError as error:
        item["status"] = "error"
        item["detail"] = "PyPI returned HTTP " + str(error.code)
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        item["status"] = "error"
        item["detail"] = "Could not fetch or compare PyPI metadata; check network and package metadata"
    return item


def main():
    options = json.loads(sys.argv[1])
    required = options.get("required", [])
    present = False
    for package in required:
        try:
            importlib.metadata.distribution(package)
            present = True
        except importlib.metadata.PackageNotFoundError:
            pass
    if required and not present:
        print("[]")
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = pool.map(
            lambda package: check(package, options["offline"], options["pinned"],
                                  constraints=options.get("constraints", ""),
                                  requirements=options.get("requirements", [])),
            options["packages"],
        )
        print(json.dumps([item for item in results if item is not None]))


if __name__ == "__main__":
    main()
