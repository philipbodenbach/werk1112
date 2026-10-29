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


def check(package, offline, pinned, metadata=importlib.metadata, get=fetch):
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
            item["status"] = "pinned" if package in pinned else "update_available"
            item["detail"] = (
                "New upstream release; this runtime is pinned by Werk"
                if package in pinned else
                "New upstream release; Python, accelerator and dependency compatibility must be checked before upgrading"
            )
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
            lambda package: check(package, options["offline"], options["pinned"]),
            options["packages"],
        )
        print(json.dumps([item for item in results if item is not None]))


if __name__ == "__main__":
    main()
