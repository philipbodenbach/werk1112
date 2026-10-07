#!/usr/bin/env python3
"""Prepare synchronized product versions and release notes without building artifacts."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib
from zoneinfo import ZoneInfo


VERSION = r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def replace_once(text, pattern, replacement):
    result, count = re.subn(pattern, lambda match: replacement(match), text, flags=re.M)
    if count != 1:
        raise ValueError(f"Expected exactly one match for {pattern!r}, found {count}")
    return result


def prepare(root, bump, date):
    paths = [
        "Cargo.toml", "Cargo.lock", "runtime/werk_media_companion.py",
        "utils/comfyUI/pyproject.toml", "utils/n8n/package.json",
        "utils/n8n/package-lock.json", "CHANGELOG.md",
    ]
    original = {path: (root / path).read_text() for path in paths}
    files = original.copy()
    current = tomllib.loads(files["Cargo.toml"])["package"]["version"]
    if not re.fullmatch(VERSION, current):
        raise ValueError(f"Expected a stable SemVer package version, got {current!r}")
    versions = {
        "Cargo.lock": next(p["version"] for p in tomllib.loads(files["Cargo.lock"])["package"]
                           if p["name"] == "werk1112" and "source" not in p),
        "ComfyUI": tomllib.loads(files["utils/comfyUI/pyproject.toml"])["project"]["version"],
        "n8n": json.loads(files["utils/n8n/package.json"])["version"],
        "n8n lockfile": json.loads(files["utils/n8n/package-lock.json"])["version"],
        "n8n lockfile root": json.loads(files["utils/n8n/package-lock.json"])["packages"][""]["version"],
    }
    companion = re.search(r'^COMPANION_VERSION = "([^"]+)"$', files["runtime/werk_media_companion.py"], re.M)
    if not companion:
        raise ValueError("Missing COMPANION_VERSION")
    versions["Media Companion"] = companion[1]
    for name, value in versions.items():
        if value != current:
            raise ValueError(f"{name} version {value} differs from Cargo.toml {current}")

    major, minor, patch = map(int, current.split("."))
    next_version = {
        "patch": f"{major}.{minor}.{patch + 1}",
        "minor": f"{major}.{minor + 1}.0",
        "major": f"{major + 1}.0.0",
    }[bump]
    tag = f"v{next_version}"
    tags = git("tag", "--list").splitlines()
    if tag in tags:
        raise ValueError(f"Tag {tag} already exists")
    stable_tags = [t for t in tags if re.fullmatch("v" + VERSION, t)]
    if not stable_tags:
        raise ValueError("No existing stable release tag found")
    latest = max(stable_tags, key=lambda t: tuple(map(int, t[1:].split("."))))
    if latest != f"v{current}":
        raise ValueError(f"Latest stable tag {latest} differs from package version v{current}")
    subprocess.run(["git", "merge-base", "--is-ancestor", latest, "HEAD"], check=True)

    for path, section in [("Cargo.toml", "package"), ("utils/comfyUI/pyproject.toml", "project")]:
        files[path] = replace_once(
            files[path], rf'(\[{section}\]\n(?:(?!\[)[^\n]*\n)*?version = ")[^"]+("[^\n]*$)',
            lambda m: m[1] + next_version + m[2],
        )
    files["Cargo.lock"] = replace_once(
        files["Cargo.lock"], r'(\[\[package\]\]\nname = "werk1112"\nversion = ")[^"]+("$)',
        lambda m: m[1] + next_version + m[2],
    )
    files["runtime/werk_media_companion.py"] = replace_once(
        files["runtime/werk_media_companion.py"], r'^COMPANION_VERSION = "[^"]+"$',
        lambda _: f'COMPANION_VERSION = "{next_version}"',
    )
    # Preserve JSON formatting and dependency entries; only edit root metadata.
    for path in ["utils/n8n/package.json", "utils/n8n/package-lock.json"]:
        files[path] = replace_once(files[path], r'^(  "version": ")[^"]+(",$)',
                                   lambda m: m[1] + next_version + m[2])
    files["utils/n8n/package-lock.json"] = replace_once(
        files["utils/n8n/package-lock.json"],
        r'(^    "": \{\n      "name": "n8n-nodes-werk1112",\n      "version": ")[^"]+(",$)',
        lambda m: m[1] + next_version + m[2],
    )
    files["utils/n8n/package.json"] = files["utils/n8n/package.json"].replace(
        f"manual installation for Werk {current}", f"manual installation for Werk {next_version}"
    )

    changelog = files["CHANGELOG.md"]
    unreleased = re.search(r'^## \[Unreleased\]\n(.*?)(?=^## \[)', changelog, re.M | re.S)
    if not unreleased or not re.search(r'^- \S', unreleased[1], re.M):
        raise ValueError("CHANGELOG.md must contain nonempty Unreleased notes")
    notes = unreleased[1].strip() + "\n"
    changelog = changelog[:unreleased.start()] + (
        f"## [Unreleased]\n\n## [{next_version}] - {date}\n\n{notes}\n"
    ) + changelog[unreleased.end():]
    changelog = replace_once(
        changelog, rf'^\[Unreleased\]: (https://[^\s]+/compare/)v{re.escape(current)}\.\.\.HEAD$',
        lambda m: f"[Unreleased]: {m[1]}{tag}...HEAD\n[{next_version}]: {m[1]}v{current}...{tag}",
    )
    files["CHANGELOG.md"] = changelog
    # Complete all validation before modifying any tracked files.
    for path, contents in files.items():
        (root / path).write_text(contents)
    return tag, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bump", choices=["patch", "minor", "major"])
    parser.add_argument("--notes-file", type=Path, required=True)
    args = parser.parse_args()
    if git("status", "--porcelain", "--untracked-files=no"):
        parser.error("Release preparation requires a clean tracked working tree")
    try:
        tag, notes = prepare(Path.cwd(), args.bump, datetime.now(ZoneInfo("Europe/Berlin")).date())
    except (ValueError, KeyError, StopIteration, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Release preparation failed: {error}\n")
    args.notes_file.write_text(notes)
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as stream:
            stream.write(f"tag={tag}\n")
    print(tag)


if __name__ == "__main__":
    main()
