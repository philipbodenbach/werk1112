#!/usr/bin/env python3
"""Prepare release metadata automatically, or validate it before publication."""

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
DOC_VERSION = r"(?P<version>[0-9]+\.[0-9]+\.[0-9]+)"
# Explicit current-release fields: do not rewrite historical validation results,
# dependency versions, protocol versions, or arbitrary prose throughout docs/.
DOC_FIELDS = {
    "docs/getting-started.md": [r'WERK_VERSION(?:=| = ")' + DOC_VERSION],
    "docs/development/packaging-releases.md": [
        r"(?:For package version `|werk1112-v)" + DOC_VERSION,
    ],
    "docs/reference/werk-protocol-v1.md": [r'"service_version": "' + DOC_VERSION],
    "utils/comfyUI/README.md": [r"package is version \*\*" + DOC_VERSION],
    "utils/n8n/README.md": [
        r"(?:`n8n-nodes-werk1112` \*\*|Werk \*\*|share release version \*\*|git switch --detach v|The Werk )" + DOC_VERSION,
    ],
    "utils/n8n/examples/README.md": [r"(?:package|Werk) \*\*" + DOC_VERSION],
    "utils/n8n/docs/comfyui-parity.md": [r"Werk/ComfyUI \*\*" + DOC_VERSION],
    "utils/n8n/docs/validation.md": [
        r"ComfyUI and the n8n package at \*\*" + DOC_VERSION,
        r"The package remains private and manually installed for Werk \*\*" + DOC_VERSION,
        r"Release reference: `v" + DOC_VERSION,
    ],
}


def update_doc_fields(files, current, next_version):
    for path, patterns in DOC_FIELDS.items():
        for pattern in patterns:
            expected = current
            replacement = next_version
            matches = list(re.finditer(pattern, files[path]))
            if not matches or any(match["version"] != expected for match in matches):
                raise ValueError(f"Missing or stale release version in {path}: expected {expected}")

            def replace(match):
                start, end = match.span("version")
                return match[0][:start - match.start()] + replacement + match[0][end - match.start():]

            files[path] = re.sub(pattern, replace, files[path])


def readme_section(text, version):
    pattern = rf"^## What’s new in v{re.escape(version)}\n.*?(?=^## |\Z)"
    matches = list(re.finditer(pattern, text, re.M | re.S))
    if len(matches) != 1:
        raise ValueError(f"README.md must contain one What's new section for v{version}")
    return matches[0]


def update_readme(text, current, next_version, date, notes):
    highlights = re.search(r'^### Highlights\n(.*?)(?=^### |\Z)', notes, re.M | re.S)
    if highlights and re.search(r'^- \S', highlights[1], re.M):
        summary = highlights[1].strip()
    else:
        # Preserve multiline bullets; a summary is copied, never invented.
        bullets = re.findall(r'^- \S[^\n]*(?:\n[ \t]+[^\n]+)*', notes, re.M)
        summary = "\n".join(bullets[:5])
        if not summary:
            raise ValueError("Release notes must contain at least one bullet for the README")
    section = readme_section(text, current)
    anchor = next_version.replace(".", "") + "---" + str(date)
    new_section = (
        f"## What’s new in v{next_version}\n\n"
        f"Werk Core, Media Companion, ComfyUI and n8n share release version **{next_version}**.\n"
        "ComfyUI and n8n remain Beta integrations.\n\n"
        f"{summary}\n\n"
        f"See the [v{next_version} changelog](CHANGELOG.md#{anchor}) "
        "for all changes and compatibility notes.\n\n"
    )
    return text[:section.start()] + new_section + text[section.end():]


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def replace_once(text, pattern, replacement):
    result, count = re.subn(pattern, lambda match: replacement(match), text, flags=re.M)
    if count != 1:
        raise ValueError(f"Expected exactly one match for {pattern!r}, found {count}")
    return result


def commits_since(tag):
    revisions = git("rev-list", "--reverse", "--no-merges", f"{tag}..HEAD").splitlines()
    return [(revision, git("show", "-s", "--format=%B", revision)) for revision in revisions]


def inferred_bump(commits):
    for _, message in commits:
        if re.match(r"^[a-zA-Z][\w-]*(?:\([^\n)]*\))?!:", message) or re.search(
            r"^BREAKING[ -]CHANGE:", message, re.M
        ):
            return "major"
    if any(re.match(r"^feat(?:\([^\n)]*\))?:", message) for _, message in commits):
        return "minor"
    return "patch"


def commit_notes(commits):
    # Escape Markdown/HTML syntax in commit subjects; never interpolate into a shell.
    bullets = []
    for revision, message in commits:
        subject = message.splitlines()[0]
        subject = subject.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        subject = re.sub(r"([\\`*_\[\]])", r"\\\1", subject)
        bullets.append(f"- {subject} (`{revision[:12]}`)")
    return "### Changes\n\n" + "\n".join(bullets) + "\n"


def prepare(root, bump, date, allow_existing=False):
    paths = [
        "Cargo.toml", "Cargo.lock", "runtime/werk_media_companion.py",
        "utils/comfyUI/pyproject.toml", "utils/n8n/package.json",
        "utils/n8n/package-lock.json", "CHANGELOG.md", "README.md",
        *DOC_FIELDS,
    ]
    files = {path: (root / path).read_text() for path in paths}
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

    tags = git("tag", "--list").splitlines()
    stable_tags = [t for t in tags if re.fullmatch("v" + VERSION, t)]
    if not stable_tags:
        raise ValueError("No existing stable release tag found")
    latest = max(stable_tags, key=lambda t: tuple(map(int, t[1:].split("."))))
    if bump == "auto":
        if tuple(map(int, current.split("."))) > tuple(map(int, latest[1:].split("."))):
            # Transition from a manually prepared release, including v1.7.0.
            return prepare(root, "check", date)
        if latest != f"v{current}":
            raise ValueError(f"Latest stable tag {latest} differs from package version v{current}")
        commits = commits_since(latest)
        if not commits:
            # A new dispatch after a partial/successful run reuses its tag/notes.
            return prepare(root, "check", date, allow_existing=True)
        bump = inferred_bump(commits)

    major, minor, patch = map(int, current.split("."))
    next_version = {
        "patch": f"{major}.{minor}.{patch + 1}",
        "minor": f"{major}.{minor + 1}.0",
        "major": f"{major + 1}.0.0",
        "check": current,
    }[bump]
    tag = f"v{next_version}"
    if tag in tags:
        if not allow_existing or git("rev-parse", f"{tag}^{{commit}}") != git("rev-parse", "HEAD"):
            raise ValueError(f"Tag {tag} already exists")
        previous_tags = [t for t in stable_tags if t != tag]
        if not previous_tags:
            raise ValueError("Cannot recover a release without a previous stable tag")
        latest = max(previous_tags, key=lambda t: tuple(map(int, t[1:].split("."))))
    if bump == "check":
        if tuple(map(int, current.split("."))) <= tuple(map(int, latest[1:].split("."))):
            raise ValueError(f"Prepared version v{current} must be newer than {latest}")
        subprocess.run(["git", "merge-base", "--is-ancestor", latest, "HEAD"], check=True)
        changelog = files["CHANGELOG.md"]
        sections = list(re.finditer(r'^## \[([^\]]+)\]([^\n]*)$', changelog, re.M))
        if len(sections) < 2 or sections[0][1] != "Unreleased" or sections[1][1] != current:
            raise ValueError(f"First dated changelog section must be [{current}]")
        if sum(section[1] == current for section in sections) != 1:
            raise ValueError(f"Duplicate changelog sections for {current}")
        date_text = sections[1][2]
        if not re.fullmatch(r" - [0-9]{4}-[0-9]{2}-[0-9]{2}", date_text):
            raise ValueError("Release changelog heading must include a YYYY-MM-DD date")
        datetime.strptime(date_text[3:], "%Y-%m-%d")
        end = sections[2].start() if len(sections) > 2 else len(changelog)
        notes = changelog[sections[1].end():end].strip()
        if not re.search(r'^- \S', notes, re.M):
            raise ValueError("Prepared release must contain nonempty changelog notes")
        for label, comparison in [("Unreleased", f"{tag}...HEAD"), (current, f"{latest}...{tag}")]:
            if not re.search(rf'^\[{re.escape(label)}\]: https://\S+/compare/{re.escape(comparison)}$', changelog, re.M):
                raise ValueError(f"Missing or incorrect changelog comparison link for {label}")
        update_doc_fields(files, current, current)
        section = readme_section(files["README.md"], current)[0]
        anchor = current.replace(".", "") + "---" + date_text[3:]
        if f"release version **{current}**" not in section or f"CHANGELOG.md#{anchor}" not in section:
            raise ValueError("README.md release version or changelog link is stale")
        return tag, notes + "\n"
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
    if not unreleased:
        raise ValueError("CHANGELOG.md must contain an Unreleased section")
    if re.search(r'^- \S', unreleased[1], re.M):
        notes = unreleased[1].strip() + "\n"
    else:
        commits = commits_since(latest)
        if not commits:
            raise ValueError("No changes since the latest release")
        notes = commit_notes(commits)
    changelog = changelog[:unreleased.start()] + (
        f"## [Unreleased]\n\n## [{next_version}] - {date}\n\n{notes}\n"
    ) + changelog[unreleased.end():]
    changelog = replace_once(
        changelog, rf'^\[Unreleased\]: (https://[^\s]+/compare/)v{re.escape(current)}\.\.\.HEAD$',
        lambda m: f"[Unreleased]: {m[1]}{tag}...HEAD\n[{next_version}]: {m[1]}v{current}...{tag}",
    )
    files["CHANGELOG.md"] = changelog
    update_doc_fields(files, current, next_version)
    files["README.md"] = update_readme(files["README.md"], current, next_version, date, notes)
    # Complete all validation before modifying any tracked files.
    for path, contents in files.items():
        (root / path).write_text(contents)
    return tag, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bump", choices=["auto", "patch", "minor", "major", "check"],
                        help="Infer SemVer from commits, choose a bump, or check prepared metadata")
    parser.add_argument("--notes-file", type=Path, required=True)
    args = parser.parse_args()
    if args.bump != "check" and git("status", "--porcelain", "--untracked-files=no"):
        parser.error("Release preparation requires a clean tracked working tree")
    try:
        tag, notes = prepare(Path.cwd(), args.bump, datetime.now(ZoneInfo("Europe/Berlin")).date())
    except (OSError, ValueError, KeyError, StopIteration, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Release preparation failed: {error}\n")
    args.notes_file.write_text(notes)
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as stream:
            stream.write(f"tag={tag}\n")
    print(tag)


if __name__ == "__main__":
    main()
