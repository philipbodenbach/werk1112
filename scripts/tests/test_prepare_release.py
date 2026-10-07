"""Exercise release preparation in disposable repositories, without publishing."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/prepare-release.py"
FILES = [
    "Cargo.toml", "Cargo.lock", "runtime/werk_media_companion.py",
    "utils/comfyUI/pyproject.toml", "utils/n8n/package.json",
    "utils/n8n/package-lock.json", "CHANGELOG.md",
    "README.md", "docs/getting-started.md", "docs/development/packaging-releases.md",
    "docs/reference/werk-protocol-v1.md", "utils/comfyUI/README.md",
    "utils/n8n/README.md", "utils/n8n/examples/README.md",
    "utils/n8n/docs/comfyui-parity.md", "utils/n8n/docs/validation.md",
]


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in FILES:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        self.current = tomllib.loads((self.root / "Cargo.toml").read_text())["package"]["version"]
        # The tests remain usable after release preparation empties Unreleased.
        changelog = self.root / "CHANGELOG.md"
        changelog.write_text(changelog.read_text().replace(
            "## [Unreleased]\n", "## [Unreleased]\n\n### Highlights\n\n- Release test change.\n", 1
        ))
        self.git("init", "-q")
        self.git("config", "user.name", "Release test")
        self.git("config", "user.email", "release@example.invalid")
        self.commit()
        self.git("tag", f"v{self.current}")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, text=True).strip()

    def commit(self):
        self.git("add", ".")
        self.git("commit", "-qm", "Fixture")

    def run_release(self, bump="patch"):
        return subprocess.run(
            [sys.executable, str(SCRIPT), bump, "--notes-file", str(self.root / "notes.md")],
            cwd=self.root, capture_output=True, text=True,
        )

    def assert_rejected_without_changes(self, message, mode="patch"):
        before = {name: (self.root / name).read_bytes() for name in FILES}
        result = self.run_release(mode)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(message, result.stderr)
        self.assertEqual(before, {name: (self.root / name).read_bytes() for name in FILES})
        self.assertFalse((self.root / "notes.md").exists())

    def test_semver_bumps_synchronize_versions_and_preserve_dependencies(self):
        major, minor, patch = map(int, self.current.split("."))
        expected = {"patch": f"{major}.{minor}.{patch + 1}",
                    "minor": f"{major}.{minor + 1}.0", "major": f"{major + 1}.0.0"}
        old_cargo = tomllib.loads((self.root / "Cargo.lock").read_text())
        old_npm = json.loads((self.root / "utils/n8n/package-lock.json").read_text())
        for bump, version in expected.items():
            with self.subTest(bump=bump):
                self.git("reset", "--hard", "HEAD")
                result = self.run_release(bump)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), f"v{version}")
                for name, section in [("Cargo.toml", "package"), ("utils/comfyUI/pyproject.toml", "project")]:
                    self.assertEqual(tomllib.loads((self.root / name).read_text())[section]["version"], version)
                self.assertIn(f'COMPANION_VERSION = "{version}"', (self.root / FILES[2]).read_text())
                self.assertEqual(json.loads((self.root / FILES[4]).read_text())["version"], version)
                cargo = tomllib.loads((self.root / "Cargo.lock").read_text())
                own = next(p for p in cargo["package"] if p["name"] == "werk1112" and "source" not in p)
                self.assertEqual(own["version"], version)
                own["version"] = self.current
                self.assertEqual(cargo, old_cargo)
                npm = json.loads((self.root / FILES[5]).read_text())
                self.assertEqual(npm["version"], version)
                self.assertEqual(npm["packages"][""]["version"], version)
                npm["version"] = npm["packages"][""]["version"] = self.current
                self.assertEqual(npm, old_npm)
                changelog = (self.root / "CHANGELOG.md").read_text()
                self.assertIn(f"## [Unreleased]\n\n## [{version}] - ", changelog)
                self.assertIn(f"/compare/v{self.current}...v{version}", changelog)
                self.assertIn(f"/compare/v{version}...HEAD", changelog)
                self.assertIn("- Release test change.", (self.root / "notes.md").read_text())
                self.assertIn(f"## [{self.current}]", changelog)
                readme = (self.root / "README.md").read_text()
                self.assertIn(f"## What’s new in v{version}", readme)
                self.assertIn("- Release test change.", readme)
                self.assertNotIn(f"## What’s new in v{self.current}", readme)
                self.assertIn(f"WERK_VERSION={version}", (self.root / "docs/getting-started.md").read_text())
                self.assertIn(f"werk1112-v{version}-linux-x86_64.tar.gz", (self.root / "docs/development/packaging-releases.md").read_text())
                n8n_readme = (self.root / "utils/n8n/README.md").read_text()
                self.assertIn(f"git switch --detach v{version}", n8n_readme)
                self.assertIn(f"release/v{version.replace('.', '-')}", n8n_readme)
                self.assertIn("npm@11.19.1", n8n_readme)

    def test_mismatched_version_is_rejected(self):
        path = self.root / "utils/comfyUI/pyproject.toml"
        path.write_text(path.read_text().replace(f'version = "{self.current}"', 'version = "0.0.0"'))
        self.commit()
        self.assert_rejected_without_changes("differs from Cargo.toml")

    def test_duplicate_tag_is_rejected(self):
        major, minor, patch = map(int, self.current.split("."))
        self.git("tag", f"v{major}.{minor}.{patch + 1}")
        self.assert_rejected_without_changes("already exists")

    def test_missing_current_release_tag_is_rejected(self):
        self.git("tag", "-d", f"v{self.current}")
        self.git("tag", "v0.0.0")
        self.assert_rejected_without_changes("differs from package version")

    def test_empty_changelog_is_rejected(self):
        path = self.root / "CHANGELOG.md"
        path.write_text(f"# Changelog\n\n## [Unreleased]\n\n## [{self.current}] - 2026-01-01\n\n- Old.\n")
        self.commit()
        self.assert_rejected_without_changes("nonempty Unreleased notes")

    def test_dirty_checkout_is_rejected(self):
        with (self.root / "Cargo.toml").open("a") as stream:
            stream.write("\n# uncommitted\n")
        self.assert_rejected_without_changes("clean tracked working tree")

    def prepared_checkout(self):
        result = self.run_release("minor")
        self.assertEqual(result.returncode, 0, result.stderr)
        (self.root / "notes.md").unlink()
        return result.stdout.strip()

    def test_check_extracts_prepared_notes_without_changing_files(self):
        tag = self.prepared_checkout()
        before = {name: (self.root / name).read_bytes() for name in FILES}
        result = self.run_release("check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), tag)
        self.assertEqual(before, {name: (self.root / name).read_bytes() for name in FILES})
        notes = (self.root / "notes.md").read_text()
        self.assertIn("- Release test change.", notes)
        self.assertNotIn(f"## [{self.current}]", notes)
        self.assertNotIn("[Unreleased]:", notes)

    def test_check_rejects_already_tagged_version(self):
        tag = self.prepared_checkout()
        self.git("tag", tag)
        self.assert_rejected_without_changes("already exists", "check")

    def test_check_rejects_mismatched_product_versions(self):
        self.prepared_checkout()
        path = self.root / "utils/n8n/package.json"
        data = json.loads(path.read_text())
        data["version"] = "0.0.0"
        path.write_text(json.dumps(data))
        self.assert_rejected_without_changes("differs from Cargo.toml", "check")

    def test_check_rejects_missing_dated_release_section(self):
        tag = self.prepared_checkout()
        path = self.root / "CHANGELOG.md"
        path.write_text(path.read_text().replace(f"## [{tag[1:]}]", "## [0.0.0]"))
        self.assert_rejected_without_changes("First dated changelog section", "check")

    def test_check_rejects_incorrect_comparison_link(self):
        self.prepared_checkout()
        path = self.root / "CHANGELOG.md"
        path.write_text(path.read_text().replace("...HEAD", "...wrong"))
        self.assert_rejected_without_changes("comparison link for Unreleased", "check")

    def test_check_rejects_empty_release_notes(self):
        tag = self.prepared_checkout()
        path = self.root / "CHANGELOG.md"
        contents = path.read_text()
        start = contents.index(f"## [{tag[1:]}]")
        body_start = contents.index("\n", start)
        end = contents.index(f"## [{self.current}]", body_start)
        path.write_text(contents[:body_start] + "\n\n" + contents[end:])
        self.assert_rejected_without_changes("nonempty changelog notes", "check")

    def test_check_rejects_invalid_release_date(self):
        tag = self.prepared_checkout()
        path = self.root / "CHANGELOG.md"
        contents = path.read_text()
        start = contents.index(f"## [{tag[1:]}]")
        end = contents.index("\n", start)
        path.write_text(contents[:start] + f"## [{tag[1:]}] - 2026-02-30" + contents[end:])
        self.assert_rejected_without_changes("day is out of range", "check")

    def test_check_rejects_version_older_than_latest_tag(self):
        self.prepared_checkout()
        self.git("tag", "v999.0.0")
        self.assert_rejected_without_changes("must be newer", "check")

    def test_docs_update_preserves_history_and_unrelated_readme_content(self):
        validation = self.root / "utils/n8n/docs/validation.md"
        history = f"\n## Historical evidence\n\nWerk {self.current} passed 123 tests on 2026-01-01.\n"
        validation.write_text(validation.read_text() + history)
        readme = self.root / "README.md"
        readme.write_text(readme.read_text() + "\nCustom footer stays intact.\n")
        self.commit()
        result = self.run_release("minor")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(validation.read_text().endswith(history))
        self.assertTrue(readme.read_text().endswith("\nCustom footer stays intact.\n"))
        self.assertIn("## Install", readme.read_text())

    def test_missing_highlights_aborts_before_modifying_any_files(self):
        path = self.root / "CHANGELOG.md"
        path.write_text(path.read_text().replace("### Highlights", "### Changes", 1))
        self.commit()
        self.assert_rejected_without_changes("### Highlights section")

    def test_check_rejects_stale_readme(self):
        tag = self.prepared_checkout()
        path = self.root / "README.md"
        path.write_text(path.read_text().replace(f"## What’s new in {tag}", "## What’s new in v0.0.0"))
        self.assert_rejected_without_changes("What's new section", "check")

    def test_check_rejects_stale_installation_version(self):
        tag = self.prepared_checkout()
        path = self.root / "docs/getting-started.md"
        path.write_text(path.read_text().replace(f"WERK_VERSION={tag[1:]}", "WERK_VERSION=0.0.0"))
        self.assert_rejected_without_changes("stale release version in docs/getting-started.md", "check")


if __name__ == "__main__":
    unittest.main()
