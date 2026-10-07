"""Exercise shell installation and removal with local archives and no network."""

import hashlib
import io
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "install.sh"
CORE = ["werk", "README.md", "LICENSE"]
OBSERVABILITY = [
    "observability/", "observability/README.md",
    "observability/grafana-dashboard.json", "observability/prometheus.yml",
]


class InstallTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.destination = self.root / "installed"
        self.archive = self.root / "release.tar.gz"
        self.checksum = self.root / "release.sha256"
        # Force the same platform on Linux CI and macOS, and serve fixtures
        # through the installer's normal download and checksum paths.
        self.command("uname", '#!/bin/sh\ncase "$1" in -s) echo Darwin;; -m) echo arm64;; esac\n')
        self.command("curl", '''#!/bin/sh
case "$2" in
    *.sha256) cp "$FIXTURE_CHECKSUM" "$4" ;;
    *.tar.gz) cp "$FIXTURE_ARCHIVE" "$4" ;;
    *) exit 1 ;;
esac
''')
        self.env = {
            **{key: value for key, value in os.environ.items()
               if not key.startswith(("WERK_", "XDG_"))},
            "HOME": str(self.root / "home"),
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "TMPDIR": str(self.root),
            "WERK_VERSION": "1.7.0",
            "WERK_INSTALL_DIR": str(self.destination),
            "FIXTURE_ARCHIVE": str(self.archive),
            "FIXTURE_CHECKSUM": str(self.checksum),
        }

    def command(self, name, contents):
        path = self.bin / name
        path.write_text(contents)
        path.chmod(0o755)

    def make_archive(self, names, symlink=None):
        with tarfile.open(self.archive, "w:gz") as archive:
            for name in names:
                entry = tarfile.TarInfo(name)
                if name == symlink:
                    entry.type = tarfile.SYMTYPE
                    entry.linkname = "README.md"
                    archive.addfile(entry)
                elif name.endswith("/"):
                    entry.type = tarfile.DIRTYPE
                    entry.mode = 0o755
                    archive.addfile(entry)
                else:
                    data = f"fixture: {name}\n".encode()
                    entry.size = len(data)
                    entry.mode = 0o644
                    archive.addfile(entry, io.BytesIO(data))
        digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.checksum.write_text(f"{digest}  werk1112-v1.7.0-macos-aarch64.tar.gz\n")

    def run_installer(self):
        result = subprocess.run(
            ["sh", str(SCRIPT)], env=self.env, capture_output=True, text=True,
            timeout=15,
        )
        self.assertEqual(list(self.root.glob("werk1112-install-*")), [])
        return result

    def test_supported_release_layouts(self):
        for names in [CORE, CORE + OBSERVABILITY]:
            with self.subTest(names=names):
                self.make_archive(names)
                result = self.run_installer()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Werk1112 installed successfully.", result.stdout)
                installed = self.destination / "werk"
                self.assertEqual(installed.read_text(), "fixture: werk\n")
                self.assertTrue(os.access(installed, os.X_OK))
                self.assertEqual(list(self.destination.iterdir()), [installed])
                result = subprocess.run(
                    ["sh", str(SCRIPT.with_name("uninstall.sh"))],
                    env=self.env, input="", capture_output=True, text=True, timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Werk1112 successfully removed.", result.stdout)
                self.assertFalse(installed.exists())

    def test_unexpected_or_incomplete_layouts_are_rejected(self):
        for names in [
            CORE + ["extra"], CORE + OBSERVABILITY + ["observability/extra"],
            CORE + ["../escape"], CORE + ["werk"], CORE[1:],
            CORE + OBSERVABILITY[:-1],
        ]:
            with self.subTest(names=names):
                self.make_archive(names)
                result = self.run_installer()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("release archive contains unexpected entries", result.stderr)
                self.assertFalse(self.destination.exists())

    def test_symlink_binary_is_rejected(self):
        self.make_archive(CORE + OBSERVABILITY, symlink="werk")
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("symbolic link: werk", result.stderr)
        self.assertFalse(self.destination.exists())

    def test_checksum_mismatch_is_rejected(self):
        self.make_archive(CORE + OBSERVABILITY)
        self.checksum.write_text("0" * 64 + "  werk1112-v1.7.0-macos-aarch64.tar.gz\n")
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checksum verification failed", result.stderr)
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
