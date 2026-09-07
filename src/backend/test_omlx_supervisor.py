"""Linux/macOS process tests using tiny launchers, no oMLX installation."""

import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


SOURCE_PATH = Path(__file__).with_name("omlx_supervisor.py")
SOURCE = SOURCE_PATH.read_text()
SPEC = importlib.util.spec_from_file_location("omlx_supervisor", SOURCE_PATH)
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


@unittest.skipUnless(os.name == "posix", "oMLX process groups require POSIX")
class OmlxSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def launcher(self, source):
        path = self.root / "omlx"
        path.write_text(source)
        return path

    def start(self, launcher, *args):
        child = subprocess.Popen(
            [sys.executable, "-c", SOURCE, str(launcher), *args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self.addCleanup(self.cleanup_child, child)
        return child

    @staticmethod
    def cleanup_child(child):
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        for stream in (child.stdin, child.stdout, child.stderr):
            if stream is not None:
                stream.close()

    def read_ready(self, child):
        readable, _, _ = select.select([child.stdout], [], [], 5)
        self.assertTrue(readable, "launcher did not become ready")
        line = child.stdout.readline()
        self.assertTrue(line, "launcher exited before readiness")
        return line.decode().strip()

    def test_open_parent_pipe_keeps_worker_alive_and_eof_stops_it(self):
        launcher = self.launcher(
            "import time\nprint('ready', flush=True)\n"
            "while True:\n    time.sleep(0.05)\n"
        )
        child = self.start(launcher, "serve")
        self.assertEqual(self.read_ready(child), "ready")
        time.sleep(0.05)
        self.assertIsNone(child.poll())
        child.stdin.close()
        self.assertEqual(child.wait(timeout=3), -signal.SIGKILL)

    def test_launcher_argv_import_directory_and_normal_exit_code_are_preserved(self):
        (self.root / "adjacent_module.py").write_text("VALUE = 'selected launcher directory'\n")
        launcher = self.launcher(
            "import adjacent_module, json, sys\n"
            "print(json.dumps({'argv':sys.argv, 'path':sys.path[0], 'name':__name__, "
            "'file':__file__, 'imported':adjacent_module.VALUE}), flush=True)\n"
            "raise SystemExit(7)\n"
        )
        child = self.start(launcher, "serve", "--model-dir", "model path")
        detail = json.loads(self.read_ready(child))
        self.assertEqual(detail["argv"], [str(launcher), "serve", "--model-dir", "model path"])
        self.assertEqual(detail["path"], str(self.root))
        self.assertEqual(detail["name"], "__main__")
        self.assertEqual(detail["file"], str(launcher))
        self.assertEqual(detail["imported"], "selected launcher directory")
        self.assertEqual(child.wait(timeout=3), 7)

    def test_abrupt_parent_exit_closes_pipe_without_python_cleanup(self):
        launcher = self.launcher(
            "import time\nprint('ready', flush=True)\n"
            "while True:\n    time.sleep(0.05)\n"
        )
        # The short-lived parent cannot run destructors after os._exit().
        # Redirect worker stderr so it cannot keep the parent's output pipe open.
        parent_source = (
            "import os, pathlib, subprocess, sys\n"
            "source = pathlib.Path(sys.argv[1]).read_text()\n"
            "worker = subprocess.Popen([sys.executable, '-c', source, sys.argv[2], 'serve'], "
            "stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)\n"
            "assert worker.stdout.readline() == b'ready\\n'\n"
            "print(worker.pid, flush=True)\n"
            "os._exit(0)\n"
        )
        parent = subprocess.run(
            [sys.executable, "-c", parent_source, str(SOURCE_PATH), str(launcher)],
            capture_output=True, text=True, timeout=5, check=True,
        )
        pid = int(parent.stdout.strip())
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            # Linux can retain an exited orphan as a zombie until init reaps it.
            status = Path(f"/proc/{pid}/stat")
            try:
                state = status.read_text().split(") ", 1)[1]
            except FileNotFoundError:
                state = ""
            if state.startswith("Z "):
                return
            time.sleep(0.02)
        os.kill(pid, signal.SIGKILL)
        self.fail("worker survived abrupt parent exit")

    def test_shared_process_group_is_never_signalled(self):
        with patch.object(supervisor.os, "getpid", return_value=1234), \
                patch.object(supervisor.os, "getpgrp", return_value=999), \
                patch.object(supervisor.os, "killpg") as kill_group, \
                patch.object(supervisor.os, "kill") as kill_process:
            supervisor.stop_owned_worker()
        kill_group.assert_not_called()
        kill_process.assert_called_once_with(1234, signal.SIGKILL)

    def test_only_own_leader_group_is_signalled(self):
        with patch.object(supervisor.os, "getpid", return_value=1234), \
                patch.object(supervisor.os, "getpgrp", return_value=1234), \
                patch.object(supervisor.os, "killpg") as kill_group, \
                patch.object(supervisor.os, "kill") as kill_process:
            supervisor.stop_owned_worker()
        kill_group.assert_called_once_with(1234, signal.SIGKILL)
        kill_process.assert_not_called()

    def test_pipe_errors_still_stop_owned_worker(self):
        with patch.object(supervisor.os, "read", side_effect=OSError), \
                patch.object(supervisor.os, "close", side_effect=OSError), \
                patch.object(supervisor, "stop_owned_worker") as stop:
            supervisor.watch_parent(1234)
        stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
