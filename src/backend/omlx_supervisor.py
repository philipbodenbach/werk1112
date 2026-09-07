"""Run a validated oMLX console script while its Werk parent holds stdin open.

Rust starts this interpreter as a new process-group leader. No MLX code is
imported here; the selected launcher retains its original argv and import path.
"""

import os
from pathlib import Path
import runpy
import signal
import sys
import threading


def stop_owned_worker():
    """Never signal a process group that also belongs to the parent or a shell."""
    pid = os.getpid()
    termination = getattr(signal, "SIGKILL", signal.SIGTERM)
    try:
        if hasattr(os, "killpg") and os.getpgrp() == pid:
            os.killpg(pid, termination)
            return
    except OSError:
        pass
    try:
        os.kill(pid, termination)
    except OSError:
        os._exit(1)


def watch_parent(fd):
    try:
        while True:
            try:
                if not os.read(fd, 4096):
                    break
            except InterruptedError:
                continue
            except OSError:
                break
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    # SIGKILL is deliberate: launcher signal handlers or blocked generation
    # cannot keep the worker/its descendants alive after Werk has exited.
    stop_owned_worker()


def main():
    if len(sys.argv) < 2:
        raise SystemExit("oMLX supervisor requires a validated launcher path")
    launcher = sys.argv[1]
    sys.argv = sys.argv[1:]
    if not (getattr(sys.flags, "safe_path", False) or sys.flags.isolated):
        sys.path[0] = str(Path(launcher).resolve().parent)
    parent_fd = os.dup(sys.stdin.fileno())
    os.set_inheritable(parent_fd, False)
    threading.Thread(target=watch_parent, args=(parent_fd,), daemon=True).start()
    runpy.run_path(launcher, run_name="__main__")


if __name__ == "__main__":
    main()
