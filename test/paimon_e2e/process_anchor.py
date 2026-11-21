"""Private portable group identity anchor; no adoption, guardian or lease."""

import os
import select
import struct
import subprocess
import sys


LAUNCH_FAILED = b"L"


def main():
    control, status = int(sys.argv[1]), int(sys.argv[2])
    # This helper is the session/group leader. The C++ command stays in that
    # group, but cannot inherit either private channel.
    try:
        child = subprocess.Popen(sys.argv[3:], close_fds=True)
    except OSError as error:
        print(f"Cannot launch C++ test binary: {error}", file=sys.stderr, flush=True)
        outcome = LAUNCH_FAILED
    else:
        while child.poll() is None:
            if select.select([control], [], [], 0.02)[0]:
                os.read(control, 1)
                # Single reaper: an unreaped direct child still reserves its PID.
                child.kill()
                break
        outcome = struct.pack("!i", child.wait())
    try:
        if os.write(status, outcome) != len(outcome):
            raise RuntimeError("Incomplete child status write")
    finally:
        os.close(status)
    # Parent must signal this group before reaping this anchor. Never translate
    # the anchor's eventual SIGKILL into a C++ result.
    while os.read(control, 1):
        pass


if __name__ == "__main__":
    main()
