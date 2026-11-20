"""Launch R normally; support site shell launchers without a valid shebang."""
import errno
import os
import subprocess


def run_r(command, **kwargs):
    try:
        return subprocess.run(command, **kwargs)
    except OSError as error:
        if os.name != "posix" or error.errno != errno.ENOEXEC:
            raise
        # Bash applies its script fallback just as an interactive shell does.
        # Arguments stay positional; paths/arguments are never interpolated as code.
        return subprocess.run(["/bin/bash", "-c", '"$@"', "--", *command], **kwargs)
