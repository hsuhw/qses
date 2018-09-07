import os
import sys

from contextlib import contextmanager


@contextmanager
def nostderr():
    stderr_fd = sys.stderr.fileno()

    def _redirect_stderr(to):
        sys.stderr.close()
        os.dup2(to.fileno(), stderr_fd)
        sys.stderr = os.fdopen(stderr_fd, 'w')

    with os.fdopen(os.dup(stderr_fd), 'w') as original_stderr:
        with open(os.devnull, 'w') as null_file:
            _redirect_stderr(null_file)
        try:
            yield
        finally:
            _redirect_stderr(original_stderr)
