import os
import sys

# MediaPipe/TFLite log a wall of INFO lines on start-up; keep the console readable.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Long runs are often redirected to a log file; show progress lines as they happen.
sys.stdout.reconfigure(line_buffering=True)

from .cli import main  # noqa: E402

# The guard matters on Windows: worker processes re-import this module as
# "__mp_main__", and must not start another copy of the command.
if __name__ == "__main__":
    main()
