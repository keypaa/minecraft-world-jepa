"""Launch latent stats computation on Modal as a subprocess."""
import subprocess, os, sys, tempfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
log_dir = Path(os.environ.get("TEMP", tempfile.gettempdir()))
log_path = log_dir / "latent_stats.log"

env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
with open(log_path, "wb") as log:
    proc = subprocess.Popen(
        ["modal", "run", "app.py::compute_latent_stats"],
        cwd=str(root),
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    print(f"Latent stats launched (PID {proc.pid}), logging to {log_path}")
    sys.stdout.flush()
