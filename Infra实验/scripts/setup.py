"""Bootstrap using any installed Python >=3.8, without changing shell policies."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="3.11", help="Python 3.11 executable or version")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if not shutil.which("uv"):
        raise SystemExit("Install uv: https://docs.astral.sh/uv/getting-started/installation/")
    python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    candidate = args.python
    if os.name == "nt" and candidate == "3.11":
        local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Python/Python311/python.exe"
        if local.is_file():
            candidate = str(local)
    uv = ["uv", "--cache-dir", str(root / ".cache/uv")]

    def call(command):
        subprocess.run(command, cwd=root, check=True)

    if not python.is_file():
        call(uv + ["venv", "--python", candidate, str(root / ".venv")])
    call([str(python), "-c", "import sys; assert sys.version_info[:2] == (3, 11), 'Expected Python 3.11'"])
    call(uv + ["pip", "install", "--python", str(python), "torch==2.9.1",
               "--index-url", "https://download.pytorch.org/whl/cu128"])
    triton = "triton-windows==3.5.1.post24" if os.name == "nt" else "triton==3.5.1"
    call(uv + ["pip", "install", "--python", str(python), triton, "numpy==2.2.6"])
    call([str(python), "-m", "kernellab", "doctor", "--smoke", "--cuda-smoke"])
    print("Ready. Run .\\lab.cmd list on Windows, or .venv/bin/python -m kernellab list on Linux.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
