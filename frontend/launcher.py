import subprocess
import sys
from pathlib import Path

from frontend.config import get_frontend_settings


def main() -> None:
    """Launch the Streamlit frontend with the configured host and port."""
    settings = get_frontend_settings()
    app_path = Path(__file__).resolve().parent / "app.py"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.address",
            settings.frontend_host,
            "--server.port",
            str(settings.frontend_port),
            "--server.headless",
            "true",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
