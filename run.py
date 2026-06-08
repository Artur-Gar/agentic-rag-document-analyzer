from threading import Thread

import uvicorn

from backend.api import create_app
from backend.settings import get_settings as get_backend_settings
from frontend.launcher import main as run_frontend


def _run_backend() -> None:
    """Start the backend API server in the current process."""
    settings = get_backend_settings()
    uvicorn.run(
        create_app(),
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )


def main() -> None:
    """Launch the backend in a thread and then start the frontend UI."""
    backend_thread = Thread(target=_run_backend, daemon=True)
    backend_thread.start()
    run_frontend()


if __name__ == "__main__":
    main()
