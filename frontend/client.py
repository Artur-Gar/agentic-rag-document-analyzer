from pathlib import Path

import httpx

from frontend.config import get_frontend_settings


class BackendClient:
    def __init__(self) -> None:
        """Initialize the HTTP client wrapper for the backend API."""
        settings = get_frontend_settings()
        self.base_url = settings.frontend_backend_url.rstrip("/")

    def health(self) -> bool:
        """Check whether the backend health endpoint is reachable."""
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=5.0)
            response.raise_for_status()
            return response.json().get("status") == "ok"
        except Exception:
            return False

    def ask(self, question: str, documents: list[Path | tuple[str, bytes]], session_id: str) -> dict:
        """Submit a question and files to the backend API."""
        file_handles = []
        try:
            files = []
            for document in documents:
                if isinstance(document, Path):
                    file_handle = document.open("rb")
                    file_handles.append(file_handle)
                    files.append(("files", (document.name, file_handle, "application/octet-stream")))
                else:
                    filename, content = document
                    files.append(("files", (filename, content, "application/octet-stream")))

            response = httpx.post(
                f"{self.base_url}/api/ask",
                data={"question": question, "session_id": session_id},
                files=files,
                timeout=None,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            try:
                payload = exc.response.json()
                message = payload.get("detail", exc.response.text)
            except Exception:
                message = exc.response.text
            raise RuntimeError(message) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "The frontend could not reach the backend. Start the backend or check FRONTEND_BACKEND_URL."
            ) from exc
        finally:
            for file_handle in file_handles:
                file_handle.close()
