from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from backend.rag_service import RagService
from backend.settings import get_settings


class HealthResponse(BaseModel):
    status: str = "ok"


class AnswerResponse(BaseModel):
    session_id: str
    draft_answer: str
    verification_report: str


def create_app() -> FastAPI:
    """Create and configure the FastAPI backend application."""
    app = FastAPI(title="Agentic RAG Backend", version="0.1.0")
    service = RagService()

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Return a simple health signal for the backend service."""
        return HealthResponse()

    @app.post("/api/ask", response_model=AnswerResponse)
    async def ask(
        question: str = Form(...),
        session_id: str | None = Form(default=None),
        files: list[UploadFile] = File(...),
    ) -> AnswerResponse:
        """Answer a question using uploaded documents and session state."""
        try:
            result = await service.answer(question=question, files=files, session_id=session_id)
            return AnswerResponse(**result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Internal server error") from exc

    return app


def run() -> None:
    """Run the FastAPI backend with Uvicorn."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        create_app(),
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
