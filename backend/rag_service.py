import hashlib
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import UploadFile
from langchain.retrievers import EnsembleRetriever

from backend.agent_workflow import AgentWorkflow
from backend.document_processing import DocumentProcessor, RetrieverBuilder
from backend.log_config import logger
from backend.settings import get_settings


@dataclass
class RetrieverSession:
    file_hashes: frozenset[str]
    retriever: EnsembleRetriever


class SessionStore:
    def __init__(self) -> None:
        """Initialize the in-memory store for per-session retrievers."""
        self._sessions: dict[str, RetrieverSession] = {}
        self._lock = Lock()

    def get(self, session_id: str) -> RetrieverSession | None:
        """Return the cached retriever session for a given session id."""
        with self._lock:
            return self._sessions.get(session_id)

    def upsert(
        self,
        session_id: str,
        file_hashes: frozenset[str],
        retriever: EnsembleRetriever,
    ) -> None:
        """Insert or replace the cached retriever for a session."""
        with self._lock:
            self._sessions[session_id] = RetrieverSession(
                file_hashes=file_hashes,
                retriever=retriever,
            )


class RagService:
    def __init__(self) -> None:
        """Wire together uploads, document processing, retrieval, and workflow steps."""
        self.settings = get_settings()
        self.settings.upload_directory.mkdir(parents=True, exist_ok=True)
        self.processor = DocumentProcessor()
        self.retriever_builder = RetrieverBuilder()
        self.workflow = AgentWorkflow()
        self.sessions = SessionStore()

    async def answer(
        self,
        question: str,
        files: list[UploadFile],
        session_id: str | None = None,
    ) -> dict[str, str]:
        """Answer a user question by reusing or rebuilding session retrieval state."""
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        if not files:
            raise ValueError("At least one document is required.")

        active_session_id = session_id or uuid4().hex
        stored_paths, file_hashes = await self._persist_uploads(active_session_id, files)

        session = self.sessions.get(active_session_id)
        if session is None or session.file_hashes != file_hashes:
            logger.info("Building retriever for session {}", active_session_id)
            chunks = self.processor.process(stored_paths)
            if not chunks:
                raise ValueError("No supported content could be extracted from the uploaded files.")

            retriever = self.retriever_builder.build_hybrid_retriever(
                chunks,
                session_key=active_session_id,
            )
            self.sessions.upsert(active_session_id, file_hashes, retriever)
        else:
            logger.info("Reusing cached retriever for session {}", active_session_id)
            retriever = session.retriever

        result = self.workflow.full_pipeline(question=question, retriever=retriever)
        return {
            "session_id": active_session_id,
            "draft_answer": result["draft_answer"],
            "verification_report": result["verification_report"],
        }

    async def _persist_uploads(
        self,
        session_id: str,
        files: list[UploadFile],
    ) -> tuple[list[Path], frozenset[str]]:
        """Save uploaded files to disk and return their paths and content hashes."""
        session_dir = self.settings.upload_directory / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        stored_paths: list[Path] = []
        file_hashes: set[str] = set()
        total_size = 0

        for uploaded_file in files:
            filename = Path(uploaded_file.filename or "uploaded-file").name
            suffix = Path(filename).suffix.lower()
            if suffix not in self.settings.allowed_extensions:
                raise ValueError(f"Unsupported file type: {suffix or filename}")

            content = await uploaded_file.read()
            file_size = len(content)
            total_size += file_size

            if file_size > self.settings.max_file_size_bytes:
                raise ValueError(
                    f"{filename} exceeds the {self.settings.max_file_size_mb} MB file size limit."
                )
            if total_size > self.settings.max_total_size_bytes:
                raise ValueError(
                    f"Total upload size exceeds the {self.settings.max_total_size_mb} MB limit."
                )

            file_hash = hashlib.sha256(content).hexdigest()
            target_path = session_dir / f"{file_hash}_{filename}"
            target_path.write_bytes(content)

            stored_paths.append(target_path)
            file_hashes.add(file_hash)
            await uploaded_file.seek(0)

        return stored_paths, frozenset(file_hashes)
