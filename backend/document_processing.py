import hashlib
import pickle
import re
from datetime import datetime, timedelta
from pathlib import Path

from docling.document_converter import DocumentConverter
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter

from backend.log_config import logger
from backend.settings import get_settings
from backend.openai_client import build_embeddings


class DocumentProcessor:
    def __init__(self) -> None:
        """Initialize document processing with cache and splitter settings."""
        self.settings = get_settings()
        self.headers = [("#", "Header 1"), ("##", "Header 2")]
        self.cache_dir = self.settings.cache_directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def process(self, files: list[Path]) -> list:
        """Load, split, cache, and deduplicate chunks from uploaded files."""
        all_chunks = []
        seen_hashes: set[str] = set()

        for file_path in files:
            if file_path.suffix.lower() not in self.settings.allowed_extensions:
                logger.warning("Skipping unsupported file type: {}", file_path.name)
                continue

            try:
                file_hash = self._generate_hash(file_path.read_bytes())
                cache_path = self.cache_dir / f"{file_hash}.pkl"

                if self._is_cache_valid(cache_path):
                    logger.info("Loading from cache: {}", file_path.name)
                    chunks = self._load_from_cache(cache_path)
                else:
                    logger.info("Processing and caching: {}", file_path.name)
                    chunks = self._process_file(file_path)
                    self._save_to_cache(chunks, cache_path)

                for chunk in chunks:
                    chunk_hash = self._generate_hash(chunk.page_content.encode("utf-8"))
                    if chunk_hash not in seen_hashes:
                        all_chunks.append(chunk)
                        seen_hashes.add(chunk_hash)
            except Exception as exc:
                logger.exception("Failed to process {}: {}", file_path.name, exc)

        logger.info("Total unique chunks: {}", len(all_chunks))
        return all_chunks

    def _process_file(self, file_path: Path) -> list:
        """Convert one document into markdown chunks using Docling."""
        converter = DocumentConverter()
        markdown = converter.convert(str(file_path)).document.export_to_markdown()
        splitter = MarkdownHeaderTextSplitter(self.headers)
        return splitter.split_text(markdown)

    @staticmethod
    def _generate_hash(content: bytes) -> str:
        """Return a stable SHA-256 hash for the given content."""
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _save_to_cache(chunks: list, cache_path: Path) -> None:
        """Persist processed document chunks to the cache directory."""
        with cache_path.open("wb") as file_handle:
            pickle.dump(
                {
                    "timestamp": datetime.now().timestamp(),
                    "chunks": chunks,
                },
                file_handle,
            )

    @staticmethod
    def _load_from_cache(cache_path: Path) -> list:
        """Load previously cached document chunks from disk."""
        with cache_path.open("rb") as file_handle:
            data = pickle.load(file_handle)
        return data["chunks"]

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check whether a cached chunk file still falls within the TTL."""
        if not cache_path.exists():
            return False

        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < timedelta(days=self.settings.cache_expire_days)


class RetrieverBuilder:
    def __init__(self) -> None:
        """Initialize retriever dependencies and persistent vector storage."""
        self.settings = get_settings()
        self.embeddings = build_embeddings()
        self.settings.chroma_directory.mkdir(parents=True, exist_ok=True)

    def build_hybrid_retriever(self, docs: list, session_key: str) -> EnsembleRetriever:
        """Build a BM25 plus vector retriever for one session's documents."""
        try:
            collection_name = self._build_collection_name(session_key)
            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=str(self.settings.chroma_directory),
                collection_name=collection_name,
            )
            logger.info("Vector store created successfully for session {}", session_key)

            bm25 = BM25Retriever.from_documents(docs)
            logger.info("BM25 retriever created successfully for session {}", session_key)

            vector_retriever = vector_store.as_retriever(
                search_kwargs={"k": self.settings.vector_search_k}
            )
            logger.info("Vector retriever created successfully for session {}", session_key)

            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25, vector_retriever],
                weights=list(self.settings.hybrid_retriever_weights),
            )
            logger.info("Hybrid retriever created successfully for session {}", session_key)
            return hybrid_retriever
        except Exception as exc:
            logger.exception("Failed to build hybrid retriever: {}", exc)
            raise

    @staticmethod
    def _build_collection_name(session_key: str) -> str:
        """Create a safe Chroma collection name from a session id."""
        normalized = re.sub(r"[^a-zA-Z0-9_-]", "-", session_key)
        return f"documents-{normalized[:40]}"
