from typing import Dict

from langchain.schema import Document

from backend.log_config import logger
from backend.settings import get_settings
from backend.openai_client import build_model


class ResearchAgent:
    """Generate draft answers from retrieved document context."""

    def __init__(self) -> None:
        """Initialize the model used to draft answers from retrieved context."""
        settings = get_settings()
        self.model = build_model(
            model_id=settings.research_model_id,
            max_tokens=300,
            temperature=0.3,
        )

    @staticmethod
    def generate_prompt(question: str, context: str) -> str:
        """Build the prompt used to generate a draft answer."""
        return f"""
You are an AI assistant designed to provide precise and factual answers based on the given context.

Instructions:
- Answer the following question using only the provided context.
- Be clear, concise, and factual.
- Return as much information as you can get from the context.

Question: {question}
Context:
{context}

Provide your answer below:
"""

    def generate(self, question: str, documents: list[Document]) -> Dict[str, str]:
        """Generate a draft answer and keep track of the source context."""
        logger.info(
            "Research agent generating answer for question with {} retrieved documents",
            len(documents),
        )
        context = "\n\n".join(doc.page_content for doc in documents)
        prompt = self.generate_prompt(question, context)

        try:
            response = self.model.chat(messages=[{"role": "user", "content": prompt}])
            llm_response = response["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.exception("Error during research generation: {}", exc)
            raise RuntimeError("Failed to generate answer due to a model error.") from exc

        draft_answer = llm_response.strip() if llm_response else (
            "I cannot answer this question based on the provided documents."
        )

        return {
            "draft_answer": draft_answer,
            "context_used": context,
        }
