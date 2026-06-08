from typing import Dict

from langchain.schema import Document

from backend.log_config import logger
from backend.settings import get_settings
from backend.openai_client import build_model


class VerificationAgent:
    """Verify draft answers against the retrieved document context."""

    def __init__(self) -> None:
        """Initialize the model used to verify draft answers."""
        settings = get_settings()
        self.model = build_model(
            model_id=settings.verification_model_id,
            max_tokens=200,
            temperature=0,
        )

    @staticmethod
    def generate_prompt(answer: str, context: str) -> str:
        """Build the prompt used to verify an answer against context."""
        return f"""
You are an AI assistant designed to verify the accuracy and relevance of answers based on provided context.

Instructions:
- Verify the following answer against the provided context.
- Check for:
1. Direct/indirect factual support (YES/NO)
2. Unsupported claims (list any if present)
3. Contradictions (list any if present)
4. Relevance to the question (YES/NO)
- Provide additional details or explanations where relevant.
- Respond in the exact format specified below without adding any unrelated information.

Format:
Supported: YES/NO
Unsupported Claims: [item1, item2, ...]
Contradictions: [item1, item2, ...]
Relevant: YES/NO
Additional Details: [Any extra information or explanations]

Answer: {answer}
Context:
{context}

Respond ONLY with the above format.
"""

    @staticmethod
    def parse_verification_response(response_text: str) -> Dict:
        """Parse the verification model output into a structured dictionary."""
        verification: Dict[str, object] = {}
        for line in response_text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            normalized_key = key.strip().lower()
            raw_value = value.strip()

            if normalized_key in {"unsupported claims", "contradictions"}:
                if raw_value.startswith("[") and raw_value.endswith("]"):
                    items = raw_value[1:-1].split(",")
                    verification[normalized_key] = [
                        item.strip().strip('"').strip("'")
                        for item in items
                        if item.strip()
                    ]
                else:
                    verification[normalized_key] = []
            else:
                verification[normalized_key] = raw_value

        return {
            "Supported": str(verification.get("supported", "NO")).upper(),
            "Unsupported Claims": verification.get("unsupported claims", []),
            "Contradictions": verification.get("contradictions", []),
            "Relevant": str(verification.get("relevant", "NO")).upper(),
            "Additional Details": str(verification.get("additional details", "")),
        }

    @staticmethod
    def format_verification_report(verification: Dict) -> str:
        """Render the parsed verification result into a readable text report."""
        unsupported_claims = verification.get("Unsupported Claims", [])
        contradictions = verification.get("Contradictions", [])
        return "\n".join(
            [
                f"Supported: {verification.get('Supported', 'NO')}",
                "Unsupported Claims: "
                + (", ".join(unsupported_claims) if unsupported_claims else "None"),
                "Contradictions: "
                + (", ".join(contradictions) if contradictions else "None"),
                f"Relevant: {verification.get('Relevant', 'NO')}",
                "Additional Details: "
                + (verification.get("Additional Details") or "None"),
            ]
        )

    def check(self, answer: str, documents: list[Document]) -> Dict[str, str]:
        """Verify a draft answer against the retrieved documents."""
        logger.info(
            "Verification agent validating answer against {} documents",
            len(documents),
        )
        context = "\n\n".join(doc.page_content for doc in documents)
        prompt = self.generate_prompt(answer, context)

        try:
            response = self.model.chat(messages=[{"role": "user", "content": prompt}])
            llm_response = response["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.exception("Error during verification generation: {}", exc)
            raise RuntimeError("Failed to verify answer due to a model error.") from exc

        if llm_response:
            verification = self.parse_verification_response(llm_response.strip())
        else:
            verification = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": "Empty response from the model.",
            }

        return {
            "verification_report": self.format_verification_report(verification),
            "context_used": context,
        }
