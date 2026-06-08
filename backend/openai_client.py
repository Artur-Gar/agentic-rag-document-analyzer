from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from backend.settings import get_settings


class OpenAIChatModel:

    def __init__(self, model_id: str, max_tokens: int, temperature: float) -> None:
        """Store model settings and initialize the shared OpenAI client."""
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

        settings = get_settings()
        client_kwargs: dict[str, str] = {}
        client_kwargs["api_key"] = settings.openai_api_key

        self.client = OpenAI(**client_kwargs)

    def chat(self, messages: list[dict]) -> dict:
        """Send chat-completion messages to OpenAI and return a dict response."""
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.model_dump()


def build_model(model_id: str, max_tokens: int, temperature: float) -> OpenAIChatModel:
    """Create a configured OpenAI chat model wrapper."""
    return OpenAIChatModel(
        model_id=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def build_embeddings() -> OpenAIEmbeddings:
    """Create the OpenAI embeddings client used for vector retrieval."""
    settings = get_settings()
    embedding_kwargs: dict[str, str] = {"model": settings.embedding_model_id}
    embedding_kwargs["api_key"] = settings.openai_api_key

    return OpenAIEmbeddings(**embedding_kwargs)
