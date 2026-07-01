import os

from openai import OpenAI


_llm_client: OpenAI | None = None


def get_llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        _llm_client = OpenAI(api_key=api_key)
    return _llm_client


def generate_response(
    system_prompt: str,
    user_message: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    client = get_llm_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""
