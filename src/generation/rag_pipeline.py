from src.retrieval.retriever import HybridRetriever
from src.generation.llm_client import generate_response
from src.generation.prompt_templates import SYSTEM_PROMPT, build_user_message


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, llm_model: str = "openai/gpt-4o-mini"):
        self.retriever = retriever
        self.llm_model = llm_model

    def query(self, question: str, top_k: int = 10) -> dict:
        chunks = self.retriever.retrieve(question)

        if not chunks:
            return {
                "question": question,
                "answer": "К сожалению, в базе знаний не найдено информации по вашему вопросу.",
                "sources": [],
            }

        user_message = build_user_message(question, chunks)
        answer = generate_response(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            model=self.llm_model,
        )

        sources = [
            {
                "title": c["title"],
                "url": c["url"],
                "relevance": round(c["score"], 4),
            }
            for c in chunks
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }
