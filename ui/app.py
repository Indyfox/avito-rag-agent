import os
import sys

from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr

from src.ingestion.embeddings import get_embedding_model, load_collection
from src.retrieval.retriever import Retriever
from src.generation.rag_pipeline import RAGPipeline

chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")

model = get_embedding_model(model_name)
collection = load_collection(persist_dir=chroma_dir)
retriever = Retriever(collection=collection, model=model, top_k=5)
pipeline = RAGPipeline(retriever=retriever, llm_model=llm_model)


def ask(question: str, history: list | None = None) -> str:
    if not question.strip():
        return "Пожалуйста, задайте вопрос."

    result = pipeline.query(question)

    sources_md = "\n\n---\n**Источники:**\n"
    for s in result["sources"]:
        sources_md += f"- [{s['title']}]({s['url']}) (релевантность: {s['relevance']:.2f})\n"

    return result["answer"] + sources_md


demo = gr.ChatInterface(
    fn=ask,
    title="AI-ассистент Авито",
    description="Задайте вопрос о правилах и функциях Авито. Ответ основан на официальных статьях поддержки.",
    theme="soft",
    examples=[
        "Как вернуть товар после покупки?",
        "Как пройти проверку профиля?",
        "Что такое Авито Доставка и как она работает?",
        "Как заблокировать нежелательного пользователя?",
        "Как изменить номер телефона в профиле?",
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
