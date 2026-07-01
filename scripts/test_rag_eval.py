import os
from dotenv import load_dotenv

load_dotenv()

from src.ingestion.embeddings import get_embedding_model, load_collection
from src.retrieval.retriever import HybridRetriever
from src.generation.rag_pipeline import RAGPipeline

model = get_embedding_model(os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"))
collection = load_collection()
retriever = HybridRetriever(collection=collection, model=model, top_k=10)
pipeline = RAGPipeline(retriever=retriever, llm_model=os.getenv("LLM_MODEL", "openai/gpt-4o-mini"))

questions = [
    "В каком случае обратная доставка при возврате товара будет платной?",
    "Что делать, если товар заказан через постамат и он повреждён?",
    "Сколько времени у продавца на проверку возвращённого товара?",
]

facts = [
    "ФАКТ ИЗ СТАТЬИ: обратная доставка платная, если процент выкупа ниже 60% и заказ уже доставили в пункт выдачи",
    "ФАКТ ИЗ СТАТЬИ: заказ через постамат/5Post нужно забрать в любом случае, даже если повреждён. Затем фото и обращение в поддержку.",
    "ФАКТ ИЗ СТАТЬИ: у продавца от 7 до 14 дней чтобы забрать товар, и ещё 2 рабочих дня на проверку. Деньги возвращаются в течение 5 рабочих дней.",
]

for q, fact in zip(questions, facts):
    print("=" * 70)
    print(f"ВОПРОС: {q}")
    print(f"СТАТЬЯ: {fact}")
    print("-" * 70)
    result = pipeline.query(q, top_k=3)
    print(f"ОТВЕТ АГЕНТА:")
    print(result["answer"])
    print("-" * 70)
    print("РЕЛЕВАНТНЫЕ СТАТЬИ:")
    for s in result["sources"]:
        print(f"  [{s['relevance']:.2f}] {s['title']}")
    print()
