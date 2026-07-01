def test_loader():
    from src.ingestion.loader import load_markdown_files
    from pathlib import Path

    sample_dir = Path("data/sample")
    if not sample_dir.exists():
        sample_dir = Path("scripts/../data/sample").resolve()
    if not sample_dir.exists():
        return

    docs = load_markdown_files(sample_dir)
    assert len(docs) > 0
    assert "title" in docs[0]
    assert "text" in docs[0]
    print(f"Loader OK: {len(docs)} articles loaded")


def test_chunker():
    docs = [{"id": "1", "title": "Test", "url": "", "text": "Hello world. " * 200}]
    from src.ingestion.chunker import chunk_documents

    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert "text" in c
        assert "article_id" in c
    print(f"Chunker OK: {len(chunks)} chunks")


def test_prompt():
    from src.generation.prompt_templates import build_user_message, SYSTEM_PROMPT

    chunks = [
        {"text": "Пример текста", "title": "Тестовая статья", "url": "https://example.com"},
    ]
    msg = build_user_message("тестовый вопрос", chunks)
    assert "тестовый вопрос" in msg
    assert "Пример текста" in msg
    assert "Тестовая статья" in msg
    print("Prompts OK")


if __name__ == "__main__":
    test_loader()
    test_chunker()
    test_prompt()
    print("All tests passed.")
