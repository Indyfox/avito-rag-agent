# Avito RAG Agent

AI-ассистент службы поддержки Авито на базе RAG (Retrieval-Augmented Generation). Отвечает на вопросы пользователей, используя официальные статьи с `support.avito.ru`.

## Архитектура

```
articles          embeddings        retrieval         generation
   │                  │                 │                  │
   ▼                  ▼                 ▼                  ▼
[.md files]  →  [ChromaDB]  →  [FAISS/HNSW]  →  [OpenAI GPT-4o-mini]
   │                                                  │
   ▼                                                  ▼
scrape_articles.py                              UI (Gradio)
```

## Tech Stack

| Слой | Технологии |
|------|-----------|
| Embeddings | `sentence-transformers` + `intfloat/multilingual-e5-small` |
| Vector DB | `ChromaDB` (persistent, HNSW) |
| LLM | `OpenAI GPT-4o-mini` (API) |
| Chunking | `langchain-text-splitters` (RecursiveCharacterTextSplitter) |
| Парсинг | `requests` + `beautifulsoup4` (HTML→MD) |
| API | `FastAPI` + `uvicorn` |
| UI | `Gradio` |
| Eval | `ragas` |

## Быстрый старт

### 1. Установка

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Настройка

```bash
copy .env.example .env
# Отредактируй .env — добавь OPENAI_API_KEY
```

### 3. Дамп статей (или используй тестовые)

```bash
# С реальных данных:
python scripts/fetch_sitemap.py
python scripts/scrape_articles.py

# Или тестовые 3 статьи:
python scripts/create_sample_data.py
```

### 4. Построение индекса (с тестовыми статьями)

```bash
python run.py build --articles-dir data/sample
```

### 5. Запуск UI

```bash
python ui/app.py
```

Открой `http://localhost:7860`

### Альтернативно: CLI / API

```bash
python run.py chat        # Чат в терминале
python run.py api         # FastAPI на :8000
```

## Структура проекта

```
avito-rag-agent/
├── scripts/
│   ├── fetch_sitemap.py        # Парсинг sitemap → список ID статей
│   ├── scrape_articles.py      # POST /api/1/article → .md файлы
│   └── create_sample_data.py   # 3 тестовые статьи для демо
├── src/
│   ├── ingestion/
│   │   ├── loader.py           # Загрузка .md → list[dict]
│   │   ├── chunker.py          # RecursiveCharacterTextSplitter
│   │   ├── embeddings.py       # SentenceTransformer + ChromaDB
│   │   └── pipeline.py         # Полный пайплайн индексации
│   ├── retrieval/
│   │   └── retriever.py        # ChromaDB query + similarity search
│   ├── generation/
│   │   ├── llm_client.py       # OpenAI API client
│   │   ├── prompt_templates.py # Системный промпт + шаблон
│   │   └── rag_pipeline.py     # Retrieve → Augment → Generate
│   └── api/
│       ├── main.py             # FastAPI server
│       └── schemas.py          # Pydantic models
├── ui/
│   └── app.py                  # Gradio ChatInterface
├── tests/
├── notebooks/                  # Jupyter для экспериментов
├── run.py                      # CLI: build | chat | api
├── requirements.txt
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| `GET` | `/health` | Статус сервера |
| `POST` | `/query` | Запрос к RAG: `{"question": "...", "top_k": 5}` |

Пример ответа:
```json
{
  "question": "Как вернуть товар?",
  "answer": "Чтобы вернуть товар...\n\n[Источник: статья 4563]",
  "sources": [
    {"title": "Как отменить или изменить заказ", "url": "...", "relevance": 0.95}
  ]
}
```

## Параметры (`.env`)

| Переменная | По умолчанию | Описание |
|-----------|-------------|----------|
| `OPENAI_API_KEY` | — | API ключ OpenAI (обязательно) |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Модель эмбеддингов |
| `LLM_MODEL` | `gpt-4o-mini` | LLM для генерации |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Директория ChromaDB |
