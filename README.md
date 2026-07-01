<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/OpenAI-API-green?style=flat&logo=openai" alt="OpenAI">
  <img src="https://img.shields.io/badge/RAG-Hybrid%20(BM25%20%2B%20Dense)-orange?style=flat" alt="RAG">
  <img src="https://img.shields.io/badge/ChromaDB-Persistent-purple?style=flat" alt="ChromaDB">
  <img src="https://img.shields.io/badge/UI-Gradio-ff69b4?style=flat" alt="Gradio">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat" alt="License">
</p>

<h1 align="center">Avito RAG Agent</h1>
<p align="center"><strong>AI-ассистент по статьям поддержки Авито<br>
Портфолио-проект для буткемпа Avito Data Science (NLP / LLM)</strong></p>

---

## Оглавление

- [О проекте](#о-проекте)
- [Архитектура](#архитектура)
- [Технологии](#технологии)
- [Демонстрация](#демонстрация)
- [Быстрый старт](#быстрый-старт)
- [Запуск на реальных данных](#запуск-на-реальных-данных)
- [CLI и API](#cli-и-api)
- [Оценка качества](#оценка-качества)
- [Структура проекта](#структура-проекта)
- [API Endpoints](#api-endpoints)
- [Конфигурация](#конфигурация)
- [Дорожная карта](#дорожная-карта)

---

## О проекте

Avito RAG Agent — это интеллектуальный ассистент, отвечающий на вопросы пользователей на основе **703 официальных статей** службы поддержки Авито (`support.avito.ru`).

Проект реализует полный RAG-пайплайн:

1. **Ingestion** — парсинг веб-страниц, очистка HTML, гранулированное разбиение на чанки (~10k чанков)
2. **Retrieval** — гибридный поиск (BM25 + dense embeddings), объединённый через Reciprocal Rank Fusion
3. **Generation** — LLM-генерация ответа с цитированием источников

Разработан как портфолио-проект для поступления на буткемп **Avito Data Science (NLP / LLM направление)**.

---

## Архитектура

```
                        ┌─────────────────────────────────────────────┐
                        │              DATA PIPELINE                  │
                        │                                             │
  sitemap.xml ─────────►│  fetch_sitemap.py  ───► article_ids.txt     │
                        │         │                                   │
                        │         ▼                                   │
                        │  scrape_articles.py  ───►  *.md (703 шт)   │
                        │         │                                   │
                        │         ▼                                   │
                        │  chunker.py (1k ch)  ───►  ~10k chunks      │
                        │         │                                   │
                        └─────────┼───────────────────────────────────┘
                                  │
                                  ▼
              ┌────────────────────────────────────────────────────┐
              │                   INDEX                            │
              │                                                    │
              │  intfloat/multilingual-e5-small                    │
              │         │                │                         │
              │         ▼                ▼                         │
              │  ChromaDB (vectors)    BM25 (keywords)             │
              │         │                │                         │
              └─────────┼────────────────┼─────────────────────────┘
                        │                │
                        ▼                ▼
              ┌────────────────────────────────────────────────────┐
              │               RUNTIME                              │
              │                                                     │
              │  User Query ──►                                     │
              │       │                                             │
              │       ├──► BM25 search                              │
              │       ├──► ChromaDB similarity search               │
              │       │       │                                     │
              │       ├────── RRF Fusion ──► top-10 chunks          │
              │       │               │                             │
              │       ▼               ▼                             │
              │  OpenAI GPT-4o-mini (через proxy)                   │
              │       │                                             │
              │       ▼                                             │
              │  Gradio UI ◄──── Answer + Sources                   │
              └─────────────────────────────────────────────────────┘
```

---

## Технологии

| Слой | Технология | Версия |
|------|-----------|--------|
| Язык | Python | 3.10+ |
| Embeddings | `intfloat/multilingual-e5-small` (sentence-transformers) | 118M params |
| Hybrid Search | `rank-bm25` + ChromaDB (cosine) | RRF fusion |
| Векторная БД | ChromaDB | Persistent, HNSW |
| LLM | OpenAI GPT-4o-mini (через routerai.ru) | 8k context |
| Chunking | `langchain-text-splitters` (RecursiveCharacter) | 1024/128 |
| UI | Gradio | ChatInterface |
| API | FastAPI + Uvicorn | REST |
| Парсинг | requests + BeautifulSoup | HTML→Markdown |

---

## Демонстрация

Пример работы агента:

<table>
<tr>
<th>Вопрос пользователя</th>
<th>Ответ агента</th>
<th>Результат</th>
</tr>
<tr>
<td><em>В каком случае обратная доставка при возврате товара будет платной?</em></td>
<td>Возврат платный, если процент выкупа ниже 60%, а заказ уже доставлен в пункт выдачи. Если отказались от товара при получении или не пришли за заказом.</td>
<td>✅ PASS</td>
</tr>
<tr>
<td><em>Что делать, если товар заказан через постамат и он повреждён?</em></td>
<td>Забрать посылку обязательно, сфотографировать товар со всех сторон, обратиться в службу поддержки.</td>
<td>✅ PASS</td>
</tr>
<tr>
<td><em>Сколько времени у продавца на проверку возвращённого товара?</em></td>
<td>2 рабочих дня после получения товара. До этого у продавца есть 7–14 дней, чтобы забрать товар из пункта выдачи.</td>
<td>✅ PASS</td>
</tr>
</table>

---

## Быстрый старт

### 1. Клонирование и установка

```bash
git clone https://github.com/Indyfox/avito-rag-agent.git
cd avito-rag-agent
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Настройка окружения

```bash
copy .env.example .env
```

Отредактируйте `.env`:

```ini
OPENAI_API_KEY=sk-xxxx                # Ваш API-ключ OpenAI
OPENAI_BASE_URL=https://routerai.ru/api/v1  # Прокси (если недоступен прямой API)
LLM_MODEL=openai/gpt-4o-mini          # Модель для генерации
EMBEDDING_MODEL=intfloat/multilingual-e5-small  # Модель эмбеддингов
```

### 3. Создание тестовых данных и запуск

```bash
# Создать 3 тестовые статьи
python scripts/create_sample_data.py

# Построить индекс (на тестовых данных)
python run.py build --articles-dir data/sample

# Запустить UI
python ui/app.py
```

Откройте **http://localhost:7860** в браузере.

---

## Запуск на реальных данных

Проект включает скрипты для сбора всех статей с `support.avito.ru`:

```bash
# Шаг 1: Скачать sitemap → извлечь ID всех 718 статей
python scripts/fetch_sitemap.py

# Шаг 2: Скрапить все статьи через API (занимает ~8 минут)
python scripts/scrape_articles.py --delay 0.3

# Шаг 3: Построить индекс (703 статьи, ~10k чанков)
python run.py build

# Шаг 4: Запустить UI
python ui/app.py
```

> ⚠️ Перед скрапингом убедитесь, что у вас есть доступ к `support.avito.ru`.

---

## CLI и API

### Режим чата в терминале

```bash
python run.py chat
```

### FastAPI сервер

```bash
python run.py api
# Сервер запущен на http://localhost:8000
# Документация: http://localhost:8000/docs (Swagger)
```

---

## Оценка качества

Для проверки качества RAG-пайплайна используется скрипт:

```bash
$env:PYTHONPATH="."; python scripts/test_rag_eval.py
```

Результаты тестирования (3 ключевых вопроса по статье **"Отказаться от товара или вернуть его"**):

| # | Вопрос | Точность | Комментарий |
|---|--------|----------|-------------|
| 1 | Условия платной обратной доставки | ✅ | Верно указан порог 60% выкупа |
| 2 | Действия при повреждённом товаре в постамате | ✅ | Верно: забрать → сфоткать → поддержка |
| 3 | Срок проверки возврата продавцом | ✅ | Верно: 2 рабочих дня + 7–14 дней на получение |

**Архитектура retrieval:** Гибридный поиск (BM25 + dense через RRF) с `dense_weight=0.6`. Это обеспечивает устойчивость к неточным формулировкам запроса.

---

## Структура проекта

```
avito-rag-agent/
│
├── scripts/                          # Скрипты для сбора и оценки данных
│   ├── fetch_sitemap.py              #   Парсинг sitemap → ID статей
│   ├── scrape_articles.py            #   Скрапинг статей через API
│   ├── create_sample_data.py         #   Создание тестовых статей
│   └── test_rag_eval.py              #   Оценка качества RAG
│
├── src/                              # Исходный код
│   ├── ingestion/                    # Пайплайн индексации
│   │   ├── loader.py                 #   Загрузка .md → list[dict]
│   │   ├── chunker.py                #   RecursiveCharacterTextSplitter
│   │   ├── embeddings.py             #   SentenceTransformer + ChromaDB
│   │   └── pipeline.py               #   Полный пайплайн индексации
│   │
│   ├── retrieval/                    # Поиск
│   │   └── retriever.py              #   Гибридный поиск (BM25 + dense)
│   │
│   ├── generation/                   # Генерация
│   │   ├── llm_client.py             #   OpenAI API (с поддержкой proxy)
│   │   ├── prompt_templates.py       #   Системный промпт + шаблон
│   │   └── rag_pipeline.py           #   RAG: retrieve → augment → generate
│   │
│   └── api/                          # REST API
│       ├── main.py                   #   FastAPI сервер
│       └── schemas.py                #   Pydantic модели
│
├── ui/
│   └── app.py                        # Gradio ChatInterface
│
├── tests/
│   └── test_pipeline.py              # Unit-тесты
│
├── run.py                            # CLI: build | chat | api
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

---

## API Endpoints

### FastAPI (запуск: `python run.py api`)

| Метод | Endpoint | Описание |
|-------|----------|----------|
| `GET` | `/health` | Проверка состояния сервера |
| `POST` | `/query` | Запрос к RAG-пайплайну |

#### Пример запроса

```json
{
  "question": "Как вернуть товар после покупки?",
  "top_k": 5
}
```

#### Пример ответа

```json
{
  "question": "Как вернуть товар после покупки?",
  "answer": "Чтобы вернуть товар, следуйте этим шагам:\n\n1. **Проверьте значок** — если в объявлении есть «14 дней на возврат», товар можно вернуть.\n2. **Откройте раздел Заказы** и нажмите «Вернуть товар».\n3. **Сфотографируйте товар** и упакуйте его.\n4. **Сдайте в пункт выдачи**.\n\n[Источник: статья 4400]",
  "sources": [
    {
      "title": "Отказаться от товара или вернуть его",
      "url": "https://support.avito.ru/articles/4400",
      "relevance": 0.95
    },
    {
      "title": "Как продавать и покупать с доставкой",
      "url": "https://support.avito.ru/articles/4234",
      "relevance": 0.87
    }
  ]
}
```

---

## Конфигурация

### Переменные окружения (`.env`)

| Переменная | По умолчанию | Описание |
|-----------|-------------|----------|
| `OPENAI_API_KEY` | — | **Обязательно.** API-ключ OpenAI |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Базовый URL для OpenAI API (прокси) |
| `LLM_MODEL` | `openai/gpt-4o-mini` | Модель для генерации ответов |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Модель эмбеддингов (sentence-transformers) |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Директория для постоянного хранилища ChromaDB |

### Рекомендуемые альтернативы embedding-моделей

| Модель | Параметры | Скорость | Качество |
|--------|-----------|----------|----------|
| `intfloat/multilingual-e5-small` | 118M | ⚡ Высокая | Среднее |
| `intfloat/multilingual-e5-large` | 560M | 🐢 Средняя (CPU) | **Высокое** |
| `intfloat/multilingual-e5-large-instruct` | 560M | 🐢 Средняя (CPU) | **Очень высокое** |

> На CPU рекомендуется `e5-small`. Для GPU — `e5-large-instruct` даёт прирост качества на ~15%.

---

## Дорожная карта

- [x] Парсинг sitemap и статей поддержки Авито
- [x] Гибридный поиск (BM25 + dense embeddings)
- [x] Gradio ChatInterface
- [x] FastAPI + Swagger
- [x] CLI-режимы (build, chat, api)
- [x] Скрипт оценки качества RAG
- [ ] Re-ranker (cross-encoder) для повышения точности
- [ ] Асинхронная обработка запросов
- [ ] Docker-контейнеризация
- [ ] CI/CD (GitHub Actions)

---

<p align="center">
  <strong>Avito RAG Agent</strong><br>
  <sub>Портфолио-проект для буткемпа Avito Data Science (NLP / LLM)</sub><br>
  <a href="https://github.com/Indyfox/avito-rag-agent">github.com/Indyfox/avito-rag-agent</a>
</p>
