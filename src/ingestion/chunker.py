from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunker(
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: list[str] | None = None,
) -> RecursiveCharacterTextSplitter:
    if separators is None:
        separators = [
            "\n## ", "\n### ", "\n#### ", "\n**", "\n\n", "\n", " ", ""
        ]
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict]:
    splitter = create_chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []

    for doc in documents:
        doc_chunks = splitter.split_text(doc["text"])
        for i, chunk_text in enumerate(doc_chunks):
            chunks.append({
                "text": chunk_text,
                "article_id": doc["id"],
                "title": doc["title"],
                "url": doc["url"],
                "chunk_index": i,
            })

    return chunks
