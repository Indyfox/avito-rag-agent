from pathlib import Path


def load_markdown_files(articles_dir: Path) -> list[dict]:
    documents = []
    for md_file in sorted(articles_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")

        lines = text.strip().split("\n")
        if not lines or not lines[0].startswith("# "):
            continue

        title = lines[0][2:].strip()
        source_url = ""
        if len(lines) > 2 and lines[2].startswith("> Источник:"):
            source_url = lines[2].replace("> Источник:", "").strip()

        content = ""
        content_started = False
        for line in lines:
            if content_started:
                content += line + "\n"
            if line.startswith("> Источник:"):
                content_started = True

        if not content.strip():
            continue

        article_id = md_file.stem.split("_")[0]
        documents.append({
            "id": article_id,
            "title": title,
            "url": source_url,
            "text": content.strip(),
        })

    return documents
