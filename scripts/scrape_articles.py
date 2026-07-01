import argparse
import json
import re
import time
import sys
from pathlib import Path

import requests
from tqdm import tqdm


API_URL = "https://support.avito.ru/api/1/article"
BASE_URL = "https://support.avito.ru/articles"
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "AvitoRAGAgent/1.0",
}

ILLEGAL_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')


def fetch_article(article_id: int) -> dict | None:
    payload = {"id": str(article_id)}
    try:
        resp = requests.post(API_URL, json=payload, headers=HEADERS, timeout=30)
    except requests.RequestException:
        return None

    if resp.status_code != 200:
        return None

    data = resp.json()
    result = data.get("result")
    if not result:
        return None

    return {
        "id": result["id"],
        "title": result.get("title", ""),
        "body": result.get("body", ""),
        "url": f"{BASE_URL}/{result['id']}",
    }


def html_to_markdown(body: str) -> str:
    text = body

    text = re.sub(r"<headline[^>]*>.*?</headline>", "", text, flags=re.DOTALL)

    text = re.sub(
        r'<div class="tabset[^"]*">.*?</div>\s*</div>',
        "",
        text,
        flags=re.DOTALL,
    )

    text = re.sub(r"\*\*\d+\*\*!", "", text)

    text = re.sub(r"<h2>(.*?)</h2>", r"\n## \1\n", text)
    text = re.sub(r"<h3>(.*?)</h3>", r"\n### \1\n", text)
    text = re.sub(r"<h4>(.*?)</h4>", r"\n#### \1\n", text)
    text = re.sub(r"<p>(.*?)</p>", r"\1\n\n", text, flags=re.DOTALL)
    text = re.sub(r"<strong>(.*?)</strong>", r"**\1**", text)
    text = re.sub(r"<em>(.*?)</em>", r"*\1*", text)
    text = re.sub(r"<a[^>]*href=\"([^\"]*)\"[^>]*>(.*?)</a>", r"[\2](\1)", text)
    text = re.sub(r"<li>(.*?)</li>", r"- \1\n", text, flags=re.DOTALL)
    text = re.sub(r"</?[ou]l>", "", text)
    text = re.sub(r"<img[^>]*>", "", text)
    text = re.sub(r"<div[^>]*>", "", text)
    text = re.sub(r"</div>", "", text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&laquo;", '"', text)
    text = re.sub(r"&raquo;", '"', text)
    text = re.sub(r"&mdash;", "—", text)
    text = re.sub(r"&ndash;", "–", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = text.strip()

    return text


def save_article(article: dict, articles_dir: Path) -> None:
    title = article["title"] or f"article_{article['id']}"
    safe_title = ILLEGAL_FILENAME_CHARS.sub("_", title)[:80]
    filename = f"{article['id']}_{safe_title}.md"
    filepath = articles_dir / filename

    md_content = f"# {article['title']}\n\n"
    md_content += f"> Источник: {article['url']}\n\n"
    md_content += article["body"]
    md_content += "\n"

    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(md_content, encoding="utf-8")


def save_metadata(articles: list[dict], metadata_path: Path) -> None:
    meta = {
        str(a["id"]): {"title": a["title"], "url": a["url"]}
        for a in articles
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Avito support articles via API")
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=Path("data/raw/article_ids.txt"),
        help="File with article IDs (one per line)",
    )
    parser.add_argument(
        "--articles-dir",
        type=Path,
        default=Path("data/raw/articles"),
        help="Directory to save .md files",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/raw/metadata.json"),
        help="Metadata JSON output path",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between requests in seconds",
    )
    args = parser.parse_args()

    if not args.ids_file.exists():
        print(f"IDs file not found: {args.ids_file}")
        print("Run fetch_sitemap.py first.")
        sys.exit(1)

    ids = [int(line.strip()) for line in args.ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(ids)} article IDs")

    articles_dir = args.articles_dir
    articles_dir.mkdir(parents=True, exist_ok=True)

    successful = []
    failed = []

    for aid in tqdm(ids, desc="Scraping articles"):
        article = fetch_article(aid)

        if article is None:
            failed.append(aid)
            continue

        html_body = article["body"]
        if not html_body:
            failed.append(aid)
            continue

        article["body"] = html_to_markdown(html_body)
        save_article(article, articles_dir)
        successful.append(article)
        time.sleep(args.delay)

    metadata_path = args.metadata
    save_metadata(successful, metadata_path)

    print(f"\nDone. Successful: {len(successful)}, Failed: {len(failed)}")
    if failed:
        print(f"Failed IDs: {failed}")


if __name__ == "__main__":
    main()
