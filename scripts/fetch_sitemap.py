import argparse
import re
import sys
from pathlib import Path

import requests


SITEMAP_URL = "https://support.avito.ru/sitemap.xml"
ARTICLE_PATTERN = re.compile(r"https://support\.avito\.ru/articles/(\d+)")


def fetch_sitemap(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def extract_article_ids(xml_text: str) -> list[int]:
    ids = []
    seen = set()
    for match in ARTICLE_PATTERN.finditer(xml_text):
        aid = int(match.group(1))
        if aid not in seen:
            seen.add(aid)
            ids.append(aid)
    ids.sort()
    return ids


def save_ids(ids: list[int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(str(i) for i in ids), encoding="utf-8")
    print(f"Saved {len(ids)} article IDs to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Avito support sitemap and extract article IDs")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/article_ids.txt"),
        help="Output file for article IDs",
    )
    args = parser.parse_args()

    print(f"Fetching sitemap from {SITEMAP_URL} ...")
    xml = fetch_sitemap(SITEMAP_URL)

    ids = extract_article_ids(xml)
    print(f"Found {len(ids)} unique article IDs")

    save_ids(ids, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
