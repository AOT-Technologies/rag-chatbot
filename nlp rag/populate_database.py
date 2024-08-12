import argparse
import os
import shutil
from loader import DataLoader, PdfDataLoader, WebCrawlerDataLoader


# CHROMA_PATH = "chroma"
# DATA_PATH = "data"


def main():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--source", default="PDF", help="Source of the data.")
    parser.add_argument("--url", default="", help="Web URL to crawl.")
    parser.add_argument("--limit", default=None, help="Limit number of pages to scroll.")
    args = parser.parse_args()
    reset_db = args.reset
    source = args.source
    url = args.url if source == 'WEB' else None
    path = 'chroma/web' if source == 'WEB' else 'chroma/pdf'
    if reset_db:
        print("✨ Clearing Database")
        clear_database(path)

    # Create (or update) the data store.
    if source == 'PDF':
        loader = PdfDataLoader(path)
    else:
        loader = WebCrawlerDataLoader(path, url=url, limit=int(args.limit) if args.limit else None)
    loader.load_data()


def clear_database(path):
    if os.path.exists(path):
        shutil.rmtree(path)


if __name__ == "__main__":
    main()
