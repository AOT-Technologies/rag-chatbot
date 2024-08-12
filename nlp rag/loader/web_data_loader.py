import time

import html2text
import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain.vectorstores.chroma import Chroma
from langchain.schema import Document
from loader import DataLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib.parse import urlparse
from collections import defaultdict


class WebCrawlerDataLoader(DataLoader):

    def __init__(self, chroma_path, url=None, limit=None):
        super().__init__(chroma_path)
        parsed_url = urlparse(url)
        self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        self.url = url
        self.dataset = []
        self.has_limit = limit and limit > 0
        self.limit = limit
        self.count = 0

    def load_data(self):
        # Crawl the data
        print("Starting crawling")
        self._crawl_website(self.url)
        df = pd.DataFrame(self.dataset)
        df.to_csv('dataset.csv', index=False)
        print("Loaded dataset")

        print(f"Loading web crawled data into Chroma DB at {self.chroma_path}")
        documents = self._create_documents_from_dataset()
        chunks = self._split_documents(documents)

        self._add_to_chroma(chunks, self.chroma_path)

        # Add your logic to load web crawled data here
        # Example: db.add_documents(web_documents)
        print("Web crawled data loaded successfully")

    def _create_documents_from_dataset(self):
        documents = []
        for entry in self.dataset:
            documents.append(Document(
                page_content=entry['content'],
                metadata={'title': entry['title'], 'url': entry['url']}
            ))
        return documents

    def _split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    def _add_to_chroma(self, chunks, chroma_path):
        db = Chroma(
            persist_directory=chroma_path, embedding_function=self.embedding_function
        )
        chunks_with_ids = self._calculate_chunk_ids(chunks)
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        print("existing_ids ", existing_ids)
        print(f"Number of existing documents in DB: {len(existing_ids)}")
        seen_ids = defaultdict(int)
        new_chunks = []
        for chunk in chunks_with_ids:
            chunk_id = chunk.metadata["id"]
            if chunk_id not in existing_ids:
                if seen_ids[chunk_id] > 0:
                    chunk.metadata["id"] = f'{chunk.metadata["id"]}:{seen_ids[chunk_id]}'
                    print(chunk.metadata["id"])
                seen_ids[chunk_id] += 1
                new_chunks.append(chunk)

        if new_chunks:
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            db.persist()
        else:
            print("✅ No new documents to add")

    def _calculate_chunk_ids(self, chunks):
        # last_page_id = None
        # current_chunk_index = 0
        for chunk in chunks:
            print("chunk.metadata ", chunk.metadata)
            source = chunk.metadata.get("url")
            # page = chunk.metadata.get("page")
            # current_page_id = f"{source}"
            # if current_page_id == last_page_id:
            #     current_chunk_index += 1
            # else:
            #     current_chunk_index = 0
            chunk_id = f"{source}"
            # last_page_id = current_page_id
            chunk.metadata["id"] = chunk_id
        return chunks

    def _parse_page(self, url, retries=3, delay=5):
        print(f"Crawling {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko)'
                          ' Chrome/58.0.3029.110 Safari/537.3'
        }

        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers, timeout=30)
                # response.raise_for_status()  # Raise an error for bad status codes
                if response.status_code != 200:
                    continue
                soup = BeautifulSoup(response.content, 'html.parser')

                title_tag = soup.find('h1')
                title = title_tag.text.strip() if title_tag else 'No title'

                main_content = soup.find('div', {'id': 'body'})
                if main_content:
                    text_content = self._get_text_from_html(str(main_content))
                    self.dataset.append({
                        'url': url,
                        'title': title,
                        'content': text_content
                    })
                    print(f"Added entry: title={title}")
                    self.count += 1
                else:
                    print("No main content found.")
                break
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    print("All retry attempts failed.")
                    raise

    def _crawl_website(self, start_url):
        visited_urls = set()
        urls_to_visit = [start_url]

        while urls_to_visit:
            if self.has_limit and self.count > self.limit:
                break
            current_url = urls_to_visit.pop(0)
            if current_url in visited_urls:
                continue
            visited_urls.add(current_url)

            if '.pdf' in current_url:
                continue

            self._parse_page(current_url)

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
            try:
                response = requests.get(current_url, headers=headers, timeout=30)
                if response.status_code == 200:
                    # response.raise_for_status()
                    soup = BeautifulSoup(response.content, 'html.parser')
                else:
                    continue
            except requests.exceptions.RequestException as e:
                print(f"Failed to fetch links from {current_url}: {e}")
                continue

            # Find all links to crawl
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/'):
                    href = self.base_url + href
                if self.base_url in href and href not in visited_urls:
                    urls_to_visit.append(href)
                    # print(f"Found new link to visit: {href}")

    def _get_text_from_html(self, html_content):
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        text = h.handle(html_content)
        return text
