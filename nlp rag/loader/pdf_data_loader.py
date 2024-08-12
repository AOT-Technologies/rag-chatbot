from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loader import DataLoader
from get_embedding_function import get_embedding_function


class PdfDataLoader(DataLoader):
    def load_data(self):
        print(f"Loading PDF data into Chroma DB at {self.chroma_path}")
        documents = self._load_documents()
        chunks = self._split_documents(documents)
        self._add_to_chroma(chunks, self.chroma_path)
        print("PDF data loaded successfully")

    def _load_documents(self):
        document_loader = PyPDFDirectoryLoader('data')
        return document_loader.load()

    def _split_documents(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    def _add_to_chroma(self, chunks: list[Document], chroma_path):
        # Load the existing database.
        db = Chroma(
            persist_directory=chroma_path, embedding_function=get_embedding_function()
        )

        # Calculate Page IDs.
        chunks_with_ids = self._calculate_chunk_ids(chunks)

        # Add or Update the documents.
        existing_items = db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Only add documents that don't exist in the DB.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            db.persist()
        else:
            print("✅ No new documents to add")

    def _calculate_chunk_ids(self, chunks):

        # This will create IDs like "data/monopoly.pdf:6:2"
        # Page Source : Page Number : Chunk Index

        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index.
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Calculate the chunk ID.
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add it to the page meta-data.
            chunk.metadata["id"] = chunk_id

        return chunks
