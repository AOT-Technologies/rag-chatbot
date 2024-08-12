import os
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader


from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings


from langchain.embeddings import HuggingFaceInstructEmbeddings


# Load and process the text files
# loader = TextLoader('single_text_file.txt')
loader = DirectoryLoader('./data', glob="./*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()



print(len(documents))


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": "cpu"})
# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'data_db'

## Here is the nmew embeddings being used
embedding = instructor_embeddings

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)

# persiste the db to disk
vectordb.persist()
vectordb = None

db = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)

retriever = db.as_retriever()

docs = retriever.get_relevant_documents("""who r maintains a record of all monitoring visits to the residential
resource""")
print(docs[0])