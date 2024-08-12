import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.vectorstores import Chroma
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
tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-T5-738M")
model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-T5-738M")

from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import torch

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=256
)

local_llm = HuggingFacePipeline(pipeline=pipe)

print(local_llm('What is the capital of England?'))
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": "cpu"})


embedding = instructor_embeddings
db = Chroma(persist_directory='data_db',
                  embedding_function=embedding)

retriever = db.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(llm=local_llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)

import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

query = "when can we consider Enhanced Out-of-Care Support Agreement"
llm_response = qa_chain(query)
process_llm_response(llm_response)