import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
import textwrap

# Cache the model and embeddings loading
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-T5-738M")
    model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-T5-738M")
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def load_embeddings():
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                          model_kwargs={"device": "cpu"})
    return instructor_embeddings


def load_retriever(embedding):
    db = Chroma(persist_directory='data_db', embedding_function=embedding)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever

# Load the models and embeddings
local_llm = load_models()
embedding = load_embeddings()
retriever = load_retriever(embedding)

qa_chain = RetrievalQA.from_chain_type(llm=local_llm,
                                       chain_type="stuff",
                                       retriever=retriever,
                                       return_source_documents=True)

# Helper functions
def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def process_llm_response(llm_response):
    wrapped_text = wrap_text_preserve_newlines(llm_response['result'])
    sources = "\n\nSources:\n" + "\n".join([source.metadata['source'] for source in llm_response["source_documents"]])
    return wrapped_text + sources

# Streamlit app
st.title("LLM Question Answering System")

query = st.text_input("Enter your query:")
if st.button("Submit"):
    with st.spinner("Processing..."):
        llm_response = qa_chain(query)
        result = process_llm_response(llm_response)
        st.text_area("Response", result, height=400)