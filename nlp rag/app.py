import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH_PDF = "chroma/pdf"
CHROMA_PATH_WEBSITE = "chroma/web"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_rag(query_text: str, source_type: str):
    # Prepare the DB based on source type.
    embedding_function = get_embedding_function()
    if source_type == "PDF documents":
        db = Chroma(persist_directory=CHROMA_PATH_PDF, embedding_function=embedding_function)
    else:
        db = Chroma(persist_directory=CHROMA_PATH_WEBSITE, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    return response_text, sources


col1, col2 = st.columns([1, 3])  # Adjust the column widths as needed
with col1:
    st.image('logo/bcgov.jpeg', use_column_width=True)

with col2:
    st.title("MCFD - Ask a Question")
    st.write("Ask a question regarding the policies to get a generative answer!\n"
             "Select 'PDF documents' to generate answer based on the provided PDF docs, "
             "or select 'Website' to generate answer based on web data.")

query_text = st.text_input("Enter your query:")
source_type = st.radio("Select source type:", ("PDF documents", "Website"))

if st.button("Submit Query"):
    if query_text:
        with st.spinner("Processing query..."):
            response_text, sources = query_rag(query_text, source_type)

        st.subheader("Response:")
        st.write(response_text)

        st.subheader("Sources:")
        st.write(sources)
    else:
        st.warning("Please enter a query.")
