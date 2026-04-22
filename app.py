import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

st.title("📄 AI Document Assistant")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded successfully!")

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Create embeddings
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    # Ask question
    query = st.text_input("Ask a question about your document:")

    if query:
        retriever = vectorstore.as_retriever()
        relevant_docs = retriever.get_relevant_documents(query)

        context = "\n".join([doc.page_content for doc in relevant_docs])

        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )

        prompt = f"""
        Answer only from this context:
        {context}

        Question: {query}
        """

        answer = llm.predict(prompt)

        st.write("### Answer:")
        st.write(answer)
