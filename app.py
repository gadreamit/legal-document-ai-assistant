import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate

# ---------------- UI ----------------
st.set_page_config(page_title="Legal AI Assistant")
st.title("📄 Legal Document AI Assistant (OpenAI)")

api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

uploaded_file = st.file_uploader("Upload Legal PDF", type="pdf")
question = st.text_input("Ask a question")

# ---------------- PDF Extraction ----------------
def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ---------------- Chunking ----------------
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

# ---------------- Vector Store ----------------
def create_vector_store(chunks, api_key):
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        model="text-embedding-3-small"
    )
    return FAISS.from_texts(chunks, embeddings)

# ---------------- LCEL Chain ----------------
def create_chain(vector_store, api_key):
    retriever = vector_store.as_retriever()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=api_key,
        temperature=0.3
    )

    prompt = PromptTemplate.from_template("""
    You are a legal assistant. Answer ONLY from the context.

    Context:
    {context}

    Question:
    {question}

    Answer clearly and accurately.
    """)

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return chain

# ---------------- Summarization ----------------
def summarize(text, api_key):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=api_key
    )

    prompt = f"""
    Summarize this legal document focusing on:
    - Key clauses
    - Obligations
    - Risks
    - Termination terms

    Document:
    {text[:5000]}
    """

    return llm.invoke(prompt)

# ---------------- Main Flow ----------------
if uploaded_file and api_key:

    text = extract_text(uploaded_file)

    if text:
        st.subheader("📄 Text Preview")
        st.write(text[:1000])

        if "vector_store" not in st.session_state:
            chunks = chunk_text(text)
            vector_store = create_vector_store(chunks, api_key)
            st.session_state.vector_store = vector_store
            st.success("Document processed!")

        vector_store = st.session_state.vector_store

        if question:
            chain = create_chain(vector_store, api_key)
            response = chain.invoke(question)
            st.subheader("💡 Answer")
            st.write(response.content)

        if st.button("Generate Summary"):
            summary = summarize(text, api_key)
            st.subheader("📝 Summary")
            st.write(summary.content)
