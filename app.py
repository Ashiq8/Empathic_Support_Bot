import os
import streamlit as st
from dotenv import load_dotenv
from textblob import TextBlob
import tempfile

# LangChain Imports
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# 1️⃣ CONFIGURATION
st.set_page_config(page_title="AI Support Bot", page_icon="🤖")
load_dotenv()

# Setup API Key
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY not found in .env file!")
    st.stop()

# 2️⃣ FUNCTIONS (The Brain Logic)

def get_vectorstore(uploaded_file):
    """File-a padichu, Memory (Vector Store) create pannum function"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load based on file type
    if uploaded_file.name.endswith('.pdf'):
        loader = PyPDFLoader(tmp_file_path)
    else:
        loader = TextLoader(tmp_file_path, encoding='utf-8')
    
    documents = loader.load()
    
    # Split Text
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Embeddings (Local & Free)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore

def get_conversation_chain(vectorstore):
    """RAG Chain with Memory create pannum function"""
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# 3️⃣ SIDEBAR (File Upload)
with st.sidebar:
    st.header("📂 Data Source")
    uploaded_file = st.file_uploader("Upload your Policy (PDF/TXT)", type=["pdf", "txt"])
    
    if uploaded_file:
        if "vectorstore" not in st.session_state:
            with st.spinner("🧠 Reading file & Building memory..."):
                # Create Vector Store only once
                st.session_state.vectorstore = get_vectorstore(uploaded_file)
                # Create Chain
                st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
            st.success("File Processed! Ready to Chat.")

# 4️⃣ MAIN CHAT INTERFACE
st.title("🤖 Empathic AI Assistant")
st.caption("Powered by Groq & LangChain")

# Initialize Chat History if not exists
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History (Visual Memory)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User Input
user_input = st.chat_input("Ask about the policy...")

if user_input:
    # 1. Add User Message to UI
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Check if File is Uploaded
    if "conversation" not in st.session_state:
        st.error("⚠️ Please upload a document in the sidebar first!")
    else:
        # 2. Sentiment Analysis (The Heart)
        blob = TextBlob(user_input)
        if blob.sentiment.polarity < -0.3:
            mood_prefix = "😔 (I see you are upset, let me help.) "
        else:
            mood_prefix = ""

        # 3. Generate Answer (The Brain)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({'question': user_input})
                answer = mood_prefix + response['answer']
                st.write(answer)
        
        # 4. Add Assistant Message to UI History
        st.session_state.messages.append({"role": "assistant", "content": answer})