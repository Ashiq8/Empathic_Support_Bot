import os
import sys
from dotenv import load_dotenv
from textblob import TextBlob

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings


# ==================================================
# 1️⃣ LOAD ENV
# ==================================================
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("❌ GROQ_API_KEY not found in .env")
    sys.exit()

print("🤖 Empathic AI Support Bot Starting...")
print("📄 Loading policy document...")


# ==================================================
# 2️⃣ LOAD POLICY FILE
# ==================================================
loader = TextLoader("data/policy.txt", encoding="utf-8")
documents = loader.load()


# ==================================================
# 3️⃣ SPLIT TEXT
# ==================================================
text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
texts = text_splitter.split_documents(documents)


# ==================================================
# 4️⃣ EMBEDDINGS (LOCAL & FREE)
# ==================================================
print("🧠 Creating vector memory...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(texts, embeddings)


# ==================================================
# 5️⃣ GROQ LLM (UPDATED MODEL ✅)
# ==================================================
print("🔗 Connecting to Groq Brain...")

llm = ChatGroq(
    model="llama-3.1-8b-instant",   # ✅ ACTIVE & FREE MODEL
    temperature=0.3
)


# ==================================================
# 6️⃣ RAG CHAIN
# ==================================================
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3})
)

print("✅ Bot is READY!")
print("==================================================")


# ==================================================
# 7️⃣ CHAT LOOP + SENTIMENT
# ==================================================
while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Bot: Bye 👋")
        break

    # ❤️ Sentiment analysis
    polarity = TextBlob(user_input).sentiment.polarity
    if polarity < -0.3:
        mood = "ANGRY 😡"
        prefix = "I'm really sorry for the inconvenience. Let me help you right away. "
    else:
        mood = "NORMAL 🙂"
        prefix = ""

    try:
        response = qa_chain.invoke({"query": user_input})
        print(f"Bot (Mood: {mood}): {prefix}{response['result']}")
    except Exception as e:
        print(f"❌ Error: {e}")
