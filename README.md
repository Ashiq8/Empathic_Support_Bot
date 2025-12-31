# 🤖 Empathic AI Assistant (RAG Pipeline)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Groq](https://img.shields.io/badge/Groq-Llama3-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)

## 📄 Overview
**Empathic AI Assistant** is a Retrieval-Augmented Generation (RAG) chatbot capable of answering questions from your custom documents (PDFs & Text files).

What makes this unique is its **Sentiment Awareness Layer**. The bot analyzes the user's emotion before responding. If the user seems frustrated or upset, the AI adjusts its tone to be more empathetic and helpful.

## ✨ Key Features
* **Chat with Data:** Upload any PDF/TXT policy document and ask questions.
* **Sentiment Analysis:** Uses NLP (`TextBlob`) to detect user mood.
* **Blazing Fast Inference:** Powered by **Groq LPU** (Llama-3-8b model).
* **Memory:** Remembers previous context in the conversation.

## 🛠️ Tech Stack
* **LLM:** Groq (Llama-3-8b-8192)
* **Orchestration:** LangChain
* **Vector DB:** FAISS (Facebook AI Similarity Search)
* **Frontend:** Streamlit
* **NLP:** TextBlob

## 📂 Project Structure
```text
Empathic-AI-Bot/
├── app.py               # Main Application Logic
├── .env                 # API Keys (GitIgnored)
├── .gitignore           # Security Rule
├── requirements.txt     # Dependencies
└── README.md            # Documentation
