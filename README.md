RAG-based Document Query System

This project is a Retrieval-Augmented Generation (RAG) application that allows users to upload and query multiple document types (PDF, DOCX, TXT, CSV). It uses Google Generative AI (Gemini) as the LLM, FAISS for vector storage, and LangChain for orchestration.

Built with Streamlit for the user interface and Streamlit Authenticator for user login/registration.

🚀 Features

🔑 User Authentication (Login/Signup with secure credential storage in YAML)

📂 Multi-file Upload Support (PDF, DOCX, TXT, CSV)

🔍 Document Parsing with Tesseract OCR for scanned PDFs

🧠 RAG Pipeline:

Embedding generation with Google Generative AI

Chunking and storing vectors in FAISS

Retrieval and context injection into LLM prompt

💬 Interactive Q&A over uploaded documents

⚡ FastAPI Ready (can be extended for API usage)

🎛️ Simple Streamlit UI for querying

🛠️ Tech Stack

Frontend/UI: Streamlit

Authentication: Streamlit Authenticator

LLM & Embeddings: Google Generative AI (Gemini)

Vector Database: FAISS

Framework: LangChain

OCR: Tesseract

API Option: FastAPI

⚙️ Installation
1. Clone the repository
git clone https://github.com/your-username/rag-doc-query.git
cd rag-doc-query

2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # for Linux/Mac
venv\Scripts\activate     # for Windows

3. Install dependencies
pip install -r requirements.txt

4. Set up environment variables

Create a .env file with the following:

GOOGLE_API_KEY=your_google_api_key

5. Configure users

Update config.yaml with usernames & hashed passwords for login.

▶️ Usage

Run the Streamlit app:

streamlit run app.py


Upload one or more documents.

Log in with your credentials.

Ask questions in natural language.

Get context-aware answers from your documents.

📂 Project Structure
rag-doc-query/
│── app.py                # Streamlit app entrypoint
│── config.yaml           # User credentials (authenticator)
│── requirements.txt      # Dependencies
│── utils/                
│   ├── ocr.py            # Tesseract OCR utilities
│   ├── parser.py         # File parsers (PDF, CSV, DOCX, TXT)
│   ├── rag_pipeline.py   # FAISS + LangChain + LLM logic
│── vector_store/         # FAISS index storage
│── .env                  # Environment variables

📌 Roadmap

 Add support for image-based documents (PNG, JPG → OCR → text).

 Enhance UI with chat history.

 Deploy backend with FastAPI + frontend separately.

 Dockerize for production deployment.
