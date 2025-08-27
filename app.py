import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from langchain_community.document_loaders import TextLoader
import tempfile
from langchain.docstore.document import Document
from PyPDF2 import PdfReader
import os
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import fitz
import io
import requests
from bs4 import BeautifulSoup

# Tesseract Path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\tesseract.exe"

CONFIG_FILE = "config.yaml"
GOOGLE_API_KEY = "AIzaSyCSaaJ5UpMN8yyioYJ0UK17-t_BS-E_-6Y"

st.title("Basic RAG")

# Auth helpers
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as file:
            return yaml.load(file, Loader=SafeLoader)
    return {
        'credentials': {'usernames': {}},
        'cookie': {'name': 'some_cookie_name', 'key': 'some_signature_key', 'expiry_days': 30},
        'preauthorized': {'emails': []}
    }

def save_config(config):
    with open(CONFIG_FILE, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

def register_user(username, name, password):
    config = load_config()
    if username in config['credentials']['usernames']:
        return False, "Username already exists"
    hashed_password = stauth.Hasher([password]).generate()[0]
    config['credentials']['usernames'][username] = {'name': name, 'password': hashed_password}
    save_config(config)
    return True, "User registered successfully"

# PDF OCR

def extract_text_from_pdf_images(uploaded_file):
    images = convert_from_bytes(uploaded_file.read())
    return "\n".join(pytesseract.image_to_string(image) for image in images)

def extract_images_from_pdf(uploaded_file):
    images_text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    doc = fitz.open(tmp_path)
    for i in range(len(doc)):
        for img_index, img in enumerate(doc.get_page_images(i)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image = Image.open(io.BytesIO(base_image["image"]))
            text = pytesseract.image_to_string(image)
            if text.strip():
                images_text += f"\n[Image OCR - Page {i+1} Image {img_index+1}]:\n{text}\n"
    doc.close()
    return images_text

# Web scraping
def scrape_website_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(["script", "style"]):
            tag.decompose()
        lines = [line.strip() for line in soup.get_text("\n").splitlines() if len(line.strip()) > 30]
        return "\n".join(lines)
    except Exception as e:
        return f"[Error scraping website: {e}]"
    


# Sidebar login/register
st.sidebar.markdown("### Welcome")
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Choose an option", ["Login", "Register"])

if menu == "Login":
    config = load_config()
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
    name, auth_status, username = authenticator.login('Login', location='main')

    if auth_status:
        st.success(f"Welcome {name}")
        if authenticator.logout('Logout', 'sidebar'):
            st.success("You have been logged out.")
            st.stop()

        input_type = st.selectbox("Choose source type", ["PDF", "Image", "Website URL", "CSV"])

        if "docs" not in st.session_state:
            st.session_state["docs"] = []
        if "docs_loaded" not in st.session_state:
            st.session_state["docs_loaded"] = False

        # File upload and processing
        if input_type == "PDF":
            uploaded_pdfs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
            if uploaded_pdfs:
                if st.session_state.get("last_file_type") != "pdf":
                    st.session_state["retriever"] = None
                    st.session_state["docs_loaded"] = False
                    st.session_state["last_file_type"] = "pdf"
            if st.button("Process PDF"):
                for file in uploaded_pdfs:
                    text = ""
                    reader = PdfReader(file)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    file.seek(0)
                    text += extract_text_from_pdf_images(file)
                    file.seek(0)
                    text += extract_images_from_pdf(file)
                    if text.strip():
                        st.session_state["docs"].append(Document(page_content=text, metadata={"source": file.name}))
                if st.session_state["docs"]:
                    st.session_state["docs_loaded"] = True
                    #st.success("PDFs processed and stored successfully.")
                else:
                    st.warning("No usable text found in the uploaded PDFs.")

       
             
    
        elif input_type == "Image":
            uploaded_images = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

            # Reset state if switching input type
            if uploaded_images and st.session_state.get("last_file_type") != "image":
                st.session_state.update({
                    "retriever": None,
                    "docs_loaded": False,
                    "last_file_type": "image",
                    "docs": []
                })

            if st.button("Process Images") and uploaded_images:
                st.session_state["docs"] = [
                    Document(page_content=pytesseract.image_to_string(Image.open(file)), metadata={"source": file.name})
                    for file in uploaded_images
                    if pytesseract.image_to_string(Image.open(file)).strip()
                ]

                if st.session_state["docs"]:
                    st.session_state["docs_loaded"] = True
                    st.success("Images processed successfully.")
                else:
                    st.warning("No usable text found in the uploaded images.")


        elif input_type == "Website URL":
            web_url = st.text_input("Enter website URL")

            if web_url and st.button("Process URL"):
                if st.session_state.get("last_file_type") != "web":
                    st.session_state.update({
                        "retriever": None,
                        "docs_loaded": False,
                        "last_file_type": "web",
                        "docs": []
                    })

                text = scrape_website_text(web_url)
                if text.strip():
                    st.session_state["docs"] = [Document(page_content=text, metadata={"source": web_url})]
                    st.session_state["docs_loaded"] = True
                    st.success("Website content processed successfully.")
                else:
                    st.warning("No usable text found at the provided URL.")

        elif input_type == "CSV":
            uploaded_csvs = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

            if uploaded_csvs and st.button("Process CSV"):
                if st.session_state.get("last_file_type") != "csv":
                    st.session_state.update({
                        "retriever": None,
                        "docs_loaded": False,
                        "last_file_type": "csv",
                        "docs": []
                    })

                import pandas as pd
                for file in uploaded_csvs:
                    try:
                        df = pd.read_csv(file)
                        text = df.to_csv(index=False)
                        if text.strip():
                            st.session_state["docs"].append(Document(page_content=text, metadata={"source": file.name}))
                    except Exception as e:
                        st.error(f"Error reading {file.name}: {e}")

                st.session_state["docs_loaded"] = True
                st.success("CSV files processed successfully.")


        if st.session_state.get("docs_loaded"):
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            documents = splitter.split_documents(st.session_state["docs"])

            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from langchain_community.vectorstores import FAISS

            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=GOOGLE_API_KEY
            )
            db = FAISS.from_documents(documents, embeddings)
            retriever = db.as_retriever(search_kwargs={"k": 3})

            from langchain_google_genai import GoogleGenerativeAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain.chains.combine_documents import create_stuff_documents_chain
            from langchain.chains import create_retrieval_chain

            llm = GoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.1)
            prompt = ChatPromptTemplate.from_template("""
          You are a highly intelligent AI assistant helping users find accurate and detailed answers from uploaded documents and extracted data. 
Documents can include text from PDFs, scanned images, CSV tables, or web pages.

 Instructions:
1. Always base your response on the provided context. Do **not** invent information not supported by the documents.
2. If the answer is **explicitly stated**, return it clearly.
3. If the answer is **not explicitly stated but can be logically inferred or deduced**, use **common sense reasoning** and clearly explain the reasoning.
4. If the content is **in a table or CSV**, extract and summarize the relevant rows and columns.
5. If the document is an **image or scanned page**, include any useful OCR text extracted.
6. If the document is a **website**, focus on the main textual content and ignore ads or irrelevant sections.
7. Always aim to **answer in detail**, not just in one sentence.
8. If the answer is **not present** in the provided content and cannot be reasonably inferred, clearly say:
   **"The answer is not available in the provided documents."**

            ---
            \U0001F4DA **Context**:  
            {context}  

            \U0001F4AC **Question**:  
            {input}  

            🧠 **Answer**:
            """)

            doc_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, doc_chain)

            query = st.text_input("Ask a question about the uploaded documents:")
            if query:
                with st.spinner("Searching..."):
                    result = retrieval_chain.invoke({"input": query})
                    st.subheader("Answer:")
                    st.write(result["answer"])

                    st.subheader("Retrieved Context:")
                    for i, doc in enumerate(result["context"], 1):
                        st.markdown(f"**Source {i}:** `{doc.metadata.get('source', 'Unknown')}`")
                        st.code(doc.page_content[:2000], language="markdown")

    elif auth_status is False:
        st.error("Invalid username or password")
    elif auth_status is None:
        st.info("Please enter your credentials")

elif menu == "Register":
    st.subheader("Create New account")
    new_username = st.text_input("Username")
    new_name = st.text_input("Full Name")
    new_password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm password", type="password")

    if st.button("Register"):
        if not new_username or not new_name or not new_password:
            st.warning("Please fill all fields")
        elif new_password != confirm_password:
            st.error("Passwords don't match")
        else:
            success, msg = register_user(new_username, new_name, new_password)
            if success:
                st.success(msg)
                st.info("Go to login page to login")
            else:
                st.error(msg)
