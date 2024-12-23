import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pptx import Presentation
import os
from pptx.enum.shapes import MSO_SHAPE_TYPE
import tempfile
import google.generativeai as genai
from langchain_experimental.agents import create_csv_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import pandas as pd
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
from googletrans import Translator, LANGUAGES


# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'content' not in st.session_state:
    st.session_state.content = ""
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None


# Load environment variables
load_dotenv()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "content" not in st.session_state:
    st.session_state.content = ""

# Configure Generative AI embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

# File handling functions
def handle_file_upload(file):
    file_path = os.path.join(UPLOAD_FOLDER, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def load_excel_and_convert_to_csv(file):
    try:
        df = pd.read_excel(file, engine="openpyxl")
        return df.to_csv(index=False)
    except Exception as e:
        return f"Error: {str(e)}"

def get_ppt_content(file):
    slides_content = []
    prs = Presentation(file)
    for slide in prs.slides:
        slide_text = "".join([shape.text for shape in slide.shapes if hasattr(shape, "text")])
        slides_content.append(slide_text)
    return "\n".join(slides_content)

def get_pdf_text(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, vector_store_path):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(vector_store_path)
    return vector_store

def load_vector_store(vector_store_path):
    return FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, respond with "Answer is not available in the context."
    Context:\n {context}\nQuestion:\n{question}\nAnswer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def process_question(question, content):
    chunks = get_text_chunks(content)
    vector_store = get_vector_store(chunks, "vector_store_index")
    new_db = load_vector_store("vector_store_index")
    docs = new_db.similarity_search(question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return response["output_text"]

def handle_submit():
    if st.session_state.user_input and st.session_state.user_input.strip():
        if not (st.session_state.content or st.session_state.uploaded_image):
            st.error("Please upload the documents")
            return
        
        st.session_state.current_question = st.session_state.user_input
        st.session_state.processing = True
        st.session_state.user_input = ""

def process_input(question):
    if st.session_state.uploaded_image is not None:
        # Process image-based question
        return get_gemini_response1(question, st.session_state.uploaded_image)
    else:
        # Process document-based question
        return process_question(question, st.session_state.content)

def get_gemini_response1(question, image):
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    try:
        response = model.generate_content([question, image])
        return response.text
    except Exception as e:
        return f"Error processing image: {str(e)}"


# Page config
st.set_page_config(layout="wide", page_title="IntelliQuery")

# Sidebar implementation
with st.sidebar:
    st.image("logo.svg", width=300)
    st.title("Upload Your Documents")
    file_type = st.selectbox("Select file type", ["PDF", "PPT", "Excel","Image"])
    
    if file_type == "PDF":
        uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    elif file_type == "PPT":
        uploaded_files = st.file_uploader("Choose PPT files", type=["pptx"], accept_multiple_files=True)
    elif file_type == "Excel":
        uploaded_files = st.file_uploader("Choose Excel files", type=["xlsx"], accept_multiple_files=True)
    elif file_type == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.success("Image uploaded successfully! You can now ask questions about the image using the chat input below.")

    if file_type!="Image":
        if uploaded_files:
            combined_content = ""
            for file in uploaded_files:
                if file_type == "PDF":
                    combined_content += get_pdf_text(file)
                elif file_type == "PPT":
                    combined_content += get_ppt_content(file)
                elif file_type == "Excel":
                    combined_content += load_excel_and_convert_to_csv(file)
            st.session_state['content'] = combined_content
            st.success("Files processed successfully!")


# Main layout
st.markdown("<h1>IntelliQuery: Empowering Precision with RAG</h1>", unsafe_allow_html=True)

st.markdown("""
<hr style="border: none; border-top: 1px solid #ccc; margin-top: -10px;">
""", unsafe_allow_html=True)

# Chat container
chat_placeholder = st.container()

with chat_placeholder:
    for q, a in st.session_state.conversation_history:
        st.markdown(f"""
            <div class='message-container'>
                <div class='user-message-container'>
                    <div class='user-message'>
                        <div class='message-icon'>👤</div>
                        <div class='message-content'>{q}</div>
                    </div>
                </div>
                <div class='bot-message-container'>
                    <div class='bot-message'>
                        <div class='message-icon'>🤖</div>
                        <div class='message-content'>{a}</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.processing and st.session_state.current_question:
        st.markdown(f"""
            <div class='message-container'>
                <div class='user-message-container'>
                    <div class='user-message'>
                        <div class='message-icon'>👤</div>
                        <div class='message-content'>{st.session_state.current_question}</div>
                    </div>
                </div>
                <div class='bot-message-container'>
                    <div class='bot-message loading'>
                        <div class='dots-container'>
                            <div class='dot'></div>
                            <div class='dot'></div>
                            <div class='dot'></div>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Process the question based on content type
        response = process_input(st.session_state.current_question)
        st.session_state.conversation_history.append((st.session_state.current_question, response))
        st.session_state.processing = False
        st.session_state.current_question = None
        st.rerun()

# Fixed position input area
input_container = st.container()
with input_container:
    col1, col2 = st.columns([6, 1])
    with col1:
        st.text_input(label="",placeholder="Type your question here...", key="user_input", on_change=handle_submit, label_visibility="collapsed")
    with col2:
        st.button("Send", on_click=handle_submit)

# Updated CSS
st.markdown("""
<style>
/* Main container spacing */
.main .block-container {
    padding-bottom: 100px !important;
}

/* Message container styles */
.message-container {
    margin: 1rem 0;
    padding: 0 1rem;
}

/* Loading animation */
.bot-message.loading {
    background-color: #1A1A1A;
    padding: 1rem;
}

.dots-container {
    display: flex;
    gap: 4px;
}

.dot {
    width: 8px;
    height: 8px;
    background-color: #0E86D4;
    border-radius: 50%;
    animation: bounce 1.4s infinite ease-in-out;
}

.dot:nth-child(1) { animation-delay: -0.32s; }
.dot:nth-child(2) { animation-delay: -0.16s; }

@keyframes bounce {
    0%, 80%, 100% { 
        transform: translateY(0);
    } 
    40% { 
        transform: translateY(-8px);
    }
}

/* Fixed input styling */
.stTextInput input {
    position: fixed !important;
    bottom: 20px !important;
    left: calc(370px + 0.5rem) !important;
    width: calc(100% - (370px + 0.5rem + 8rem)) !important;
    background: #000 !important;
    padding: 15px 20px !important;
    border-radius: 8px !important;
    border: 1px solid #4A4A4A !important;
    color: white !important;
    font-size: 0.9rem !important;
    height: 40px !important;
    z-index: 999 !important;
}

/* Send button styling */
.stButton button {
    position: fixed !important;
    bottom: 20px !important;
    right: 2rem !important;
    background: transparent !important;
    color: #0E86D4 !important;
    width: 80px !important;
    height: 40px !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    z-index: 999 !important;
}

/* Hide default elements */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
footer {
    display: none !important;
}

/* Message styling */
.user-message-container {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 0.5rem;
}

.bot-message-container {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 0.5rem;
}

.user-message, .bot-message {
    padding: 0.8rem;
    border-radius: 15px;
    max-width: 80%;
    display: flex;
    align-items: flex-start;
}

.user-message {
    background-color: #2C3333;
    border-radius: 15px 15px 0 15px;
    margin-left: auto;
}

.bot-message {
    background-color: #1A1A1A;
    border-radius: 15px 15px 15px 0;
}

.message-icon {
    margin-right: 0.5rem;
    font-size: 1.2rem;
}

.message-content {
    color: #FFFFFF;
    line-height: 1.5;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    .stTextInput input {
        left: 1rem !important;
        width: calc(100% - 7rem) !important;
        bottom: 10px !important;
    }
    
    .stButton button {
        right: 0.5rem !important;
        bottom: 10px !important;
    }
}
</style>
""", unsafe_allow_html=True)