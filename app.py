import torch
import torch._classes
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pptx import Presentation
import os
import re
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
from fpdf import FPDF
import base64
from datetime import datetime
from langchain_core.documents import Document
import cohere
from transformers import BertTokenizer, BertForSequenceClassification
import shutil
import speech_recognition as sr
import whisper
from pydub import AudioSegment
import cv2
import ffmpeg
import time


# Add these to your initial session state initialization at the top of your file
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
if 'uploaded_audio' not in st.session_state:
    st.session_state.uploaded_audio = None
if 'uploaded_video' not in st.session_state:
    st.session_state.uploaded_video = None

# Load environment variables
load_dotenv()

cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

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

def clear_old_index():
    index_folder = "vector_store_index"
    if os.path.exists(index_folder):
        shutil.rmtree(index_folder)  # Delete old FAISS index
        print("✅ Old FAISS index deleted!")

def handle_file_upload(uploaded_file):
    """Saves an uploaded file to the UPLOAD_FOLDER and returns the file path."""
    if not uploaded_file:
        return None
        
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    print(f"📂 [DEBUG] File saved at: {file_path}")
    return file_path



def load_excel_and_convert_to_csv(file):
    try:
        df = pd.read_excel(file, engine="openpyxl")
        print(f"📊 [DEBUG] Loaded Excel file with {df.shape[0]} rows and {df.shape[1]} columns.")  # Debug statement
        return df.to_csv(index=False)
    except Exception as e:
        print(f"❌ [DEBUG] Error reading Excel file: {e}")  # Debug statement
        return f"Error: {str(e)}"

def get_ppt_content(file):
    slides_content = []
    prs = Presentation(file)
    for slide in prs.slides:
        slide_text = "".join([shape.text for shape in slide.shapes if hasattr(shape, "text")])
        slides_content.append(slide_text)
    print(f"📊 [DEBUG] Extracted {len(slides_content)} slides from PPT.")  # Debug statement
    return "\n".join(slides_content)

def get_pdf_text(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    print(f"📄 [DEBUG] Extracted {len(text.split())} words from PDF.")  # Debug statement
    return text

def extract_audio(video_path):
    """Extracts audio from a video file and saves it as a WAV file."""
    if not os.path.exists(video_path):
        print(f"❌ [DEBUG] Video file does not exist: {video_path}")
        return None

    audio_output = os.path.splitext(video_path)[0] + ".wav"  # Converts any extension to .wav
    
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_output, format='wav', acodec='pcm_s16le', ar='16000')  # Ensures compatibility
            .run(overwrite_output=True)
        )
        if os.path.exists(audio_output):
            print(f"✅ [DEBUG] Audio extracted and saved to: {audio_output}")
            return audio_output
        else:
            print(f"❌ [DEBUG] Audio extraction failed.")
            return None
    except Exception as e:
        print(f"❌ [DEBUG] Error extracting audio: {e}")
        return None


WHISPER_MODEL = whisper.load_model("base")

def transcribe_audio(audio_file_path):
    """Transcribes speech from an audio file using OpenAI Whisper."""
    if not isinstance(audio_file_path, str) or not os.path.exists(audio_file_path):
        print(f"❌ [DEBUG] Invalid file path: {audio_file_path}")
        return None

    try:
        result = WHISPER_MODEL.transcribe(audio_file_path)
        transcription = result["text"]
        print(f"🎤 [DEBUG] Transcription: {transcription[:100]}...")  # Show first 100 characters
        return transcription
    except Exception as e:
        print(f"❌ [DEBUG] Whisper Transcription Error: {e}")
        return None

def process_video(uploaded_video):
    """Process uploaded video file and add transcription to RAG system."""
    if not uploaded_video:
        return False
        
    try:
        # Save the video file
        video_path = handle_file_upload(uploaded_video)
        if not video_path:
            return False
            
        # Extract audio from video
        audio_path = extract_audio(video_path)
        if not audio_path:
            print("❌ [DEBUG] Failed to extract audio from video")
            return False
            
        # Transcribe the audio
        transcription = transcribe_audio(audio_path)
        
        if transcription:
            # Store transcription in session state
            if 'content' not in st.session_state:
                st.session_state.content = ""
            st.session_state.content += transcription + "\n"
            
            # Clear old index and update vector store
            clear_old_index()
            text_chunks = get_text_chunks(st.session_state.content)
            get_vector_store(text_chunks, "vector_store_index")
            
            # Clean up temporary files
            try:
                os.remove(video_path)
                os.remove(audio_path)
                print(f"🧹 [DEBUG] Cleaned up temporary files")
            except Exception as e:
                print(f"⚠️ [DEBUG] Could not remove temporary files: {e}")
            
            return True
            
        return False
        
    except Exception as e:
        print(f"❌ [DEBUG] Error processing video: {e}")
        return False

def process_audio(uploaded_audio):
    """Process uploaded audio file with optimized vector store updates."""
    if not uploaded_audio:
        return False
        
    try:
        # Save the file
        audio_path = handle_file_upload(uploaded_audio)
        if not audio_path:
            return False
            
        # Transcribe the audio
        transcription = transcribe_audio(audio_path)
        
        if transcription:
            # Clear old index first
            clear_old_index()
            
            # Split transcription into smaller, optimized chunks
            text_chunks = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Smaller chunks for audio transcripts
                chunk_overlap=50,  # Minimal overlap
                separators=["\n\n", "\n", ".", "!", "?", ",", " "]  # Better handling of speech patterns
            ).split_text(transcription)
            
            # Create optimized vector store for transcript
            vector_store = FAISS.from_texts(
                texts=text_chunks,
                embedding=embeddings,
                normalize_L2=True  # Enable L2 normalization for better search
            )
            
            # Save the optimized index
            vector_store.save_local("vector_store_index")
            
            # Store transcription in session state
            if 'content' not in st.session_state:
                st.session_state.content = ""
            st.session_state.content = transcription  # Replace content instead of appending
            
            # Clean up the file after processing
            try:
                os.remove(audio_path)
                print(f"🧹 [DEBUG] Cleaned up temporary file: {audio_path}")
            except Exception as e:
                print(f"⚠️ [DEBUG] Could not remove temporary file: {e}")
            
            return True
            
        return False
        
    except Exception as e:
        print(f"❌ [DEBUG] Error processing audio: {e}")
        return False


def transcribe_audio_speech_recognition(audio_file):
    """Transcribes audio using Google Speech API (SpeechRecognition)."""
    if not os.path.exists(audio_file):
        print(f"❌ [DEBUG] Audio file does not exist: {audio_file}")
        return None

    recognizer = sr.Recognizer()
    
    # Convert to WAV only if needed
    if not audio_file.endswith(".wav"):
        temp_audio_file = "temp_audio.wav"
        AudioSegment.from_file(audio_file).export(temp_audio_file, format="wav")
    else:
        temp_audio_file = audio_file

    with sr.AudioFile(temp_audio_file) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        print(f"🎤 [DEBUG] Transcription: {text[:100]}...")  # Show first 100 characters
        return text
    except sr.UnknownValueError:
        print("❌ [DEBUG] Google Speech API could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"❌ [DEBUG] Google Speech API request error: {e}")
        return None


def extract_frames(video_file, frame_interval=1):
    """Extracts frames from a video at a given interval (default: every 1 second)."""
    if not os.path.exists(video_file):
        print(f"❌ [DEBUG] Video file does not exist: {video_file}")
        return None

    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get frame rate
    frame_count = 0

    output_dir = "frames"
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    while cap.isOpened():
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % int(frame_rate * frame_interval) == 0:  # Save frames at specified intervals
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"🖼️ [DEBUG] Saved frame: {frame_path}")
            frame_count += 1

    cap.release()
    print(f"✅ [DEBUG] Extracted {frame_count} frames.")
    return output_dir  # Return the directory where frames are saved
    print(f"✅ [DEBUG] Extracted {frame_count} frames.")


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

def clean_text(text):
    """Removes non-UTF-8 characters, emojis, and surrogate Unicode pairs safely."""
    if not text:
        return ""

    # Replace surrogates and invalid UTF-8 characters
    text = text.encode("utf-8", "replace").decode("utf-8")  # Replaces invalid chars with "?"
    
    # Remove emojis & special symbols (Surrogate Pairs & Non-ASCII)
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)  # Removes all emoji & special symbols

    return text.strip()

def get_vector_store(text_chunks, vector_store_path):
    """Cleans text before creating FAISS vector store to prevent encoding errors."""
    cleaned_chunks = [clean_text(chunk) for chunk in text_chunks]  # Sanitize text
    vector_store = FAISS.from_texts(cleaned_chunks, embedding=embeddings)  # Store in FAISS
    vector_store.save_local(vector_store_path)
    print(f"✅ [DEBUG] FAISS index created with {len(cleaned_chunks)} chunks.")  # Debug statement
    return vector_store

def load_vector_store(vector_store_path):
    if not os.path.exists(vector_store_path):
        print("⚠️ [DEBUG] No FAISS index found. Returning an empty vector store.")  # Debug Statement
        return None
    print(f"✅ [DEBUG] FAISS index loaded from: {vector_store_path}")  # Debug Statement
    return FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, respond with "Answer is not available in the context."
    Context:\n {context}\nQuestion:\n{question}\nAnswer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def get_late_chunked_text(retrieved_docs, chunk_size=1000, chunk_overlap=100):
    """Dynamically chunks retrieved documents while maintaining structured document format."""
    chunked_docs = []
    
    for doc in retrieved_docs:
        if isinstance(doc, Document):  # Ensure doc is a LangChain Document object
            text = clean_text(doc.page_content)
        elif isinstance(doc, dict) and "page_content" in doc:
            text = clean_text(doc["page_content"])
        else:
            text = clean_text(str(doc))  # Convert to string as fallback
        
        start = 0
        while start < len(text):
            chunked_docs.append(Document(page_content=text[start: start + chunk_size]))  # Use Document object
            start += chunk_size - chunk_overlap  # Maintain overlap

    return chunked_docs

def retrieve_documents(query):
    """Searches FAISS or other vector stores using the enhanced query but does not modify the final answer."""
    try:
        # Load FAISS vector store
        vector_store = load_vector_store("vector_store_index")

        # Retrieve relevant documents using improved query
        retrieved_docs = vector_store.similarity_search(query)

        # Debugging: Print retrieved document details in the console
        print("\n==============================")
        print(f"🔍 [DEBUG] Query: {query}")
        print(f"📑 [DEBUG] Retrieved {len(retrieved_docs)} relevant documents.")
        for i, doc in enumerate(retrieved_docs[:3]):  # Print first 3 documents for debug
            print(f"📄 [DEBUG] Document {i+1}: {doc.page_content[:200]}...")  # Show first 200 chars
        print("==============================\n")

        return retrieved_docs  # Return retrieved documents for processing

    except Exception as e:
        print(f"❌ [DEBUG] Error retrieving documents: {e}")
        return []



def process_question(question, retrieved_docs):
    """Processes the user's question using the retrieved documents."""
    if not retrieved_docs:
        print(f"⚠️ [DEBUG] No documents found for query: {question}")  # Debug statement
        return "No relevant documents found."

    # Convert list of retrieved documents to a single string
    content = "\n".join([doc.page_content for doc in retrieved_docs if hasattr(doc, "page_content")])

    # Ensure content is cleaned before embedding
    cleaned_content = clean_text(content)

    # Store cleaned content in FAISS
    vector_store = FAISS.from_texts([cleaned_content], embedding=embeddings)
    vector_store.save_local("vector_store_index")

    print(f"✅ [DEBUG] Created FAISS index from retrieved documents.")  # Debug statement

    # Retrieve relevant document again (to be extra safe)
    new_db = FAISS.load_local("vector_store_index", embeddings, allow_dangerous_deserialization=True)
    retrieved_docs = new_db.similarity_search(question)

    # Ensure retrieved docs are LangChain Document objects
    formatted_docs = [doc if isinstance(doc, Document) else Document(page_content=str(doc)) for doc in retrieved_docs]

    # Apply late chunking after retrieval
    chunked_docs = get_late_chunked_text(formatted_docs)

    # Pass cleaned chunked documents to the QA model
    chain = get_conversational_chain()
    response = chain({"input_documents": chunked_docs, "question": question}, return_only_outputs=True)

    print(f"💬 [DEBUG] Response generated: {response['output_text'][:200]}...")  # Debug Statement (First 200 chars)
    return response["output_text"]



import time
import streamlit as st

# def handle_submit():
#     if st.session_state.user_input and st.session_state.user_input.strip():
#         if not (st.session_state.content or st.session_state.uploaded_audio or st.session_state.uploaded_video):
#             st.error("Please upload the necessary files.")
#             return
        
#         question = st.session_state.user_input
#         st.session_state.processing = True

#         # Process video
#         if st.session_state.uploaded_video:
#             video_path = extract_audio_from_video(st.session_state.uploaded_video)
#             transcript = transcribe_audio(video_path)
#             video_analysis = analyze_video(st.session_state.uploaded_video)
#             st.session_state.content += transcript + "\n" + video_analysis

#         # Process audio
#         if st.session_state.uploaded_audio:
#             transcript = transcribe_audio(st.session_state.uploaded_audio)
#             music_analysis = analyze_music_features(st.session_state.uploaded_audio)
#             st.session_state.content += transcript + "\n" + music_analysis
        
#         # Use extracted content for query processing
#         response = process_input(question)
#         st.session_state.conversation_history.append((question, response))

def handle_submit():
    if st.session_state.user_input and st.session_state.user_input.strip():
        if not (st.session_state.content or st.session_state.uploaded_image):
            st.error("Please upload the documents")
            return
        
        st.session_state.current_question = st.session_state.user_input
        st.session_state.processing = True
        st.session_state.user_input = ""




def fetch_related_terms(query):
    try:
        response = cohere_client.generate(
            model="command",
            prompt=f"Provide a list of related search terms (separated by commas) for improving retrieval. Do NOT change the meaning or structure of the query: '{query}'",
            max_tokens=15  # Ensures only short keywords are generated
        )
        
        related_terms = response.generations[0].text.strip()

        # Keep only valid terms
        related_terms = ", ".join([term.strip() for term in related_terms.split(",") if term.strip()])

        print("\n==============================")
        print(f"🔍 [DEBUG] Original Query: {query}")
        print(f"🔎 [DEBUG] Related Terms for Retrieval: {related_terms}")
        print("==============================\n")

        return related_terms
    except Exception as e:
        print(f"❌ [DEBUG] Error fetching related terms from Cohere API: {e}")
        return ""



def process_input(question):
    if len(question.split()) < 15:  # Short query
        related_terms = fetch_related_terms(question)  # Get related terms for better retrieval

        # Use related terms only for document retrieval, NOT for the final model input
        combined_query = f"{question} {related_terms}" if related_terms else question

        # Fetch relevant documents using enhanced query
        retrieved_docs = retrieve_documents(combined_query)

        # Pass only the original user query for answering
        return process_question(question, retrieved_docs)
    else:  # Long query
        return process_question_with_bert(question, st.session_state.content)


bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BertForSequenceClassification.from_pretrained(bert_model_name)

def remove_invalid_unicode(text):
    """Removes non-UTF-8 characters and surrogate Unicode pairs safely."""
    return text.encode("utf-8", "ignore").decode("utf-8")  # Ignore invalid characters

def process_question_with_bert(question, content):
    """Processes long queries using Hierarchical BERT and logs the process in the console."""
    if not content:
        print("⚠️ [DEBUG] No relevant content available.")
        return "No relevant content available."

    print("\n🔍 [DEBUG] Hierarchical BERT Processing Started...")

    # Step 1: Chunk the content into smaller pieces
    chunk_size = 512  # BERT token limit
    text_chunks = get_text_chunks(content)  # Use existing text chunking function

    print(f"📄 [DEBUG] Total Chunks Created: {len(text_chunks)}")

    # Step 2: Process each chunk using BERT
    all_scores = []
    for i, chunk in enumerate(text_chunks):
        inputs = tokenizer(chunk, question, truncation=True, padding="max_length", max_length=chunk_size, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs.logits.squeeze().tolist()
        all_scores.append((chunk, scores))

        print(f"🔹 [DEBUG] Chunk {i+1}/{len(text_chunks)} Processed | Score: {scores}")

    # Step 3: Rank and select the top chunks
    sorted_chunks = sorted(all_scores, key=lambda x: x[1], reverse=True)  # Sort by highest BERT score
    top_chunks = [chunk for chunk, _ in sorted_chunks[:3]]  # Select top 3 relevant chunks

    print(f"🏆 [DEBUG] Top {len(top_chunks)} Chunks Selected for Final Answer")

    # Step 4: Generate final response using generative AI
    final_context = "\n".join(top_chunks)

    # In process_question_with_bert function:
    cleaned_context = remove_invalid_unicode(final_context)
    print("✅ [DEBUG] Final Context Passed to the Model:\n", cleaned_context[:500], "...")  # Print first 500 characters

    return process_question(question, [Document(page_content=final_context)])



def get_gemini_response1(question, image):
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    try:
        response = model.generate_content([question, image])
        return response.text
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Chat PDF class for generating PDFs
# Chat PDF class for generating PDFs
# ChatPDF class with aligned and justified formatting
# ChatPDF class with aligned and justified formatting
class ChatPDF(FPDF):
    def __init__(self, file_names):
        super().__init__()
        self.file_names = file_names
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_margins(20, 20, 20)

    def header(self):
        self.set_font('helvetica', 'B', 16)
        self.cell(0, 10, 'IntelliQuery Conversation', 0, 1, 'C')
        self.ln(5)

        if self.file_names:
            self.set_font('helvetica', 'I', 10)
            file_names_str = ", ".join(self.file_names)
            self.cell(0, 10, f'Files: {file_names_str}', 0, 1, 'L')

        self.set_font('helvetica', 'I', 10)
        self.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'R')
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)

        # Footer with hyperlink on the left
        self.set_x(10)
        self.set_text_color(0, 102, 204)  # Hyperlink color
        self.cell(0, 10, 'IntelliQuery built by Pranav Vuddagiri', link="https://example.com", align='L')

        # Page number on the right
        self.set_x(-30)
        self.set_text_color(0)  # Reset to black
        self.cell(0, 10, f'Page {self.page_no()}', align='R')

    def add_message(self, role, content):
        # Set initial Y position
        initial_y = self.get_y()
        
        # Add separator line before message
        self.line(self.l_margin, initial_y - 2, self.w - self.r_margin, initial_y - 2)
        self.ln(5)  # Space after line
        
        # Reset Y position after line
        message_start_y = self.get_y()
        
        # Role label settings
        self.set_font('helvetica', 'B', 12)
        role_label_width = 30
        
        # Calculate content width
        content_x_start = self.l_margin + role_label_width
        content_width = self.w - self.r_margin - content_x_start
        
        # Add role label
        self.set_xy(self.l_margin, message_start_y)
        self.cell(role_label_width, 10, f"{role}:", 0, 0, 'L')
        
        # Add content
        self.set_font('helvetica', '', 11)
        for line in content.splitlines():
            self.set_xy(content_x_start, message_start_y)
            self.multi_cell(content_width, 10, line, 0, 'J')
            message_start_y = self.get_y()
        
        # Add spacing after message
        self.ln(5)

# Function to create and download PDF
def create_download_pdf(file_names):
    try:
        pdf = ChatPDF(file_names)
        pdf.alias_nb_pages()

        if "conversation_history" in st.session_state and st.session_state.conversation_history:
            for question, answer in st.session_state.conversation_history:
                pdf.add_message("User", question)
                pdf.add_message("Assistant", answer)
        else:
            pdf.add_message("System", "No conversations available.")

        return pdf.output(dest='S').encode('latin1')
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None




# Page config
st.set_page_config(layout="wide", page_title="IntelliQuery")

# Sidebar implementation
with st.sidebar:
    st.image("logo.svg", width=250)

    st.markdown("""
        <style>
        .stDownloadButton {
            background-color: transparent !important;
            color: #fff !important;
            border: 1px solid #262730 !important;
            padding: 10px 10px !important;
            border-radius: 4px !important;
            width: 20% !important;
            margin: 5px auto !important;
            display: block !important;
            text-align: center !important;
            font-weight: 500 !important;
        }
        .stDownloadButton:hover {
            background-color: #000 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Download button with container div
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    pdf_data = create_download_pdf(file_names=[])
    st.download_button(
        label="📥 Download Conversation",
        data=pdf_data if pdf_data else b"",
        file_name=f"IntelliQuery Conversation {datetime.now()}.pdf",
        mime="application/pdf"
    )
    st.markdown('</div>', unsafe_allow_html=True)



    st.title("Upload Your Documents")
    file_type = st.selectbox("Select file type", ["PDF", "PPT", "Excel","Image","Audio", "Video"])
    uploaded_files = None 
    uploaded_audio = None
    uploaded_video = None
    if file_type == "PDF":
        uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    
    elif file_type == "Audio":
        uploaded_audio = st.file_uploader("Choose an audio file", type=["mp3", "wav"])
    
        if uploaded_audio is not None:
            with st.spinner("Processing audio file..."):
                if process_audio(uploaded_audio):
                    st.success("✅ Audio processed successfully! You can now ask questions about its content.")
                else:
                    st.error("❌ Error processing the audio file.")

    
    elif file_type == "Video":
        uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi"])
    
        if uploaded_video is not None:
            if process_video(uploaded_video):
                st.success("✅ Video processed successfully! You can now ask questions about its content.")
            else:
                st.error("❌ Error processing the video file.")

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

    if file_type != "Image":
        if uploaded_files:
            combined_content = ""

            # Clear FAISS index before processing new files
            clear_old_index()

            # Process each uploaded file
            for file in uploaded_files:
                if file_type == "PDF":
                    combined_content += get_pdf_text(file) + "\n"
                elif file_type == "PPT":
                    combined_content += get_ppt_content(file) + "\n"
                elif file_type == "Excel":
                    combined_content += load_excel_and_convert_to_csv(file) + "\n"

            # Store the NEW content in session state
            st.session_state['content'] = combined_content

            # Split text into chunks and re-create FAISS index
            text_chunks = get_text_chunks(combined_content)
            get_vector_store(text_chunks, "vector_store_index")

            st.success(f"✅ {len(uploaded_files)} files processed successfully!")




# Main layout
# st.markdown("<h1>IntelliQuery: Empowering Precision with RAG</h1>", unsafe_allow_html=True)

header_container = st.container()
with header_container:
    st.markdown("<h1>IntelliQuery: Empowering Precision with RAG</h1>", unsafe_allow_html=True)

# Collect file names
if "uploaded_files" in st.session_state:
    file_names = [file.name for file in st.session_state.uploaded_files]
else:
    file_names = []

if "uploaded_audio" in locals() and uploaded_audio is not None:
    transcription = transcribe_audio(uploaded_audio)

if "uploaded_video" in locals() and uploaded_video is not None:
    process_video(uploaded_video)  # ✅ Safe to use uploaded_video



# Custom CSS for download button
st.markdown("""
<style>
    .stDownloadButton {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 999;
        background-color: #0E86D4;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stDownloadButton:hover {
        background-color: #0A6AAE;
        color: white;
    }
</style>
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