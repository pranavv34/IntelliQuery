# IntelliQuery

IntelliQuery is a Streamlit application that provides document-based question answering using a **Retrieval-Augmented Generation (RAG)** approach. It supports multiple document types (PDF, PPT, Excel, audio/video transcripts, images) and answers user questions based on the embedded content. Additionally, it integrates **Hierarchical BERT** (HBERT) for **long queries** (more than 15 words) to improve retrieval on large or complex questions, while retaining a simpler vector store approach for short queries.

---

## Features

- **Document Upload & Processing**  
  - Upload **PDF**, **PPT**, **Excel**, **Image**, **Audio**, or **Video** files.
  - Automatic **text extraction** from files (PPT slides, PDF pages, Excel sheets, etc.).
  - **Speech-to-text** transcription for audio and video via Whisper.
  - Cleans and tokenizes text for downstream tasks.

- **Vector Store with FAISS**  
  - Uses **Google Generative AI Embeddings** (`models/embedding-001`) for short queries.
  - Stores embeddings in a FAISS index for fast similarity search.

- **Hierarchical BERT for Long Queries**  
  - Automatically switches to **Hierarchical BERT** retrieval if the query exceeds 15 words.
  - Splits documents into large “doc chunks,” which are further split into sub-chunks.  
  - Performs **two-stage retrieval**:
    1. First retrieve top doc chunks.
    2. Retrieve top sub-chunks for finer context.

- **Q&A Pipeline**  
  - Uses **Google Generative AI** (Gemini) for final answer generation.
  - Summaries or direct answers to user queries based on the retrieved text.
  - Optional **cohere** expansions for short queries, to enrich retrieval.

- **Responsive UI**  
  - Built with **Streamlit** for an interactive web app.
  - Displays conversation history and supports continuous queries.
  - Downloadable PDF of the conversation history.

---

## Tech Stack

1. **Python 3.9+**  
2. **Streamlit** – interactive UI for uploading files and chatting.  
3. **Whisper** – speech-to-text for audio/video.  
4. **FAISS** – vector database for fast similarity searches.  
5. **Google Generative AI Embeddings** & **Gemini** – text embeddings and LLM Q&A.  
6. **SentenceTransformers** (Hierarchical BERT) – hierarchical embedding for long queries.  
7. **Cohere** – fetch related terms for short queries.  
8. **PyPDF2**, **python-pptx**, **Pandas** – for file parsing.  

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/pranavv34/IntelliQuery.git
   ```

2. **Create and Activate a Virtual Environment** (recommended)
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # Linux/Mac
   # OR myenv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Make sure you have libraries like `faiss-cpu`, `sentence-transformers`, `whisper`, etc.

4. **Add API Keys**  
   - Create a `.env` file in the project root and add:
     ```
     COHERE_API_KEY=YOUR_COHERE_API_KEY
     GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
     ```
   Adjust as needed if you have more environment variables.

---

## Usage

1. **Run the App**
   ```bash
   streamlit run app.py
   ```
   This will launch the Streamlit interface in your default browser.

2. **Upload Documents**  
   - In the **sidebar**, select the file type and upload one or more files.
   - IntelliQuery automatically processes and indexes these documents (creating a FAISS index for short queries).

3. **Ask Questions**  
   - Type your query in the text box at the bottom.
   - If the query is **15 words or fewer**, IntelliQuery uses **standard FAISS** + **Google Generative AI Embeddings**.
   - If the query is **more than 15 words**, IntelliQuery **on-demand** builds/loads the **Hierarchical BERT** indexes and performs a **two-stage retrieval**.

4. **View and Download Conversations**  
   - The conversation displays in a chat-like format.
   - Click **“Download Conversation”** to get a PDF summary of the Q&A.

---

## Hierarchical BERT for Long Queries

- **On-demand Indexing**: If you ask a query longer than 15 words, the app checks if the **Hierarchical BERT** indexes are built. If not, it builds them from the same text corpus already stored.
- **Two-stage Retrieval**:  
  1. **Doc-level retrieval** with ~5k-character chunks.  
  2. **Sub-chunk retrieval** with ~500-character chunks to narrow context further.
- Passes the final **sub-chunks** to the Q&A chain for a detailed answer.

---

## Contact / Author

- **Author**: [Pranav Vuddagiri](https://github.com/pranavv34)  
- **Issues**: Please open an issue on this repo if you encounter any problems.

Enjoy using **IntelliQuery**! Let us know if you have any questions or feature suggestions.
