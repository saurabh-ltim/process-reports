import os
import chromadb
from flask import Flask, request, jsonify
from google.cloud import storage
import google.generativeai as genai
import fitz  

# Constants
PROJECT_ID = "ltim-delab-app"
REGION_ID = "us-east1"
BUCKET_NAME = "ai-app-gcs"
FILE_NAME = "sample_cast_report.pdf"
MODEL_NAME = "gemini-pro"
EMBEDDING_MODEL_NAME = "models/embedding-001"  
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Initialize Clients
storage_client = storage.Client()
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize ChromaDB in-memory mode
chroma_client = chromadb.Client()  

# Create or get collection
collection = chroma_client.get_or_create_collection("cast_highlight_reports")

def extract_text_from_pdf(pdf_bytes):
    """Extract text content from the PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join(page.get_text("text") for page in doc)

def generate_gemini_response(text):
    """Generate AI response using Gemini-Pro."""
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(f"Summarize this text:\n{text}")
    return response.text

def generate_embeddings(text):
    response = genai.embed_content(
        model=EMBEDDING_MODEL_NAME,  
        content=text,
        task_type="RETRIEVAL_DOCUMENT"
    )
    return response["embedding"]  

def store_embeddings_in_chroma(file_name, text):
    """Store embeddings in ChromaDB."""
    embedding_vector = generate_embeddings(text)
    collection.add(
        ids=[file_name],
        embeddings=[embedding_vector],
        metadatas=[{"file_name": file_name, "content": text[:500]}]  # Store first 500 chars as metadata
    )

app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process_file():
    """Process PDF file from GCS, generate summary & store embeddings."""
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(FILE_NAME)
    pdf_bytes = blob.download_as_bytes()

    extracted_text = extract_text_from_pdf(pdf_bytes)
    gemini_response = generate_gemini_response(extracted_text)
    store_embeddings_in_chroma(FILE_NAME, extracted_text)

    return jsonify({"message": "Processing complete", "file": FILE_NAME, "gemini_summary": gemini_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
