import os
import logging
import chromadb
from flask import Flask, request, jsonify
from google.cloud import storage
import google.generativeai as genai
import fitz
import requests

# Configure Logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to logging.INFO in production
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ID = "ltim-delab-app"
REGION_ID = "us-east1"
BUCKET_NAME = "ai-app-gcs"
FILE_NAME = "sample_cast_report.pdf"
MODEL_NAME = "gemini-pro"
EMBEDDING_MODEL_NAME = "models/embedding-001"
CHROMA_DB_URL = "https://chromadb-891176152394.us-central1.run.app"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Initialize Clients
storage_client = storage.Client()
genai.configure(api_key=GOOGLE_API_KEY)

# Fetch IAM Token
try:
    IAM_TOKEN = requests.get(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience=https://chromadb-891176152394.us-central1.run.app",
        headers={"Metadata-Flavor": "Google"},
    ).text.strip()
    logger.info(f"IAM Token Fetched: {IAM_TOKEN[:20]}...")
except Exception as e:
    logger.error(f"Error fetching IAM token: {e}")
    IAM_TOKEN = None

# Test API call to ChromaDB heartbeat
headers = {"Authorization": f"Bearer {IAM_TOKEN}"}
try:
    response = requests.get(f"{CHROMA_DB_URL}/api/v1/heartbeat", headers=headers)
    logger.info(f"🔍 ChromaDB Heartbeat Status: {response.status_code}, Response: {response.text}")
except Exception as e:
    logger.error(f"Error checking ChromaDB heartbeat: {e}")

# Initialize ChromaDB Client
try:
    chroma_client = chromadb.HttpClient(CHROMA_DB_URL, headers=headers)
    logger.info("ChromaDB client initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing ChromaDB client: {e}")

# Create or get collection
try:
    collection = chroma_client.get_or_create_collection("cast_highlight_reports")
    logger.info(f"ChromaDB Collection Retrieved: {collection.name}")
except Exception as e:
    logger.error(f"Error creating/getting ChromaDB collection: {e}")

def extract_text_from_pdf(pdf_bytes):
    """Extract text content from the PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join(page.get_text("text") for page in doc)

def generate_gemini_response(text):
    """Generate AI response using Gemini-Pro."""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(f"Summarize this text:\n{text}")
        return response.text
    except Exception as e:
        logger.error(f"Error generating Gemini response: {e}")
        return None

def generate_embeddings(text):
    """Generate embeddings using Gemini AI."""
    try:
        response = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        logger.info("Embedding Generated Successfully.")
        return response["embedding"]
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return None

def store_embeddings_in_chroma(file_name, text):
    """Store embeddings in ChromaDB and print stored vectors."""
    logger.info(f"🔍 Storing Embeddings for: {file_name}")

    embedding_vector = generate_embeddings(text)
    if embedding_vector is None:
        logger.error("Skipping storage due to embedding failure.")
        return
    
    logger.info(f"Embedding vectors:{embedding_vector}")  

    try:
        logger.debug(f"Embedding Vector Length: {len(embedding_vector)}")

        # Store embedding in ChromaDB
        collection.add(
            ids=[file_name],
            embeddings=[embedding_vector],
            metadatas=[{"file_name": file_name, "content": text[:500]}]
        )
        logger.info(f"Successfully stored {file_name} in ChromaDB")

        # Print the stored embedding vector
        logger.info(f"\n📌 **Stored Embedding Vector for {file_name}:**")

    except Exception as e:
        logger.error(f"Error storing in ChromaDB: {e}")


# Flask App
app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process_file():
    """Process PDF file from GCS, generate summary & store embeddings."""
    logger.info("\n🚀 Processing File...")
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(FILE_NAME)
        pdf_bytes = blob.download_as_bytes()

        extracted_text = extract_text_from_pdf(pdf_bytes)
        logger.info(f"Extracted Text Length: {len(extracted_text)} characters")

        gemini_response = generate_gemini_response(extracted_text)
        store_embeddings_in_chroma(FILE_NAME, extracted_text)

        return jsonify({"message": "Processing complete", "file": FILE_NAME, "gemini_summary": gemini_response})
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

