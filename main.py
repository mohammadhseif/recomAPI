from flask import Flask, request, jsonify
from google.cloud import storage
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import re  # For standardization

app = Flask(__name__)

# GCP Configuration (consider environment variables for better security)
BUCKET_NAME = "t2m-labels"
EMBEDDINGS_FILE = "ideal_prompt_embeddings.npy"
FAISS_INDEX_FILE = "ideal_prompt_index.faiss"

# Load resources (do this once when the container starts)
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

blob = bucket.blob(EMBEDDINGS_FILE)
blob.download_to_filename(EMBEDDINGS_FILE)
ideal_prompt_embeddings = np.load(EMBEDDINGS_FILE)

blob = bucket.blob(FAISS_INDEX_FILE)
blob.download_to_filename(FAISS_INDEX_FILE)
index = faiss.read_index(FAISS_INDEX_FILE)

model = SentenceTransformer('all-mpnet-base-v2')

def standardize_prompt(prompt):
    """Basic prompt standardization."""
    prompt = prompt.lower()
    prompt = re.sub(r'[^\w\s]', '', prompt)  # Remove punctuation
    return prompt.strip()

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_prompt = data.get('prompt')

    if not user_prompt:
        return jsonify({"error": "Missing 'prompt' in request"}), 400

    # Standardize the user prompt
    standardized_user_prompt = standardize_prompt(user_prompt)

    # Embed the standardized user prompt
    user_embedding = model.encode([standardized_user_prompt])

    # Search the Faiss index for the 4 nearest neighbors
    distances, indices = index.search(user_embedding, k=4)

    # Retrieve the recommended prompts
    recommended_prompts = [ideal_prompts[i] for i in indices[0]]

    return jsonify({"recommendations": recommended_prompts})

if __name__ == "__main__":
    # In Cloud Run, the port is provided by the PORT environment variable
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
