import os
import magic
import requests
import io
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util

# Load env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Globals
DOCUMENT_CHUNKS = []
embedder = SentenceTransformer('all-MiniLM-L6-v2')

ALLOWED_MIME_TYPES = {"text/plain", "application/pdf"}

# Extract text

def extract_text(file_bytes, mime_type):
    if mime_type == "text/plain":
        return file_bytes.decode("utf-8", errors="ignore")
    elif mime_type == "application/pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
        return text.strip()
    return ""

# Chunking

def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Embedding search

def get_top_chunks(query, chunks, top_k=3):
    question_embedding = embedder.encode(query, convert_to_tensor=True)
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
    top_results = scores.argsort(descending=True)[:top_k]
    return [chunks[i] for i in top_results]

# Groq API call

def call_groq_api(context, query):
    prompt = f"""You are a helpful assistant. Use the context below to answer the question in markdown format. Prefer tables or bullet points if possible.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    return f"Error: {response.status_code} - {response.text}"

# HTML UI

@app.route("/", methods=["GET"])
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Doc Chat with Groq</title>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            body { font-family: Arial; padding: 40px; background: #f9f9f9; }
            h2 { color: #333; }
            form, .chatbox { background: #fff; padding: 20px; margin-bottom: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            button { background: #007bff; color: #fff; padding: 10px 15px; border: none; border-radius: 5px; cursor: pointer; }
            input[type="text"] { width: 80%; padding: 10px; }
            #response { white-space: pre-wrap; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h2>📄 Upload a PDF/TXT and Ask Questions</h2>

        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <button type="submit">Upload File</button>
        </form>

        <div class="chatbox">
            <h3>💬 Ask a Question</h3>
            <input type="text" id="queryInput" placeholder="Type your question here">
            <button onclick="askQuery()">Ask</button>
            <div id="response"></div>
        </div>

        <script>
            function askQuery() {
                const query = document.getElementById("queryInput").value;
                fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query })
                })
                .then(res => res.json())
                .then(data => {
                    document.getElementById("response").innerHTML = marked.parse(data.answer);
                })
                .catch(err => {
                    document.getElementById("response").innerText = "Error: " + err;
                });
            }
        </script>
    </body>
    </html>
    """)

# Upload file

@app.route("/upload", methods=["POST"])
def upload_file():
    global DOCUMENT_CHUNKS

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_bytes = file.read()
    mime_type = magic.from_buffer(file_bytes, mime=True)

    if mime_type not in ALLOWED_MIME_TYPES:
        return jsonify({"error": f"Unsupported file type: {mime_type}"}), 400

    text = extract_text(file_bytes, mime_type)
    if not text:
        return jsonify({"error": "Could not extract text from file"}), 400

    DOCUMENT_CHUNKS = split_text_into_chunks(text)
    return """
        <script>alert("✅ File uploaded and processed successfully! You can now ask questions."); window.location.href="/";</script>
    """

# Ask a question

@app.route("/ask", methods=["POST"])
def ask():
    global DOCUMENT_CHUNKS

    data = request.get_json()
    query = data.get("query", "")

    if not DOCUMENT_CHUNKS:
        return jsonify({"error": "No document uploaded yet."}), 400
    if not query.strip():
        return jsonify({"error": "Query cannot be empty."}), 400

    top_chunks = get_top_chunks(query, DOCUMENT_CHUNKS)
    context = "\n\n".join(top_chunks)

    answer = call_groq_api(context, query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
