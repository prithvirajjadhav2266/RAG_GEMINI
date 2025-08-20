import pickle
import numpy as np
import faiss
from flask import Flask, request, render_template_string, jsonify
from sentence_transformers import SentenceTransformer
import requests

# Load FAISS index and metadata
FAISS_PATH = 'faiss_index.bin'
META_PATH = 'faiss_meta.pkl'
MODEL_NAME = 'all-MiniLM-L6-v2'


print("[DEBUG] Loading FAISS index and metadata...")
index = faiss.read_index(FAISS_PATH)
with open(META_PATH, 'rb') as f:
    meta = pickle.load(f)
print(f"[DEBUG] Loaded {len(meta)} chunks from metadata.")
for i, chunk in enumerate(meta):
    print(f"[DEBUG] Chunk {i+1}: Heading: {chunk['heading']}, Content: {chunk['content'][:100]}{'...' if len(chunk['content']) > 100 else ''}")
print("[DEBUG] Initializing embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("[DEBUG] Embedding model loaded.")

# Gemini API setup (replace with your actual Gemini API endpoint and key)
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_API_KEY = ""

def get_gemini_answer(context, question):
    headers = {"Content-Type": "application/json"}
    # Add a system prompt to guide the model's behavior
    system_prompt = (
        "You are an expert assistant for a Retrieval-Augmented Generation (RAG) application. "
        "Given the provided context chunk from a document and a user question, answer as accurately and concisely as possible. "
        "If the answer is not present in the context, say 'The answer is not available in the provided context.' "
        "Always cite the heading in your answer if relevant."
    )
    print("[DEBUG] System prompt for Gemini:")
    print(system_prompt)
    print("[DEBUG] Prompt sent to Gemini:")
    print(f"Context:\n{context}\n\nQuestion:\n{question}")
    data = {
        "contents": [
            {"role": "user", "parts": [{"text": system_prompt}]},
            {"role": "user", "parts": [{"text": f"Context:\n{context}\n\nQuestion:\n{question}"}]}
        ]
    }
    response = requests.post(
        f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
        headers=headers,
        json=data
    )
    if response.status_code == 200:
        result = response.json()
        # Adjust this parsing based on Gemini's actual response format
        return result['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Error from Gemini API: {response.text}"

def search_chunks(query, top_k=5):
    print(f"[DEBUG] Embedding user query: {query}")
    query_emb = model.encode([query])
    print(f"[DEBUG] Query embedding: {query_emb}")
    # Normalize for cosine similarity
    faiss.normalize_L2(query_emb)
    faiss.normalize_L2(index.reconstruct_n(0, index.ntotal))
    D, I = index.search(query_emb.astype(np.float32), top_k)
    print(f"[DEBUG] FAISS search distances: {D}")
    print(f"[DEBUG] FAISS search indices: {I}")
    results = []
    for idx in I[0]:
        if idx < len(meta):
            print(f"[DEBUG] Retrieved chunk index: {idx}, Heading: {meta[idx]['heading']}, Content: {meta[idx]['content'][:100]}{'...' if len(meta[idx]['content']) > 100 else ''}")
            results.append(meta[idx])
    return results

app = Flask(__name__)

HTML = """
<!doctype html>
<html lang='en'>
<head>
    <meta charset='utf-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1'>
    <title>RAG Gemini Demo</title>
    <link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css' rel='stylesheet'>
    <style>
        body { background: #f8fafc; }
        .container { max-width: 800px; margin-top: 40px; }
        .card { margin-bottom: 24px; }
        .context-block { background: #f1f3f6; border-radius: 8px; padding: 16px; margin-top: 12px; font-size: 1rem; }
        .answer-block { background: #e7fbe7; border-radius: 8px; padding: 16px; margin-top: 12px; font-size: 1.1rem; }
        .heading { color: #2c3e50; }
        .footer { margin-top: 40px; color: #888; font-size: 0.95em; text-align: center; }
    </style>
</head>
<body>
    <div class='container'>
        <div class='card shadow-sm'>
            <div class='card-body'>
                <h2 class='mb-4 heading'>RAG Gemini Demo</h2>
                <form method='post' class='mb-3'>
                    <div class='input-group input-group-lg'>
                        <input name='query' class='form-control' placeholder='Ask a question...' autofocus required>
                        <button class='btn btn-primary' type='submit'>Ask</button>
                    </div>
                </form>
                {% if answer %}
                    <div class='answer-block'><strong>Answer:</strong><br>{{ answer }}</div>
                {% endif %}
                        {% if context_list %}
                            <div><strong>Context Chunks:</strong></div>
                            {% for c in context_list %}
                                <div class='context-block mb-3'><strong>{{ c.heading }}</strong><br>{{ c.content }}</div>
                            {% endfor %}
                        {% endif %}
            </div>
        </div>
        <div class='footer'>
            <span>Powered by Gemini API &middot; Retrieval-Augmented Generation Demo</span>
        </div>
    </div>
    <script src='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js'></script>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    answer = None
    context_list = []
    if request.method == 'POST':
        user_query = request.form['query']
        chunks = search_chunks(user_query, top_k=5)
        if chunks:
            context = "\n\n---\n\n".join([f"{c['heading']}\n\n{c['content']}" for c in chunks])
            context_list = [{"heading": c['heading'], "content": c['content']} for c in chunks]
            answer = get_gemini_answer(context, user_query)
        else:
            answer = "No relevant context found."
    return render_template_string(HTML, answer=answer, context_list=context_list)

@app.route('/api/ask', methods=['POST'])
def api_ask():
    data = request.json
    user_query = data.get('query', '')
    chunks = search_chunks(user_query, top_k=1)
    if chunks:
        context = f"{chunks[0]['heading']}\n\n{chunks[0]['content']}"
        answer = get_gemini_answer(context, user_query)
    else:
        answer = "No relevant context found."
        context = ""
    return jsonify({'answer': answer, 'context': context})

if __name__ == '__main__':
    app.run(debug=True)
