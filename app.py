import os
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
from rag import RAGIndex

app = Flask(__name__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
rag = RAGIndex()

PAGE = """
<!doctype html>
<title>PDF Q&A (RAG)</title>
<body style="font-family:sans-serif;max-width:700px;margin:40px auto">
<h2>PDF Question Answering (RAG)</h2>
<input type="file" id="f" accept=".pdf"> <button onclick="up()">Upload</button>
<p id="s"></p><hr>
<input id="q" style="width:75%" placeholder="Ask a question about your PDF">
<button onclick="ask()">Ask</button>
<h4>Answer</h4><p id="a"></p><h4>Sources</h4><div id="src"></div>
<script>
async function up(){
  const fd=new FormData(); fd.append('file',document.getElementById('f').files[0]);
  document.getElementById('s').innerText='Indexing...';
  const r=await fetch('/upload',{method:'POST',body:fd}); const d=await r.json();
  document.getElementById('s').innerText=d.message||d.error;
}
async function ask(){
  const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question:document.getElementById('q').value})});
  const d=await r.json();
  document.getElementById('a').innerText=d.answer||d.error;
  document.getElementById('src').innerHTML=(d.sources||[]).map(
    s=>'<p><b>'+s.source+' (page '+s.page+', score '+s.score.toFixed(2)+')</b><br>'+s.text+'</p>').join('');
}
</script></body>
"""

@app.route("/")
def home():
    return render_template_string(PAGE)

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".pdf"):
        return jsonify(error="Please upload a PDF file."), 400
    name = secure_filename(f.filename)
    path = os.path.join(UPLOAD_DIR, name)
    f.save(path)
    n = rag.add_pdf(path, name)
    if n == 0:
        return jsonify(error="No extractable text found (scanned PDF?)."), 400
    return jsonify(message=f"Indexed {n} chunks from {name}.")

@app.route("/ask", methods=["POST"])
def ask():
    question = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not question:
        return jsonify(error="Please enter a question."), 400
    answer, hits = rag.answer(question)
    return jsonify(answer=answer, sources=hits)

if __name__ == "__main__":
    app.run(debug=True)
