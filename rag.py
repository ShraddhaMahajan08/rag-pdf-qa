"""Core RAG pipeline: PDF -> chunks -> embeddings -> FAISS index -> retrieval -> answer."""
import os
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
MIN_SCORE = 0.10  # chunks less similar than this are treated as "not in the document"


def load_pdf(path):
    """Return a list of (page_number, text) for every non-empty page."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text, chunk_words=150, overlap=30):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_words]))
        start += chunk_words - overlap
    return chunks


class RAGIndex:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.index = None
        self.chunks = []  # list of dicts: {"text", "source", "page"}
        self._generator = None

    def add_pdf(self, path, source_name):
        new_chunks = []
        for page_no, text in load_pdf(path):
            for c in chunk_text(text):
                new_chunks.append({"text": c, "source": source_name, "page": page_no})
        if not new_chunks:
            return 0
        vectors = self.embedder.encode(
            [c["text"] for c in new_chunks], normalize_embeddings=True
        ).astype("float32")
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])  # cosine via normalized inner product
        self.index.add(vectors)
        self.chunks.extend(new_chunks)
        return len(new_chunks)

    def search(self, query, k=3):
        if self.index is None:
            return []
        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [
            {**self.chunks[i], "score": float(s)}
            for s, i in zip(scores[0], ids[0]) if i != -1
        ]

    def answer(self, question, k=3):
        if self.index is None:
            return "Please upload a PDF first.", []
        hits = [h for h in self.search(question, k) if h["score"] >= MIN_SCORE]
        if not hits:
            return "I couldn't find an answer to that in the uploaded document.", []
        context = "\n\n".join(h["text"] for h in hits)
        prompt = (
            "Answer the question using only the context below. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        return self._generate(prompt), hits

    def _generate(self, prompt):
        # Option 1: Claude API if a key is set. Option 2: free local model.
        if os.getenv("ANTHROPIC_API_KEY"):
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        if self._generator is None:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            name = "google/flan-t5-base"
            self._tokenizer = AutoTokenizer.from_pretrained(name)
            self._generator = AutoModelForSeq2SeqLM.from_pretrained(name)
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        output_ids = self._generator.generate(**inputs, max_new_tokens=200)
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
