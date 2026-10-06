# PDF Question Answering with RAG

A Retrieval-Augmented Generation (RAG) app. Upload a PDF, ask questions, and get answers grounded in the document, with the source passages and page numbers shown.

## How it works
1. **Ingest:** extract text from the PDF (pypdf) and split it into overlapping chunks.
2. **Embed:** convert chunks to vectors with Sentence-Transformers (`all-MiniLM-L6-v2`).
3. **Index:** store vectors in a FAISS index (cosine similarity).
4. **Retrieve:** embed the question and fetch the top-3 most similar chunks.
5. **Generate:** pass the question plus retrieved context to an LLM (Claude via API if `ANTHROPIC_API_KEY` is set, otherwise a free local `flan-t5-base` model).
6. **Serve:** Flask REST API (`/upload`, `/ask`) with a simple web page.

## Run
```bash
pip install -r requirements.txt
# optional: use Claude for better answers
# export ANTHROPIC_API_KEY=your_key      (Windows: set ANTHROPIC_API_KEY=your_key)
python app.py
```
Open http://127.0.0.1:5000, upload a PDF, and ask a question.

## Tech
Python, Flask, Sentence-Transformers, FAISS, pypdf, Hugging Face Transformers, Anthropic API (optional).
