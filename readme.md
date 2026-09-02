# LanceRAG — Document Semantic Search Pipeline

A small, self-contained Retrieval-Augmented Generation (RAG) preprocessing
pipeline. It ingests documents (`.txt`, `.pdf`, `.docx`), cleans and chunks
their text, embeds the chunks with a Sentence-Transformers model
(`BAAI/bge-large-en-v1.5`), stores the embeddings in a local **LanceDB**
vector database, and lets you run natural-language semantic search over
them from the terminal.

## How it works

```
Documents/*.pdf,*.docx,*.txt
        │
        ▼
document_loader.py   → extracts raw text (per page for PDFs)
        │
        ▼
text_cleaner.py       → normalizes whitespace
        │
        ▼
chunk_creator.py       → splits text into overlapping word chunks
        │
        ▼
vector_store.py        → embeds chunks (BGE-large) + stores in LanceDB
        │
        ▼
main.py                → orchestrates the steps above, then lets you
                          type a query and returns the top-k most
                          semantically similar chunks (cosine similarity)
```

## Project structure

| File                  | Responsibility                                              |
|------------------------|--------------------------------------------------------------|
| `main.py`              | Entry point — runs the full pipeline end to end.             |
| `document_loader.py`   | Reads `.txt` / `.pdf` / `.docx` files into raw text.          |
| `text_cleaner.py`      | Normalizes whitespace in extracted text.                     |
| `chunk_creator.py`     | Splits text into overlapping chunks for embedding.           |
| `vector_store.py`      | Creates embeddings and handles LanceDB storage + search.     |
| `Documents/`           | Sample input documents.                                      |
| `Output/`              | Generated `chunks.json` (created on first run).              |
| `vector_db/`           | LanceDB's local database files (created on first run).       |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Set `INPUT_FILE` in `main.py` to the document you want to ingest.
2. Run:
   ```bash
   python main.py
   ```
3. The script will load, clean, chunk, save the chunks to
   `Output/chunks.json`, embed + store them in LanceDB, and then prompt you
   for a search query. It prints the top 3 most relevant chunks with their
   source file, page number, and cosine similarity score.

## Notes

- Embedding model: `BAAI/bge-large-en-v1.5` (1024-dimensional embeddings).
  Larger and more accurate than smaller models like `all-MiniLM-L6-v2`, at
  the cost of slower encoding and more memory usage.
- Similarity metric: cosine similarity (embeddings are normalized, and
  LanceDB's cosine distance is converted back to similarity for readability).
- Re-running the pipeline on the same documents is safe — `store_chunks()`
  replaces existing chunks with matching IDs instead of duplicating them.
