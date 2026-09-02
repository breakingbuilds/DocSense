"""
main.py
-------
Entry point for the RAG (Retrieval-Augmented Generation) preprocessing +
search pipeline named "LanceRAG".

Pipeline stages, in order:
    1. Load       - read a source document (.txt / .pdf / .docx) into
                     raw text (document_loader.py)
    2. Clean       - normalize whitespace in that text (text_cleaner.py)
    3. Chunk       - split the cleaned text into small overlapping pieces
                     (chunk_creator.py)
    4. Save        - write the chunks to a JSON file for inspection/debugging
    5. Embed+Store - turn each chunk into a vector embedding and store it
                     in a local LanceDB table (vector_store.py)
    6. Search      - take a query from the user, embed it the same way,
                     and return the most semantically similar chunks

Run it with:  python main.py
"""

import json
import os
from document_loader import load_document
from text_cleaner import clean_text
from chunk_creator import create_chunks
from vector_store import count, store_chunks, search_chunks

# -----------------------------------
# Input/Output file paths
# -----------------------------------
INPUT_FILE = r"Documents\employee_training_islamabad.pdf"  # source document to ingest
OUTPUT_FILE = "Output/chunks.json"  # where the generated chunks are saved for inspection

#------------------------------------------
# Step 1: Load Document
#------------------------------------------
# Reads the source file and returns a list of {content, source, page} dicts
# (one entry per page for PDFs, one entry total for .txt/.docx).
documents = load_document(INPUT_FILE)
print(f"Loaded pages/documents: {len(documents)}")

#------------------------------------------
# Step 2: Clean Text
#------------------------------------------
# Normalizes whitespace in each document's text so chunking/embedding
# operate on tidy, consistent input.
for doc in documents:
    doc["content"] = clean_text(
        doc["content"]
    )

#------------------------------------------
# Step 3: Create Chunks
#------------------------------------------
# Splits each document's text into overlapping word-count-based chunks,
# which are the actual units that get embedded and searched.
chunks = create_chunks(
    documents,
    chunk_size=120,
    overlap=20
)
print(f"Total chunks created: {len(chunks)}")

#------------------------------------------
# Step 4: Save chunks to Json
#------------------------------------------
# Persists the chunks to disk as JSON, mainly so you can open the file and
# manually inspect/debug what got chunked and how.
os.makedirs(
    "output",
    exist_ok=True
)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        chunks,
        f,
        indent=4,
        ensure_ascii=False
    )
print("RAG chunks created successfully!")
print(f"Saved at: {OUTPUT_FILE}")

# -----------------------------------
# Step 5: Store in LanceDB
# -----------------------------------
# Embeds every chunk with the BGE-large sentence-transformer model and
# upserts the resulting vectors (+ metadata) into the local LanceDB table.
store_chunks(chunks)
print(f"Vector records: {count()}")

# -----------------------------------
# Step 6: Semantic Search
# -----------------------------------
# Lets the user type a natural-language query, embeds it the same way the
# chunks were embedded, and retrieves the top-k most similar chunks by
# cosine similarity.
query = input("Enter your query: ").strip()

if query:
    results = search_chunks(
        query,
        top_k=3
    )
    if results:
        print("\nTop 3 results:")

        for index, result in enumerate(
            results,
            start=1
        ):
            print(
                f"\nResult {index}:"
            )
            print(
                "Text: ", result["text"]
            )
            print(
                "Source: ", result["source"]
            )
            print(
                "Page: ", result["page"]
            )
            print(
                f"Cosine Similarity: {result['similarity']:.4f}"
            )
