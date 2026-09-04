"""
main.py
-------
Entry point for the RAG (Retrieval-Augmented Generation) preprocessing +
search pipeline named "LanceRAG".

Pipeline stages, in order:
    0. Choose     - ask the user whether to ingest a Wikipedia article or
                     every file in the Documents folder, and get the
                     topic if needed (this file)
    1. Load       - read the chosen source(s) into raw text
                     (document_loader.py)
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
import sys
from document_loader import load_document, load_documents_folder
from text_cleaner import clean_text
from chunk_creator import create_chunks
from vector_store import count, store_chunks, search_chunks

# -----------------------------------
# Output path
# -----------------------------------
DOCUMENTS_FOLDER = "Documents"  # folder scanned for .txt / .pdf / .docx files
OUTPUT_FILE = "Output/chunks.json"  # where the generated chunks are saved for inspection

#------------------------------------------
# Step 0: Ask the user for a data source
#------------------------------------------
print("Where should I get the data from?")
print("  1. Wikipedia")
print("  2. Documents folder")
source_choice = input("Enter 1 or 2: ").strip()

#------------------------------------------
# Step 1: Load Document(s)
#------------------------------------------
# Reads the chosen source and returns a list of {content, source, page}
# dicts (one entry per page for PDFs, one entry total for .txt/.docx/
# Wikipedia).
if source_choice == "1":
    topic = input("Enter the Wikipedia topic: ").strip()

    if not topic:
        print("No topic entered. Exiting.")
        sys.exit(1)

    try:
        documents = load_document(f"wiki:{topic}")
    except Exception as e:
        print(f"Could not load Wikipedia topic '{topic}': {e}")
        sys.exit(1)

elif source_choice == "2":
    try:
        documents = load_documents_folder(DOCUMENTS_FOLDER)
    except Exception as e:
        print(f"Could not load documents folder '{DOCUMENTS_FOLDER}': {e}")
        sys.exit(1)

else:
    print("Invalid choice. Please enter 1 or 2. Exiting.")
    sys.exit(1)

print(f"Loaded pages/documents: {len(documents)}")

if not documents:
    print("Nothing was loaded, so there's nothing to chunk. Exiting.")
    sys.exit(1)

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
# Step 3: Ask for chunk settings, then create chunks
#------------------------------------------
# Splits each document's text into overlapping word-count-based chunks,
# which are the actual units that get embedded and searched.
DEFAULT_CHUNK_SIZE = 120
DEFAULT_OVERLAP = 20

chunk_size_input = input(f"Enter chunk size (press Enter for default {DEFAULT_CHUNK_SIZE}): ").strip()
overlap_input = input(f"Enter overlap (press Enter for default {DEFAULT_OVERLAP}): ").strip()

chunk_size = DEFAULT_CHUNK_SIZE
overlap = DEFAULT_OVERLAP

if chunk_size_input:
    try:
        chunk_size = int(chunk_size_input)
        if chunk_size <= 0:
            raise ValueError
    except ValueError:
        print(f"Invalid chunk size '{chunk_size_input}', using default {DEFAULT_CHUNK_SIZE}.")
        chunk_size = DEFAULT_CHUNK_SIZE

if overlap_input:
    try:
        overlap = int(overlap_input)
        if overlap < 0:
            raise ValueError
    except ValueError:
        print(f"Invalid overlap '{overlap_input}', using default {DEFAULT_OVERLAP}.")
        overlap = DEFAULT_OVERLAP

# Overlap has to be smaller than chunk size, or the sliding window in
# chunk_creator.py stalls (start never advances) or skips text entirely.
if overlap >= chunk_size:
    fallback_overlap = DEFAULT_OVERLAP if DEFAULT_OVERLAP < chunk_size else chunk_size // 4
    print(
        f"Overlap ({overlap}) must be smaller than chunk size ({chunk_size}); "
        f"using overlap {fallback_overlap} instead."
    )
    overlap = fallback_overlap

print(f"Using chunk size {chunk_size}, overlap {overlap}.")

chunks = create_chunks(
    documents,
    chunk_size=chunk_size,
    overlap=overlap
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