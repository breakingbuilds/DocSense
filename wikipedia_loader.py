"""
wikipedia_loader.py
--------------------
Optional standalone entry point for ingesting a Wikipedia article, kept
separate from main.py so you can build/inspect chunks.json for a topic
without also running the (slower) embed + store + search steps.

Note: since document_loader.load_document() now understands a "wiki:"
prefix, you can also just set, in main.py:

    INPUT_FILE = "wiki:Digital electronics"

...and run main.py as usual to go straight through embedding + storage +
search. This script is for when you only want the clean/chunk/save steps.

Reuses the SAME modules as main.py (text_cleaner, chunk_creator) so the
output JSON has an identical shape and identical Chunk_ID numbering to
chunks produced from local files.
"""

import json
import os

from document_loader import load_document
from text_cleaner import clean_text
from chunk_creator import create_chunks

OUTPUT_FILE = "Output/wikipedia_chunks.json"


def build_wikipedia_chunks(topic, chunk_size=120, overlap=20):
    """
    Fetch a Wikipedia topic and turn it into chunks, using the exact same
    clean -> chunk steps main.py uses for local files.

    Args:
        topic (str): Article title, e.g. "Digital electronics".
        chunk_size (int): Words per chunk (kept in sync with main.py's default).
        overlap (int): Overlapping words between chunks.

    Returns:
        list[dict]: Chunks in the standard Chunk_ID/Text/Source/Page/Chunk_Index shape.
    """
    documents = load_document(f"wiki:{topic}")
    print(f"Loaded pages/documents: {len(documents)}")

    for doc in documents:
        doc["content"] = clean_text(doc["content"])

    chunks = create_chunks(documents, chunk_size=chunk_size, overlap=overlap)
    print(f"Total chunks created: {len(chunks)}")

    return chunks


def save_chunks(chunks, output_file=OUTPUT_FILE):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(chunks)} chunks to {output_file}")


if __name__ == "__main__":
    topic = input("Enter Wikipedia topic: ").strip()

    if not topic:
        print("No topic entered.")
    else:
        chunks = build_wikipedia_chunks(topic)

        if chunks:
            save_chunks(chunks)

            # Uncomment to also embed + store into LanceDB right away:
            #
            # from vector_store import store_chunks, count
            # store_chunks(chunks)
            # print(f"Vector records: {count()}")
        else:
            print("No chunks were created.")