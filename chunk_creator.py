"""
chunk_creator.py
-----------------
Splits cleaned document text into smaller, overlapping "chunks". Chunking
matters because:
  - Embedding models work best on short passages, not entire documents.
  - Overlap between consecutive chunks preserves context that would
    otherwise be lost right at the cut point (e.g. a sentence split in half).
"""

def create_chunks(
    documents,
    chunk_size=10,
    overlap=2
):
    """
    Slide a fixed-size window over each document's word list to build chunks.

    Args:
        documents (list[dict]): Cleaned documents from document_loader.py,
            each with "content", "source", and "page" keys.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of words repeated between one chunk and the
            next, so context isn't lost at chunk boundaries.

    Returns:
        list[dict]: One dict per chunk with keys:
            Chunk_ID     - globally unique, increasing integer ID
            Text         - the chunk's text
            Source       - originating file path
            Page         - originating page number (or None)
            Chunk_Index  - position of this chunk within its source document
    """
    chunks = []
    chunk_counter = 1  # unique ID across ALL documents, not reset per document

    for doc in documents:
        text = doc["content"]
        words = text.split()
        start = 0
        index = 0  # position of this chunk within the current document

        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "Chunk_ID": chunk_counter,
                "Text": chunk_text,
                "Source": doc["source"],
                "Page": doc["page"],
                "Chunk_Index": index
            })

            chunk_counter += 1
            index += 1
            # Move the window forward by (chunk_size - overlap) words, so
            # the last `overlap` words of this chunk reappear at the start
            # of the next one.
            start = end - overlap

    return chunks
