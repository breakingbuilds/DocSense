"""
vector_store.py
----------------
This module is the "storage + retrieval" layer of the RAG pipeline.

What it does:
1. Turns text chunks into vector embeddings using a Sentence-Transformers model.
2. Stores those embeddings (plus their metadata) inside a local LanceDB table.
3. Lets you run a semantic search: turn a query into a vector, then find the
   most similar chunks already stored in LanceDB (using cosine similarity).

Think of it as a mini search engine: instead of matching keywords, it matches
"meaning" by comparing vectors.
"""

import lancedb
from sentence_transformers import SentenceTransformer

# -----------------------------------
# Embedding model
# -----------------------------------
# We use BAAI/bge-small-en-v1.5 ("BGE-small"): a good quality/speed
# tradeoff that produces 384-dimensional embeddings (vs. 1024 for
# BGE-large), so it's noticeably faster and lighter on memory while still
# outperforming smaller/older models like all-MiniLM-L6-v2 on retrieval
# benchmarks.
MODEL_NAME = "BAAI/bge-small-en-v1.5"
model = SentenceTransformer(MODEL_NAME)

# BGE models are trained to work best when the QUERY (not the stored
# documents/chunks) is prefixed with this instruction string. This is a
# quirk specific to the BGE model family and is what the "-en-v1.5" model
# card recommends for retrieval tasks. Leaving it out still works, but
# search quality is noticeably better with it.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# -----------------------------------
# LanceDB connection
# -----------------------------------
# lancedb.connect() creates (or opens) a local, file-based vector database
# under the "vector_db" folder. No external server is required.
db = lancedb.connect("vector_db")
TABLE_NAME = "course_documents"


# -----------------------------------
# create embeddings for the chunks of text
# -----------------------------------
def create_embeddings(chunks):
    """
    Convert a list of text chunks into a list of embedding vectors.

    Args:
        chunks (list[dict]): Each dict must contain a "Text" key
            (see chunk_creator.py for the exact shape).

    Returns:
        list[list[float]]: One embedding vector per chunk, same order as input.
        Returns an empty list if `chunks` is empty.
    """
    if not chunks:
        return []

    # Pull out just the raw text of every chunk, since that's all the
    # embedding model needs.
    texts = [
        chunk["Text"]
        for chunk in chunks
    ]

    # normalize_embeddings=True scales every vector to unit length, which
    # makes the dot product between two vectors equal to their cosine
    # similarity. This matches the "cosine" metric LanceDB uses below, so
    # the numbers stay consistent between indexing and searching.
    #
    # Note: documents/chunks are NOT prefixed with BGE_QUERY_INSTRUCTION -
    # that prefix is only added to the search query, not to stored text.
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings.tolist()


# -----------------------------------
# helpers
# -----------------------------------
def _get_table():
    """Return the LanceDB table if it already exists, otherwise None."""
    if TABLE_NAME not in db.table_names():
        return None
    return db.open_table(TABLE_NAME)


def count():
    """Return how many chunks are currently stored in the vector store."""
    table = _get_table()
    return table.count_rows() if table is not None else 0


# -----------------------------------
# store chunks
# -----------------------------------
def store_chunks(chunks):
    """
    Embed a list of chunks and save them (with metadata) into LanceDB.

    Behaves like an "upsert": if a chunk with the same ID already exists
    in the table, its old copy is deleted before the new one is added, so
    re-running the pipeline on the same documents doesn't create duplicates.

    Args:
        chunks (list[dict]): Output of chunk_creator.create_chunks().
    """
    if not chunks:
        print("No chunks to store.")
        return

    embeddings = create_embeddings(chunks)

    # Build one LanceDB row per chunk, pairing each chunk with its vector.
    rows = []
    for chunk, embedding in zip(chunks, embeddings):
        rows.append({
            "id": str(chunk["Chunk_ID"]),
            "text": chunk["Text"],
            "vector": embedding,
            "source": str(chunk["Source"]),
            "page": chunk["Page"] if chunk["Page"] is not None else 0,
            "chunk_index": chunk["Chunk_Index"],
        })

    table = _get_table()

    if table is None:
        # First run: no table exists yet, so create it from these rows.
        db.create_table(TABLE_NAME, data=rows)
    else:
        # Table already exists: remove any rows we're about to re-add
        # (by ID) before inserting the fresh versions. This keeps the
        # store idempotent when the pipeline is re-run.
        ids_to_replace = ", ".join(f"'{row['id']}'" for row in rows)
        table.delete(f"id IN ({ids_to_replace})")
        table.add(rows)

    print(f"Stored {len(chunks)} chunks in the vector store.")


# -----------------------------------
# semantic search (with cosine similarity)
# -----------------------------------
def search_chunks(query, top_k=5):
    """
    Find the chunks most semantically similar to a natural-language query.

    Args:
        query (str): The user's search query, in plain text.
        top_k (int): How many top matches to return.

    Returns:
        list[dict] | None: A list of result dicts (text, source, page,
            chunk_index, similarity), sorted from most to least similar.
            Returns [] if the query is empty, or None if the store has
            no data yet.
    """
    if not query:
        print("Query is empty.")
        return []

    table = _get_table()
    if table is None or table.count_rows() == 0:
        print("No chunks in the vector store.")
        return None

    # BGE models expect an instruction prefix on the QUERY side only, to
    # help the model distinguish "this is what I'm looking for" from the
    # stored passages. This step is skipped for the chunks themselves.
    prefixed_query = BGE_QUERY_INSTRUCTION + query

    query_embedding = model.encode(
        [prefixed_query],
        normalize_embeddings=True
    )[0].tolist()

    # Ask LanceDB for the top_k nearest vectors using cosine distance.
    raw_results = (
        table.search(query_embedding)
        .metric("cosine")
        .limit(top_k)
        .to_list()
    )

    # LanceDB returns cosine DISTANCE (1 - cosine similarity) in "_distance".
    # Convert it back to similarity so it's intuitive to read (1.0 = identical).
    results = []
    for row in raw_results:
        cosine_distance = row["_distance"]
        cosine_similarity = 1 - cosine_distance
        results.append({
            "text": row["text"],
            "source": row["source"],
            "page": row["page"],
            "chunk_index": row["chunk_index"],
            "similarity": cosine_similarity,
        })

    return results
