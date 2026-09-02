"""
document_loader.py
-------------------
This module is the "ingestion" layer of the RAG pipeline: it knows how to
open a raw source file (.txt, .pdf, or .docx) and turn it into a common,
predictable format that the rest of the pipeline (cleaner, chunker, vector
store) can work with, regardless of the original file type.

Every loader function below returns a list of dicts shaped like:
    {
        "content": "<raw extracted text>",
        "source":  "<path to the original file>",
        "page":    <page number, or None if the format has no pages>
    }
A PDF returns one dict per page; a .txt or .docx file returns a single
dict for the whole document (since they don't have a native "page" concept).
"""

import os
from pypdf import PdfReader
from docx import Document

# ------------------------------------------
# Load from text file
# ------------------------------------------
def load_txt(file_path):
    """Read a plain .txt file and wrap it in the standard document format."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return [{
        "content": text,
        "source": file_path,
        "page": None  # plain text files don't have pages
    }]

# ------------------------------------------
# Load from pdf file
# ------------------------------------------
def load_pdf(file_path):
    """
    Read a .pdf file and return one document dict per page, so downstream
    steps (and search results) can report exactly which page a chunk of
    text came from.
    """
    reader = PdfReader(file_path)
    documents = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()

        documents.append({
            "content": text,
            "source": file_path,
            "page": page_number + 1  # human-friendly, 1-indexed page number
        })

    return documents

# ------------------------------------------
# Load from docx file
# ------------------------------------------
def load_docx(file_path):
    """
    Read a Word .docx file by concatenating all of its paragraphs into a
    single block of text (Word files don't expose fixed "pages" the way a
    PDF does, so we treat the whole document as one unit).
    """
    doc = Document(file_path)

    text = "\n".join(
        paragraph.text
        for paragraph in doc.paragraphs
    )

    return [{
        "content": text,
        "source": file_path,
        "page": None  # .docx has no reliable page concept
    }]


def load_document(file_path):
    """
    Entry point used by main.py: inspects the file extension and routes
    the file to the matching loader above.

    Raises:
        Exception: if the file extension isn't one of .txt, .pdf, .docx.
    """
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".txt":
        return load_txt(file_path)
    elif extension == ".pdf":
        return load_pdf(file_path)
    elif extension == ".docx":
        return load_docx(file_path)
    else:
        raise Exception(  # noqa: TRY002
            "Unexpected file type."
        )
