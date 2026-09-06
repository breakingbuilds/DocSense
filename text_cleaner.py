"""
text_cleaner.py
----------------
Small utility module that normalizes raw extracted text before it gets
chunked and embedded. Cleaner text -> more consistent chunks -> better
embeddings.
"""

import re


def clean_text(text):
    """
    Collapse all whitespace (spaces, tabs, newlines, repeated blank lines
    from PDF extraction, etc.) into single spaces, and trim leading /
    trailing whitespace.

    Args:
        text (str | None): Raw text extracted from a document. May be
            None if a PDF page had no extractable text.

    Returns:
        str: Cleaned, single-line-spaced text ("" if input was None).
    """
    if text is None:
        return ""

    # Safety net: strip leftover citation/footnote artifacts, in case
    # any slip through the loader (e.g. "[ 1 ]", "[ a ]", stray "↑"
    # footnote-back-reference arrows from Wikipedia extraction).
    # document_loader.py already removes most of these at the source,
    # but this keeps text_cleaner.py robust for any input source.
    text = re.sub(r"\[\s*[0-9]+\s*\]", " ", text)   # e.g. "[ 106 ]"
    text = re.sub(r"\[\s*[a-z]\s*\]", " ", text)    # e.g. "[ a ]"
    text = text.replace("↑", " ")

    # Replace any run of whitespace characters (spaces, tabs, newlines...)
    # with a single space, so text extracted from PDFs/DOCX (which often
    # has irregular line breaks) becomes one clean, continuous string.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = text.strip()

    return text