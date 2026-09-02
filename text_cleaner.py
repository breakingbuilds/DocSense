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
