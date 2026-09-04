"""
document_loader.py
-------------------
This module is the "ingestion" layer of the RAG pipeline: it knows how to
open a raw source (a local .txt / .pdf / .docx file, OR a Wikipedia topic)
and turn it into a common, predictable format that the rest of the
pipeline (cleaner, chunker, vector store) can work with, regardless of the
original source type.

Every loader function below returns a list of dicts shaped like:
    {
        "content": "<raw extracted text>",
        "source":  "<path to the original file, or a URL>",
        "page":    <page number, or None if the source has no pages>
    }
A PDF returns one dict per page; a .txt, .docx, or Wikipedia page returns
a single dict for the whole document (since they don't have a native
"page" concept).
"""

import os
import requests
from bs4 import BeautifulSoup
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

# ------------------------------------------
# Resolve a (possibly inexact) topic to a real article title
# ------------------------------------------
def resolve_wikipedia_title(topic):
    """
    Use Wikipedia's search API to find the real article title for a topic,
    the same way typing a phrase into Wikipedia's search box and hitting
    "Search" would.

    This is what makes ingestion work for natural phrases like
    "Python programming" even though the real article is titled
    "Python (programming language)" - building the URL directly from the
    user's exact words would 404 / return an empty page.

    Args:
        topic (str): Whatever the user typed, e.g. "Python programming".

    Returns:
        str | None: The resolved, real article title (e.g.
            "Python (programming language)"), or None if no article
            matches the topic at all.
    """
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "format": "json",
        "srlimit": 1,
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(api_url, params=params, headers=headers, timeout=10)
    response.raise_for_status()

    results = response.json().get("query", {}).get("search", [])

    if not results:
        return None

    return results[0]["title"]


# ------------------------------------------
# Load from Wikipedia
# ------------------------------------------
def load_wikipedia(topic):
    """
    Fetch a Wikipedia article and wrap it in the standard document format,
    just like load_txt/load_pdf/load_docx do for local files.

    The topic is first resolved via resolve_wikipedia_title() so that
    natural search phrases (not just exact article titles) work.

    Args:
        topic (str): A topic/search phrase, e.g. "Python programming".

    Returns:
        list[dict]: A single-item list with "content" (paragraph + list
            text from the article), "source" (the article URL), and
            "page" (None, since web pages have no page concept).

    Raises:
        Exception: if no article matches the topic, the page can't be
            fetched (non-200 status), or the main content div isn't found.
    """
    resolved_title = resolve_wikipedia_title(topic)

    if resolved_title is None:
        raise Exception(f"No Wikipedia article found for topic: '{topic}'")

    if resolved_title.lower() != topic.strip().lower():
        print(f"Showing results for: '{resolved_title}'")

    url_topic = resolved_title.strip().replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{url_topic}"

    # Wikipedia's etiquette wants a descriptive User-Agent (app name +
    # contact info), not a spoofed browser string.
    headers = {
        "User-Agent": "LanceRAG/1.0 (https://example.com/contact; you@example.com)"
    }
    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch Wikipedia page (status {response.status_code}): {url}")

    soup = BeautifulSoup(response.text, "lxml")

    # IMPORTANT: Wikipedia pages can contain MORE THAN ONE element with
    # class "mw-parser-output" - e.g. the little protection-padlock icon
    # tooltip near the top of the page is rendered through the same
    # wikitext parser and gets wrapped in its own tiny "mw-parser-output"
    # div, which appears in the HTML *before* the real article body.
    # soup.find() grabs the first match, which can silently be that tiny
    # icon snippet instead of the actual article - giving you an empty
    # (or near-empty) result with no error raised.
    #
    # The real article content always lives inside <div id="mw-content-text">,
    # so scope the search to there instead of searching the whole page.
    content_wrapper = soup.find("div", id="mw-content-text")
    if content_wrapper is None:
        raise Exception(f"Could not find #mw-content-text for: {url}")

    content = content_wrapper.find("div", class_="mw-parser-output")
    if content is None:
        raise Exception(f"Could not find article content for: {url}")

    # Paragraphs + list items, same extraction logic as the original
    # scraping script this loader was based on.
    elements = content.find_all(["p", "li"])
    pieces = [el.get_text(" ", strip=True) for el in elements]
    pieces = [p for p in pieces if p]  # drop empties

    text = "\n".join(pieces)

    return [{
        "content": text,
        "source": url,
        "page": None  # Wikipedia articles have no page concept
    }]


# ------------------------------------------
# Load every supported file inside a folder
# ------------------------------------------
# Maps file extension -> the loader function that handles it. Add a new
# entry here if you ever want to support another file type.
SUPPORTED_EXTENSIONS = {
    ".txt": load_txt,
    ".pdf": load_pdf,
    ".docx": load_docx,
}


def load_documents_folder(folder_path):
    """
    Walk through every file inside `folder_path` (including subfolders)
    and load each supported file (.txt, .pdf, .docx) with its matching
    loader above, combining everything into one list of documents.

    Files with an unsupported extension are skipped with a printed
    warning instead of crashing the whole run. A file that fails to load
    (corrupt, unreadable, etc.) is also skipped with a warning so one bad
    file doesn't stop the rest of the folder from being processed.

    Args:
        folder_path (str): Path to the folder containing source documents,
            e.g. "Documents".

    Returns:
        list[dict]: All documents from every file found, in the same
            {content, source, page} shape as the individual loaders.

    Raises:
        Exception: if `folder_path` doesn't exist / isn't a directory.
    """
    if not os.path.isdir(folder_path):
        raise Exception(f"Documents folder not found: {folder_path}")

    all_documents = []
    files_loaded = 0

    for root, _dirs, files in os.walk(folder_path):
        for filename in sorted(files):
            file_path = os.path.join(root, filename)
            extension = os.path.splitext(filename)[1].lower()
            loader = SUPPORTED_EXTENSIONS.get(extension)

            if loader is None:
                print(f"Skipping unsupported file: {file_path}")
                continue

            try:
                documents = loader(file_path)
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
                continue

            all_documents.extend(documents)
            files_loaded += 1
            print(f"Loaded: {file_path} ({len(documents)} page(s)/doc(s))")

    print(f"Loaded {files_loaded} file(s), {len(all_documents)} page(s)/doc(s) total.")
    return all_documents


def load_document(file_path):
    """
    Entry point used by main.py: figures out whether `file_path` is a
    local file (routed by extension) or a Wikipedia topic (prefixed with
    "wiki:"), and calls the matching loader above.

    Examples:
        load_document(r"Documents\\Unit01.pdf")       -> load_pdf(...)
        load_document("wiki:Digital electronics")     -> load_wikipedia("Digital electronics")

    Raises:
        Exception: if the file extension isn't one of .txt, .pdf, .docx,
            and the input isn't a "wiki:" reference either.
    """
    if file_path.lower().startswith("wiki:"):
        topic = file_path[len("wiki:"):]
        return load_wikipedia(topic)

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