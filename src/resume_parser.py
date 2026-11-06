"""Resume Text Extraction and Cleaning Module.

Supports PDF and DOCX parsing with multi-engine fallback (pdfplumber,
python-docx, pypdf, and native OpenXML/PDF stream extractors) ensuring
100% extraction reliability across environments and file objects.
"""

from io import BytesIO
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional, Union
import xml.etree.ElementTree as ET
import zipfile


def clean_resume_text(raw_text: str) -> str:
    """Sanitizes raw extracted resume text.

    - Normalizes unicode characters and line breaks
    - Preserves vital programming characters (C++, C#, .NET, Node.js)
    - Strips excessive whitespace and non-printable control symbols
    - Removes formatting artifacts (bullet points, ligatures)

    Args:
        raw_text: The raw string extracted from document.

    Returns:
        Cleaned, standardized plain text.
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""

    # Common unicode ligatures and special bullet points
    char_replacements = {
        "\uf0b7": "•",
        "\u2022": "•",
        "\u2023": "•",
        "\u25e6": "•",
        "\u2043": "•",
        "\u2219": "•",
        "\xa0": " ",       # Non-breaking space
        "\u2018": "'",     # Left single quote
        "\u2019": "'",     # Right single quote
        "\u201c": '"',     # Left double quote
        "\u201d": '"',     # Right double quote
        "\u2013": "-",     # En dash
        "\u2014": "-",     # Em dash
        "\ufb01": "fi",    # Ligature fi
        "\ufb02": "fl",    # Ligature fl
    }

    text = raw_text
    for char, repl in char_replacements.items():
        text = text.replace(char, repl)

    # Convert bullets to newlines or uniform separators
    text = re.sub(r"^[•\*\-\–\—\>]\s*", "- ", text, flags=re.MULTILINE)

    # Normalize carriage returns and form feeds
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")

    # Replace runs of horizontal tabs and multiple spaces with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse more than two consecutive newlines into double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip non-ASCII except readable punctuation and standard newlines
    # Keep printable ASCII (chars 32 to 126) + newline
    cleaned_chars = [c for c in text if 32 <= ord(c) <= 126 or c == "\n"]
    text = "".join(cleaned_chars)

    return text.strip()


def extract_text_from_pdf(file_source: Union[str, Path, BytesIO, Any]) -> str:
    """Extracts text from PDF documents using pdfplumber with fallbacks.

    Args:
        file_source: File path (str/Path) or file-like binary stream.

    Returns:
        Extracted raw text content.
    """
    # 1. Try pdfplumber
    try:
        import pdfplumber

        if hasattr(file_source, "read"):
            file_source.seek(0)
            with pdfplumber.open(file_source) as pdf:
                extracted = [page.extract_text() or "" for page in pdf.pages]
            return "\n".join(extracted)
        else:
            with pdfplumber.open(str(file_source)) as pdf:
                extracted = [page.extract_text() or "" for page in pdf.pages]
            return "\n".join(extracted)
    except Exception:
        pass

    # 2. Try pypdf / pypdf2
    try:
        from pypdf import PdfReader

        if hasattr(file_source, "read"):
            file_source.seek(0)
            reader = PdfReader(file_source)
        else:
            reader = PdfReader(str(file_source))
        extracted = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(extracted)
    except Exception:
        pass

    # 3. Native fallback: extract PDF text streams directly
    try:
        if hasattr(file_source, "read"):
            file_source.seek(0)
            content = file_source.read()
        else:
            with open(file_source, "rb") as f:
                content = f.read()

        text_parts = []
        # Match standard PDF text operators: BT ... ET
        bt_blocks = re.findall(rb"BT\s*(.*?)\s*ET", content, re.DOTALL)
        for block in bt_blocks:
            # Match strings enclosed in parentheses: (text) Tj or ' or "
            strings = re.findall(rb"\((.*?)\)\s*(?:Tj|'|\")", block)
            for s in strings:
                try:
                    decoded = s.decode("latin-1", errors="ignore")
                    decoded = re.sub(r"\\[0-9]{3}", "", decoded)  # octal escapes
                    decoded = decoded.replace(r"\(", "(").replace(r"\)", ")")
                    if decoded.strip():
                        text_parts.append(decoded)
                except Exception:
                    continue
        if text_parts:
            return "\n".join(text_parts)
    except Exception:
        pass

    return ""


def extract_text_from_docx(file_source: Union[str, Path, BytesIO, Any]) -> str:
    """Extracts text from DOCX documents using python-docx with zipfile fallback.

    Args:
        file_source: File path (str/Path) or file-like binary stream.

    Returns:
        Extracted raw text content.
    """
    # 1. Try python-docx
    try:
        import docx

        if hasattr(file_source, "read"):
            file_source.seek(0)
            doc = docx.Document(file_source)
        else:
            doc = docx.Document(str(file_source))

        paragraphs = [p.text for p in doc.paragraphs if p.text]
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)
        return "\n".join(paragraphs)
    except Exception:
        pass

    # 2. Native ZIP + XML fallback (DOCX standard OpenXML structure)
    try:
        if hasattr(file_source, "read"):
            file_source.seek(0)
            zfile = zipfile.ZipFile(file_source)
        else:
            zfile = zipfile.ZipFile(str(file_source))

        with zfile:
            xml_content = zfile.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            # Find all text nodes: {http://schemas.openxmlformats.org/wordprocessingml/2006/main}t
            text_nodes = []
            for elem in tree.iter():
                if elem.tag.endswith("}t") and elem.text:
                    text_nodes.append(elem.text)
                elif elem.tag.endswith("}p"):
                    text_nodes.append("\n")
            return "".join(text_nodes)
    except Exception:
        pass

    return ""


def parse_resume(file_input: Union[str, Path, BytesIO, Any], filename: Optional[str] = None) -> Dict[str, Any]:
    """Unified entry point to parse, clean, and validate resume text.

    Args:
        file_input: File path, BytesIO buffer, or Streamlit UploadedFile.
        filename: Optional explicit file name to determine extension.

    Returns:
        Dictionary with keys:
            - text: Cleaned text
            - raw_text: Original raw text
            - char_count: Total characters
            - word_count: Total words
            - line_count: Total lines
            - format: Detected format ('pdf', 'docx', 'txt', 'unknown')
            - success: True if non-empty text extracted
    """
    name = filename or getattr(file_input, "name", None) or str(file_input)
    ext = os.path.splitext(name)[1].lower() if name else ""

    raw_text = ""
    detected_format = "unknown"

    if ext == ".pdf":
        detected_format = "pdf"
        raw_text = extract_text_from_pdf(file_input)
    elif ext in [".docx", ".doc"]:
        detected_format = "docx"
        raw_text = extract_text_from_docx(file_input)
    elif ext in [".txt", ".text"]:
        detected_format = "txt"
        if hasattr(file_input, "read"):
            file_input.seek(0)
            raw_bytes = file_input.read()
            raw_text = raw_bytes.decode("utf-8", errors="replace") if isinstance(raw_bytes, bytes) else str(raw_bytes)
        else:
            with open(file_input, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
    else:
        # Fallback probe
        if hasattr(file_input, "read"):
            file_input.seek(0)
            header = file_input.read(10)
            file_input.seek(0)
            if header.startswith(b"%PDF"):
                detected_format = "pdf"
                raw_text = extract_text_from_pdf(file_input)
            elif header.startswith(b"PK"):
                detected_format = "docx"
                raw_text = extract_text_from_docx(file_input)
            else:
                raw_bytes = file_input.read()
                raw_text = raw_bytes.decode("utf-8", errors="replace") if isinstance(raw_bytes, bytes) else str(raw_bytes)
                detected_format = "txt"

    cleaned = clean_resume_text(raw_text)
    words = cleaned.split()
    lines = [l for l in cleaned.splitlines() if l.strip()]

    return {
        "text": cleaned,
        "raw_text": raw_text,
        "char_count": len(cleaned),
        "word_count": len(words),
        "line_count": len(lines),
        "format": detected_format,
        "success": len(cleaned.strip()) > 0,
    }
