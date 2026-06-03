"""Extract text (and optionally images) from the TOMAS paper PDF.

Run with the Windows Python that has PyMuPDF (fitz) installed:
    py tools/extract_pdf.py
"""
import sys
import fitz  # PyMuPDF

SRC = r"C:/Users/Bingxiao Du/Downloads/s00158-024-03835-6.pdf"
OUT = r"E:/Working/reproduceTOMAS/paper_text.txt"


def main():
    doc = fitz.open(SRC)
    parts = []
    for i, page in enumerate(doc):
        parts.append(f"\n\n===== PAGE {i + 1} / {doc.page_count} =====\n\n")
        parts.append(page.get_text("text"))
    text = "".join(parts)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {OUT}: pages={doc.page_count} chars={len(text)}")


if __name__ == "__main__":
    main()
