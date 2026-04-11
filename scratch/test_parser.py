import sys
from pathlib import Path
from markitdown import MarkItDown

def test_parse(file_path):
    print(f"Testing MarkItDown on: {file_path}")
    md = MarkItDown()
    try:
        result = md.convert(str(file_path))
        print("Success!")
        print(f"Text Content (first 500 chars):\n{result.text_content[:500]}")
        print("-" * 40)
        print(f"Total Chars: {len(result.text_content or '')}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    pdf_path = Path("e:/Programming/personal_rag_windows_system/data/uploads/6e9e47b9_Project-598750-EPP-1-2018-1-DE-EPPKA2-CBHE-JP.pdf")
    test_parse(pdf_path)
