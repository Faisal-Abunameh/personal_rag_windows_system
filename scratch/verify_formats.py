import os
import sys
from pathlib import Path

# Add project roots to sys.path
sys.path.append(os.getcwd())

from app.services.document_parser import parse_document
from app.config import REFERENCES_DIR

def test_excel():
    print("--- Testing Excel Parsing ---")
    # Search for an excel file in references
    excel_files = list(REFERENCES_DIR.glob("*.xlsx"))
    if not excel_files:
        print("No Excel files found in references.")
        return
    
    target = excel_files[0]
    print(f"Parsing: {target}")
    try:
        result = parse_document(target)
        print(f"Success! Extracted {len(result['text'])} characters.")
        print(f"Snippet: {result['text'][:200]}...")
    except Exception as e:
        print(f"FAILED: {e}")

def test_code():
    print("\n--- Testing Code Parsing ---")
    # Parse ourselves!
    target = Path(__file__)
    print(f"Parsing: {target}")
    try:
        result = parse_document(target)
        print(f"Success! Extracted {len(result['text'])} characters.")
        print(f"Snippet: {result['text'][:200]}...")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_excel()
    test_code()
