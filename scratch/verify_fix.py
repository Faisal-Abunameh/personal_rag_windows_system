import sys
from pathlib import Path
from app.services.document_parser import parse_document

def verify_fix(file_path):
    print(f"Verifying fix on: {file_path}")
    try:
        result = parse_document(file_path)
        text = result["text"]
        print("SUCCESS!")
        print(f"File: {result['filename']}")
        print(f"Content Length: {len(text)} characters")
        print("-" * 40)
        print(f"Text Preview (first 500 chars):\n{text[:500]}")
        
        # Check if binary markers are gone
        if "%PDF" in text[:100] or "obj <<" in text[:500]:
            print("\nWARNING: Binary markers detected! The fix might have failed.")
        else:
            print("\nCONFIRMED: Output appears to be clean text.")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    pdf_path = Path("e:/Programming/personal_rag_windows_system/data/uploads/6e9e47b9_Project-598750-EPP-1-2018-1-DE-EPPKA2-CBHE-JP.pdf")
    verify_fix(pdf_path)
