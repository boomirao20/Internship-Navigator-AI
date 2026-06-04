import sys
import traceback
try:
    from fpdf import FPDF
    print("fpdf successfully imported")
    print(f"FPDF module path: {sys.modules['fpdf']}")
except Exception as e:
    print("Failed to import fpdf:")
    traceback.print_exc()
