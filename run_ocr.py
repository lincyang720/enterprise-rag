import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import ocr_ingest

if __name__ == "__main__":
    ocr_ingest.main()
