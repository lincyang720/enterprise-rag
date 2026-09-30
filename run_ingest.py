import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import ingest

if __name__ == "__main__":
    ingest.main()
