#!/usr/bin/env python3
"""
Simple Company Name Sanitizer CLI

Lightweight regex-based sanitizer for quick cleaning.
For advanced features (GLiNER, batch processing, DuckDB), use sanitize.py instead.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sanitization.simple_sanitizer import main

if __name__ == "__main__":
    main()

