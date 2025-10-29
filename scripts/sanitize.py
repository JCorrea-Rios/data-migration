#!/usr/bin/env python3
"""
Company Name Sanitization CLI

Production-ready sanitization with:
- Unicode normalization
- Advanced regex cleaning
- GLiNER entity classification (optional)
- Multiple match keys (strict, aggressive)
- Batch processing with DuckDB resumability
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sanitization.company_name_sanitiser import main

if __name__ == "__main__":
    main()

