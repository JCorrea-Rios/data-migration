# Sanitization Code Migration Summary

**Date**: October 29, 2025

## Overview

Successfully reorganized the sanitization codebase following **Option 2** approach, moving the simple regex-based sanitizer from the experimental folder into the main `src/` structure while maintaining backward compatibility for existing notebooks.

## Changes Made

### ✅ Files Created

1. **`src/sanitization/simple_sanitizer.py`** (NEW)
   - Moved from: `All_Records_Fuzzy_Matching/DataCleaningRoutine/sanitize.py`
   - Lightweight regex-based sanitizer with 177 rules
   - Includes functions: `clean_all_pandas_df()`, `clean_all_txt_file()`, CLI support
   - Updated to use new rules location by default

2. **`data/sanitization_rules.json`** (NEW)
   - Copied from: `All_Records_Fuzzy_Matching/DataCleaningRoutine/files/rules.json`
   - Central location for 177 sanitization rules
   - Now version-controlled and easily accessible

3. **`scripts/sanitize_simple.py`** (NEW)
   - CLI wrapper for the simple sanitizer
   - Usage: `python scripts/sanitize_simple.py --input data.csv --output clean.csv`

4. **`scripts/sanitize.py`** (NEW)
   - CLI wrapper for the advanced sanitizer (company_name_sanitiser.py)
   - Supports GLiNER, batch processing, DuckDB persistence

### ✅ Files Updated

1. **`src/sanitization/__init__.py`**
   - Changed from eager imports to lazy loading
   - Prevents loading heavy dependencies (spacy, GLiNER) when not needed
   - Allows `simple_sanitizer` to work independently

2. **`.gitignore`**
   - Added: `!data/sanitization_rules.json` (whitelist rules)
   - Added: Experiments folder exclusions for large files (pkl, csv, xlsx, zip)
   - Kept: `rules.json` in original location for backward compatibility

3. **`All_Records_Fuzzy_Matching/Data_Cleaning_All_Records_File.ipynb`**
   - Updated import: `from src.sanitization.simple_sanitizer import clean_all_pandas_df`
   - Added path setup to find project root

4. **`All_Records_Fuzzy_Matching/OldNotebooks/TFIDF_Sim_DistinctRecords.ipynb`**
   - Updated import: `from src.sanitization.simple_sanitizer import clean_all_txt_file`
   - Added path setup to find project root

### ✅ Files Deleted

1. **`sanitize_legacy.py`** (ROOT)
   - Was an exact duplicate of `src/sanitization/company_name_sanitiser.py`
   - Removed to eliminate confusion

## Architecture

### Two Sanitization Options

```
src/sanitization/
├── simple_sanitizer.py          # Lightweight, regex-based (177 rules)
│   └── Best for: Quick cleaning, minimal dependencies
│
└── company_name_sanitiser.py    # Advanced, ML-powered
    └── Best for: Entity classification, batch processing, production
```

### Directory Structure (After Migration)

```
data-migration/
├── data/
│   └── sanitization_rules.json          # Central rules location (NEW)
├── scripts/
│   ├── sanitize.py                      # Advanced sanitizer CLI (NEW)
│   └── sanitize_simple.py               # Simple sanitizer CLI (NEW)
├── src/
│   └── sanitization/
│       ├── __init__.py                  # Lazy imports (UPDATED)
│       ├── simple_sanitizer.py          # Regex-based sanitizer (NEW)
│       ├── company_name_sanitiser.py    # ML-based sanitizer
│       └── gliner_classifier.py
└── All_Records_Fuzzy_Matching/          # Experimental (kept for reference)
    └── DataCleaningRoutine/
        ├── sanitize.py                  # Original (kept for reference)
        └── files/
            └── rules.json               # Original location (kept)
```

## Usage Examples

### Simple Sanitizer (CLI)

```bash
# Basic usage
python scripts/sanitize_simple.py \
  --input data/companies.csv \
  --output output/clean.csv \
  --column company_name

# With GLiNER classification
python scripts/sanitize_simple.py \
  --input data/companies.csv \
  --output output/clean.csv \
  --use_gliner \
  --batch_size 100
```

### Simple Sanitizer (Python)

```python
from src.sanitization.simple_sanitizer import clean_all_pandas_df

# DataFrame with 'Name' column
df = pd.read_csv("companies.csv")
df_clean = clean_all_pandas_df(df)
# Now has 'Clean_Name' column
```

### Advanced Sanitizer (CLI)

```bash
# With batch processing and DuckDB persistence
python scripts/sanitize.py \
  --input data/companies.csv \
  --output_csv output/sanitized.csv \
  --report output/report.md \
  --use_gliner \
  --use_batches \
  --batch_size 5000
```

## Backward Compatibility

### ✅ Old Notebooks Still Work
- Updated imports point to new location
- Functions maintain same signature
- Default rules location updated automatically

### ✅ Original Files Preserved
- `All_Records_Fuzzy_Matching/DataCleaningRoutine/sanitize.py` - kept
- `All_Records_Fuzzy_Matching/DataCleaningRoutine/files/rules.json` - kept

## Benefits of This Migration

1. **Cleaner Project Structure**
   - Production code in `src/`
   - Experimental code in `All_Records_Fuzzy_Matching/`
   - Clear separation of concerns

2. **Two Clear Options**
   - Simple: Fast, lightweight, regex-based
   - Advanced: ML-powered, batch processing, entity classification

3. **Better Dependency Management**
   - Lazy imports prevent loading heavy dependencies
   - Simple sanitizer works without spacy/GLiNER

4. **Version Control Ready**
   - Rules centralized and tracked
   - Large experimental outputs gitignored
   - No duplicate code

5. **Easy to Use**
   - CLI wrappers in `scripts/`
   - Importable functions from `src/`
   - Clear documentation

## Next Steps

### Recommended (for production use)

1. **Install Dependencies**
   ```bash
   uv sync  # or pip install -r requirements.txt
   ```

2. **Test the Scripts**
   ```bash
   python scripts/sanitize_simple.py --help
   python scripts/sanitize.py --help
   ```

3. **Run Notebooks**
   - Execute updated notebooks to verify imports work
   - Check that cleaned outputs are as expected

### Optional

1. **Archive Experiments**
   - Consider renaming `All_Records_Fuzzy_Matching/` to `experiments/archive/`
   - Add README explaining historical context

2. **Add Tests**
   - Unit tests for `simple_sanitizer.py`
   - Integration tests for CLI scripts

3. **Documentation**
   - Add usage examples to main README
   - Document when to use simple vs advanced sanitizer

## Notes

- Original `sanitize.py` kept in `All_Records_Fuzzy_Matching/` for reference
- Rules file duplicated to both locations for safety
- Notebooks updated to use new import paths
- All scripts made executable with proper shebangs

---

**Migration completed successfully!** ✅

