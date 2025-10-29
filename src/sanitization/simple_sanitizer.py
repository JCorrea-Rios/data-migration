"""
Simple regex-based company name sanitizer.

This is a lightweight sanitizer that uses a rule-based approach with regex patterns.
For more advanced sanitization with entity classification, use company_name_sanitiser.py instead.

Originally from: All_Records_Fuzzy_Matching/DataCleaningRoutine/sanitize.py
"""
import json
import re
from typing import List, Dict
from pathlib import Path

def sanitize_company_name(name: str, rules: List[Dict[str, str]]) -> str:
    """Sanitize a company name according to a list of sanitization rules."""
    for rule in rules:
        pattern = rule['pattern']
        replacement = rule['replacement']
        name = re.sub(pattern, replacement, name)
    return name.strip()

def load_sanitization_rules(file_path: str) -> List[Dict[str, str]]:
    """Load sanitization rules from a JSON file."""
    with open(file_path, 'r') as file:
        rules = json.load(file)
    return rules

def write_sanitized_names(names: List[str], original_names: List[str], file_path: str) -> None:
    """Write a list of sanitized company names to a file."""
    with open(file_path, 'w') as file:
        for name, original_name in zip(names, original_names):
            file.write(f"{original_name};{name}\n")

def sanitize_all(names: List[str], rules: List[Dict[str, str]]) -> List[str]:
    """Sanitize all company names in a list using a list of sanitization rules."""
    return [sanitize_company_name(name, rules) for name in names]

def remove_duplicates(names: List[str]) -> List[str]:
    """Remove duplicate company names from a list."""
    return list(set(names))

def proper_case(name: str) -> str:
    """Convert a company name to proper case."""
    return name.title()

def smart_title_case(name: str) -> str:
    """
    Smart title case that:
    - Uppercases known legal designators
    - Preserves special cases like 3M, IBM
    - Title cases regular words
    """
    # Known acronyms that should be UPPERCASE
    acronyms = {
        'pty', 'ltd', 'inc', 'llc', 'llp', 'plc', 
        'sa', 'srl', 'gmbh', 'ag', 'pte', 'bhd', 'sdn',
        'nz', 'au', 'uk', 'us', 'eu', 'pte'
    }
    
    # Special cases that need specific casing
    special = {
        '3m': '3M',
        'ibm': 'IBM',
        'hp': 'HP',
        'it': 'IT',
        'ai': 'AI',
        'io': 'IO'
    }
    
    # Words that stay lowercase (except first word)
    lowercase = {'of', 'and', 'the', 'a', 'an', 'for', 'at', 'by', 'to', 'in', 'on'}
    
    words = name.split()
    result = []
    
    for i, word in enumerate(words):
        word_lower = word.lower()
        
        # Special cases (3M, IBM, etc.)
        if word_lower in special:
            result.append(special[word_lower])
        # Known acronyms - UPPERCASE
        elif word_lower in acronyms:
            result.append(word_lower.upper())
        # Small words stay lowercase (except first word)
        elif i > 0 and word_lower in lowercase:
            result.append(word_lower)
        # Default: title case
        else:
            result.append(word.title())
    
    return ' '.join(result)

def classify_with_gliner(names: List[str], batch_size: int = 100, context_strategy: str = "neutral"):
    """
    Classify company names using GLiNER with GPU acceleration.
    
    Args:
        names: List of names to classify
        batch_size: Batch size for processing
        context_strategy: How to add context ("neutral", "explicit", "minimal", "none")
    
    Returns:
        List of (type, confidence) tuples
    """
    from src.sanitization.gliner_classifier import GLiNERClassifier
    
    print(f"  Initializing GLiNER...")
    classifier = GLiNERClassifier(context_strategy=context_strategy)
    
    print(f"  Classifying {len(names)} names...")
    results = classifier.classify_batch(names, batch_size=batch_size)
    
    return results

def clean_all_pandas_df(df, save_location="", rules_loc=None):
    """
    Clean a pandas DataFrame with company names.
    
    Args:
        df: DataFrame with 'Name' column
        save_location: Where to save the results (optional)
        rules_loc: Path to rules JSON file (defaults to data/sanitization_rules.json)
    
    Returns:
        DataFrame with 'Clean_Name' column added
    """
    import pandas as pd
    from src.utils.common import smart_title_case_suffix
    
    if rules_loc is None:
        # Default to the new location
        rules_loc = Path(__file__).parent.parent.parent / "data" / "sanitization_rules.json"
    
    # Load the sanitization rules
    rules = load_sanitization_rules(str(rules_loc))
    
    names = df['Name'].str.lower().tolist()
    original_names = names[:]

    # Sanitize the company names
    names = sanitize_all(names, rules)

    # Convert the company names to proper case
    names = [proper_case(name) for name in names]
    
    # Final step: standardize legal suffixes (PTY LTD → Pty Ltd, etc.)
    names = [smart_title_case_suffix(name) for name in names]

    # Write the sanitized company names to a file if requested
    if save_location:
        write_sanitized_names(names, original_names, save_location)

    df['Clean_Name'] = names

    return df

def clean_all_txt_file(file_loc="", save_location="", rules_loc=None):
    """
    Clean company names from a text file.
    
    Args:
        file_loc: Path to input text file (one name per line)
        save_location: Where to save the results
        rules_loc: Path to rules JSON file (defaults to data/sanitization_rules.json)
    
    Returns:
        List of cleaned names
    """
    from src.utils.common import smart_title_case_suffix
    
    if rules_loc is None:
        # Default to the new location
        rules_loc = Path(__file__).parent.parent.parent / "data" / "sanitization_rules.json"
    
    # Load the sanitization rules
    rules = load_sanitization_rules(str(rules_loc))

    # Load the company names
    with open(file_loc, 'r') as file:
        names = [line.strip().lower() for line in file]
    
    original_names = names[:]

    # Sanitize the company names
    names = sanitize_all(names, rules)

    # Convert the company names to proper case
    names = [proper_case(name) for name in names]
    
    # Final step: standardize legal suffixes (PTY LTD → Pty Ltd, etc.)
    names = [smart_title_case_suffix(name) for name in names]

    # Write the sanitized company names to a file
    write_sanitized_names(names, original_names, save_location)

    return names

def main():
    """Main CLI entry point."""
    import argparse
    import pandas as pd
    from src.utils.common import smart_title_case_suffix
    
    parser = argparse.ArgumentParser(
        description="Simple company name sanitizer using regex rules (lightweight version)"
    )
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--column", default="record_name", help="Column name to sanitize (default: record_name)")
    parser.add_argument(
        "--rules", 
        default=None,
        help="Rules JSON file (default: data/sanitization_rules.json)"
    )
    parser.add_argument("--use_gliner", action="store_true", help="Add GLiNER classification")
    parser.add_argument("--batch_size", type=int, default=100, help="GLiNER batch size (default: 100)")
    parser.add_argument(
        "--context_strategy", 
        default="neutral", 
        choices=["neutral", "explicit", "minimal", "none"],
        help="GLiNER context strategy (default: neutral)"
    )
    
    args = parser.parse_args()
    
    # Default rules location
    if args.rules is None:
        args.rules = Path(__file__).parent.parent.parent / "data" / "sanitization_rules.json"
    
    # Load rules
    print(f"Loading rules from {args.rules}...")
    rules = load_sanitization_rules(str(args.rules))
    print(f"  Loaded {len(rules)} rules\n")
    
    # Load CSV
    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input, dtype=str, encoding="utf-8")
    print(f"  Loaded {len(df):,} rows")
    print(f"  Columns: {', '.join(df.columns)}\n")
    
    if args.column not in df.columns:
        raise ValueError(f"Column '{args.column}' not found. Available: {list(df.columns)}")
    
    # Sanitize with regex rules
    print(f"Sanitizing company names...")
    original_names = df[args.column].fillna("").tolist()
    names = [name.lower() for name in original_names]
    names = sanitize_all(names, rules)
    names = [smart_title_case(name) for name in names]
    # Final step: standardize legal suffixes (PTY LTD → Pty Ltd, etc.)
    names = [smart_title_case_suffix(name) for name in names]
    df['sanitized_name'] = names
    print(f"  ✓ Applied {len(rules)} regex rules + suffix standardization\n")
    
    # Add GLiNER classification if requested
    if args.use_gliner:
        try:
            print(f"\nRunning GLiNER classification (strategy: {args.context_strategy})...")
            results = classify_with_gliner(names, batch_size=args.batch_size, 
                                          context_strategy=args.context_strategy)
            
            df['gliner_type'] = [r[0] for r in results]
            df['gliner_confidence'] = [r[1] for r in results]
            
            avg_confidence = df['gliner_confidence'].mean()
            print(f"  ✓ Classification complete (avg confidence: {avg_confidence:.3f})")
            
            # Show top types
            print(f"\n  Top entity types:")
            for type_name, count in df['gliner_type'].value_counts().head(5).items():
                pct = count / len(df) * 100
                print(f"    {type_name:30s} {count:4d} ({pct:5.1f}%)")
                
        except Exception as e:
            print(f"  ⚠️  GLiNER failed: {e}")
            print(f"     Continuing without classification...\n")
    
    # Save
    print(f"\nSaving to {args.output}...")
    df.to_csv(args.output, index=False, encoding="utf-8")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✓ Complete! Processed {len(df):,} records")
    print(f"  Input:  '{args.column}'")
    print(f"  Output: 'sanitized_name'", end="")
    if args.use_gliner and 'gliner_type' in df.columns:
        print(f", 'gliner_type', 'gliner_confidence'")
    else:
        print()
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

