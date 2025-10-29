#!/usr/bin/env python3
# See docstring for usage details.

# Importing 
import os
import sys
import hashlib
import json

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Now import from the src package
from src.sanitization.gliner_classifier import GLiNERClassifier
from src.storage.db_manager import ProcessingDBManager
from src.utils.common import generate_config_hash

import argparse
from datetime import datetime
import re
import unicodedata
import pandas as pd

def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def ascii_dashes_quotes(s: str) -> str:
    return (
        s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
         .replace("\u00a0"," ")
         .replace("\u201c", '"').replace("\u201d", '"')
         .replace("\u2018", "'").replace("\u2019", "'")
    )

def norm_ws(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"^[\s,.;:/\\&-]+", "", s)
    s = re.sub(r"[\s,.;:/\\&-]+$", "", s)
    return s

def normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return norm_ws(ascii_dashes_quotes(nfkc(s.lower())))

INSTR_RE = re.compile(r"(?:\d{1,2}\.\d{1,3}%|\b(?:class|series)\s+[a-z0-9\-]+|\b(?:adr|american depositary)\b|\b\d{8}\b)")
GOV_RE   = re.compile(r"\b(ministry|department|council|university|hospital|city of|state of)\b")
TRUST_RE = re.compile(r"\b(the trustee for|trust|superannuation|pension fund|fund)\b")
DBA_RE   = re.compile(r"\b(t/a|t\/a|trading as|dba|t-as|t as)\b")

DESIG_PATTERNS = [
    (re.compile(r"\bproprietary\s+limited\b"), "pty ltd"),
    (re.compile(r"\bpty\.?\s*ltd\.?\b"), "pty ltd"),
    (re.compile(r"\binc(?:orporated)?\.?\b"), "inc"),
    (re.compile(r"\blimited\b"), "ltd"),
    (re.compile(r"\bltd\.?\b"), "ltd"),
    (re.compile(r"\bplc\b"), "plc"),
    (re.compile(r"\bgmbh\b"), "gmbh"),
    (re.compile(r"\bs\.?a\.?\b"), "sa"),
    (re.compile(r"\bs\.?r\.?l\.?\b"), "srl"),
    (re.compile(r"\baktiengesellschaft\b|\bag\b"), "ag"),
    (re.compile(r"\bco\.?\b"), "co"),
    (re.compile(r"\bcompany\b"), "co"),
]

STOPWORDS = {
    "holdings","holding","group","company","co","services","service",
    "international","australia","australian","aust","pty","ltd","plc",
    "inc","sa","srl","gmbh","ag","bv","llc","llp","lp","limited",
    "pt","pte","pte.","pte","pte. ltd","pte ltd","sa"
}

GEO_PAREN_RE = re.compile(r"\((?:australia|au|nsw|vic|qld|wa|sa|tas|act|nt|singapore|china|uk|usa|us|canada)\)")

def route_type(s: str) -> str:
    if INSTR_RE.search(s): return "security"
    if TRUST_RE.search(s): return "trust"
    if GOV_RE.search(s): return "government"
    return "company"

def strip_trustee_dba(s: str) -> str:
    s = re.sub(r"^\bthe trustee for\b\s+", "", s)
    s = DBA_RE.sub(" ", s)
    return norm_ws(s)

def canon_designators(s: str) -> str:
    out = s
    for pat, rep in DESIG_PATTERNS:
        out = pat.sub(rep, out)
    return norm_ws(out)

def drop_geo_parentheticals(s: str) -> str:
    return norm_ws(GEO_PAREN_RE.sub("", s))

def display_name_fn(s: str, type_hint: str) -> str:
    base = strip_trustee_dba(s)
    base = drop_geo_parentheticals(base)
    return canon_designators(base)

def canonical_name_fn(s: str) -> str:
    s = canon_designators(s)
    toks = s.split(" ")
    def tc(tok):
        return tok.upper() if tok in {"pty","ltd","plc","sa","ag","llc","llp","bv","gmbh","srl","inc","co"} else tok.title()
    return norm_ws(" ".join(tc(t) for t in toks))

def match_key_strict_fn(s: str) -> str:
    s = canon_designators(s)
    s = re.sub(r"\b(?:pty|ltd|plc|sa|ag|llc|llp|bv|gmbh|srl|inc|co)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return norm_ws(s)

def match_key_aggressive_fn(s: str) -> str:
    s = canon_designators(s)
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = [t for t in s.split() if t not in STOPWORDS]
    return norm_ws(" ".join(toks))

def process_batch(df: pd.DataFrame, gliner_classifier=None) -> pd.DataFrame:
    """Process a batch of records."""
    # Add original record
    df["orig"] = df["record_name"]
    
    # Apply normalization
    df["norm"] = df["record_name"].map(normalize)
    
    # Use rule-based classification
    df["type_hint"] = df["norm"].map(route_type)
    
    # Apply GLiNER classification if available
    if gliner_classifier is not None:
        try:
            df = gliner_classifier.process_dataframe(df, name_column="norm", batch_size=100)
            
            # Use GLiNER classification when confidence is high enough
            confidence_threshold = 0.6
            df["type_hint"] = df.apply(
                lambda row: row["gliner_type"] if row.get("gliner_confidence", 0) >= confidence_threshold else row["type_hint"], 
                axis=1
            )
        except Exception as e:
            print(f"Error in GLiNER classification: {e}")
            # Continue with rule-based classification
    
    # Apply other sanitization steps
    df["display_name"] = df.apply(lambda r: display_name_fn(r["norm"], r["type_hint"]), axis=1)
    df["canonical_name"] = df["display_name"].map(canonical_name_fn)
    df["match_key_strict"] = df["display_name"].map(match_key_strict_fn)
    df["match_key_aggressive"] = df["display_name"].map(match_key_aggressive_fn)
    
    return df

def generate_final_report(db: ProcessingDBManager, report_path: str):
    """Generate a final report with statistics."""
    stats = db.get_processing_stats()
    
    report = f"""# Sanitisation run report
Date: {datetime.now().isoformat(timespec='seconds')}

## Processing Statistics
- Total records processed: {stats['total_processed']}
- Batches: {stats['batch_stats']['total']} total, {stats['batch_stats']['completed']} completed, {stats['batch_stats']['failed']} failed

## Routing (type_hint)
"""
    
    for type_name, count in stats['type_counts'].items():
        report += f"- {type_name}: {count}\n"
    
    report += """
## Columns emitted
- record_name (original)
- display_name (minimally cleaned)
- canonical_name (title-cased with acronym preservation)
- match_key_strict (designators removed; punctuation/spacing normalised)
- match_key_aggressive (stopwords removed; &→and; punctuation removed)
- type_hint (entity type classification)
"""
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"Report written to {report_path}")

def run_with_batches(input_csv: str, output_csv: str, report_path: str, 
                    use_gliner: bool = True, batch_size: int = 5000, 
                    db_path: str = "processing.duckdb", resume: bool = True):
    """Run the sanitization process with batch processing and resumable state."""
    
    # Create configuration hash for tracking
    config = {
        "use_gliner": use_gliner,
        "batch_size": batch_size,
        # Add any other configuration parameters here
    }
    config_hash = generate_config_hash(config)
    
    # Initialize database manager
    db = ProcessingDBManager(db_path)
    
    try:
        # Load the full dataset to get total size
        print(f"Loading dataset from {input_csv}...")
        df_full = pd.read_csv(input_csv, dtype=str, encoding="utf-8", on_bad_lines="skip")
        total_records = len(df_full)
        
        if "record_name" not in df_full.columns:
            raise ValueError("Input file must contain a 'record_name' column.")
        
        # Determine which records need processing
        if resume:
            print("Checking for unprocessed records...")
            unprocessed_indices = db.get_unprocessed_indices(total_records)
            if not unprocessed_indices:
                print("All records have been processed. Nothing to do.")
                # Generate final report and export results
                db.export_results(output_csv)
                generate_final_report(db, report_path)
                return
            
            print(f"Found {len(unprocessed_indices)} unprocessed records.")
        else:
            # Process all records
            unprocessed_indices = list(range(total_records))
            print(f"Processing all {total_records} records from scratch.")
        
        # Initialize GLiNER if needed
        gliner_classifier = None
        if use_gliner:
            try:
                print("Initializing GLiNER classifier...")
                gliner_classifier = GLiNERClassifier()
            except Exception as e:
                print(f"Error initializing GLiNER: {e}")
                print("Continuing without GLiNER classification.")
        
        # Process in batches
        for i in range(0, len(unprocessed_indices), batch_size):
            batch_indices = unprocessed_indices[i:i+batch_size]
            start_idx = min(batch_indices)
            end_idx = max(batch_indices)
            
            print(f"Processing batch {i//batch_size + 1}/{(len(unprocessed_indices) + batch_size - 1)//batch_size}: records {start_idx}-{end_idx}")
            
            # Start a new batch in the database
            batch_id = db.start_batch(start_idx, end_idx, total_records, config_hash)
            
            try:
                # Extract the batch from the full dataset
                df_batch = df_full.iloc[batch_indices].copy()
                
                # Process the batch
                df_batch = process_batch(df_batch, gliner_classifier)
                
                # Store the results
                db.store_batch_results(batch_id, df_batch)
                
                # Mark batch as completed
                db.complete_batch(batch_id)
                
            except Exception as e:
                print(f"Error processing batch {batch_id}: {e}")
                db.fail_batch(batch_id, str(e))
                # Continue with next batch
        
        # Export final results
        print(f"Exporting final results to {output_csv}...")
        records_exported = db.export_results(output_csv)
        print(f"Exported {records_exported} records.")
        
        # Generate final report
        generate_final_report(db, report_path)
        
    finally:
        db.close()

def run(input_csv: str, output_csv: str, report_path: str, use_gliner: bool = True) -> None:
    # Load and preprocess data as before
    df = pd.read_csv(input_csv, dtype=str, encoding="utf-8", on_bad_lines="skip")
    if "record_name" not in df.columns:
        raise ValueError("Input file must contain a 'record_name' column.")
    df = df[["record_name"]].copy()
    df["record_name"] = df["record_name"].fillna("")
    df["orig"] = df["record_name"]

    # Apply normalization as before
    df["norm"] = df["record_name"].map(normalize)
    
    # Use rule-based classification first
    df["type_hint"] = df["norm"].map(route_type)
    
    # If GLiNER is enabled, apply it and combine results
    if use_gliner:
        try:
            print("Initializing GLiNER classifier...")
            classifier = GLiNERClassifier()
            
            print(f"Processing {len(df)} company names with GLiNER...")
            df = classifier.process_dataframe(df, name_column="norm", batch_size=100)
            
            # Use GLiNER classification when confidence is high enough, otherwise keep rule-based
            confidence_threshold = 0.6
            df["type_hint"] = df.apply(
                lambda row: row["gliner_type"] if row.get("gliner_confidence", 0) >= confidence_threshold else row["type_hint"], 
                axis=1
            )
            
            print("GLiNER classification complete.")
        except Exception as e:
            print(f"Error using GLiNER classifier: {e}")
            print("Falling back to rule-based classification only.")
    
    # Continue with other sanitization steps as before
    df["display_name"] = df.apply(lambda r: display_name_fn(r["norm"], r["type_hint"]), axis=1)
    df["canonical_name"] = df["display_name"].map(canonical_name_fn)
    df["match_key_strict"] = df["display_name"].map(match_key_strict_fn)
    df["match_key_aggressive"] = df["display_name"].map(match_key_aggressive_fn)
    
    # Prepare output dataframe
    output_columns = ["orig", "display_name", "canonical_name", "match_key_strict", 
                     "match_key_aggressive", "type_hint"]
    
    # Add GLiNER confidence if available
    if use_gliner and "gliner_confidence" in df.columns:
        output_columns.append("gliner_confidence")
    
    df_out = df[output_columns].rename(columns={"orig": "record_name"})
    df_out.to_csv(output_csv, index=False, encoding="utf-8")
    
    # Generate report with additional GLiNER metrics if available
    n = len(df)
    lengths = df["orig"].str.len()
    charset_non_ascii = df["orig"].map(lambda s: any(ord(ch) > 127 for ch in str(s))).mean()
    counts = df["type_hint"].value_counts().to_dict()
    
    # Add GLiNER stats to report if available
    gliner_stats = ""
    if use_gliner and "gliner_confidence" in df.columns:
        avg_confidence = df["gliner_confidence"].mean()
        high_conf_ratio = (df["gliner_confidence"] >= confidence_threshold).mean()
        gliner_counts = df["gliner_type"].value_counts().to_dict()
        
        gliner_stats = f"""
## GLiNER Classification
- Average confidence: {float(avg_confidence):.4f}
- High confidence ratio: {float(high_conf_ratio):.4%}
- GLiNER classifications:
  - professional_services: {gliner_counts.get('professional_services', 0)}
  - legal_services: {gliner_counts.get('legal_services', 0)}
  - technology: {gliner_counts.get('technology', 0)}
  - telecommunications: {gliner_counts.get('telecommunications', 0)}
  - financial_services: {gliner_counts.get('financial_services', 0)}
  - healthcare: {gliner_counts.get('healthcare', 0)}
  - retail: {gliner_counts.get('retail', 0)}
  - energy: {gliner_counts.get('energy', 0)}
  - industrial: {gliner_counts.get('industrial', 0)}
  - logistics: {gliner_counts.get('logistics', 0)}
  - corporation: {gliner_counts.get('corporation', 0)}
  - government: {gliner_counts.get('government', 0)}
  - educational: {gliner_counts.get('educational', 0)}
  - non_profit: {gliner_counts.get('non_profit', 0)}
"""
    
    report = f"""# Sanitisation run report
Date: {datetime.now().isoformat(timespec='seconds')}

## Input
- Source file: {input_csv}
- Rows: {n}
- Empty or null: {int((df['orig']=='').sum())}
- Unique (orig): {df['orig'].nunique(dropna=False)}
- Lengths (chars): min {int(lengths.min() if n else 0)}, median {float(lengths.median() if n else 0):.1f}, p90 {float(lengths.quantile(0.90) if n else 0):.1f}, max {int(lengths.max() if n else 0)}
- Non-ASCII ratio: {float(charset_non_ascii):.4%}

## Routing (type_hint)
- company: {counts.get('company',0)}
- trust: {counts.get('trust',0)}
- government: {counts.get('government',0)}
- security: {counts.get('security',0)}
{gliner_stats}
## Columns emitted
- record_name (original)
- display_name (minimally cleaned)
- canonical_name (title-cased with acronym preservation)
- match_key_strict (designators removed; punctuation/spacing normalised)
- match_key_aggressive (stopwords removed; &→and; punctuation removed)
- type_hint (company|trust|government|security|professional_services|...)
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Company Name Sanitiser")
    parser.add_argument("--input", required=True, help="Path to input CSV with a `record_name` column")
    parser.add_argument("--output_csv", required=True, help="Path to write cleaned CSV")
    parser.add_argument("--report", required=True, help="Path to write Markdown report")
    parser.add_argument("--use_gliner", action="store_true", help="Use GLiNER for enhanced classification")
    parser.add_argument("--batch_size", type=int, default=5000, help="Batch size for processing")
    parser.add_argument("--db_path", default="processing.duckdb", help="Path to DuckDB database file")
    parser.add_argument("--no_resume", action="store_true", help="Don't resume from previous state, process all records")
    parser.add_argument("--use_batches", action="store_true", help="Use batch processing with DuckDB")
    args = parser.parse_args()
    
    if args.use_batches:
        run_with_batches(
            args.input, 
            args.output_csv, 
            args.report, 
            use_gliner=args.use_gliner,
            batch_size=args.batch_size,
            db_path=args.db_path,
            resume=not args.no_resume
        )
    else:
        run(args.input, args.output_csv, args.report, args.use_gliner)

if __name__ == "__main__":
    main()