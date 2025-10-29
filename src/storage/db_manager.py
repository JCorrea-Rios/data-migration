#!/usr/bin/env python3
import duckdb
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

class ProcessingDBManager:
    """Manages DuckDB database for storing batch processing results and logs."""
    
    def __init__(self, db_path: str = "processing.duckdb"):
        """Initialize the database manager with the given path."""
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._initialize_tables()
        
    def _initialize_tables(self):
        """Create necessary tables if they don't exist."""
        # Drop existing table if it exists to avoid schema mismatches
        self.conn.execute("DROP TABLE IF EXISTS processed_records")
        
        # Table for processed records
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_records (
                record_id INTEGER,
                record_name VARCHAR,
                orig VARCHAR,
                norm VARCHAR,
                display_name VARCHAR,
                canonical_name VARCHAR,
                match_key_strict VARCHAR,
                match_key_aggressive VARCHAR,
                type_hint VARCHAR,
                gliner_type VARCHAR,
                gliner_confidence DOUBLE,
                batch_id INTEGER,
                processed_at TIMESTAMP
            )
        """)
        
        # Table for processing batches
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS processing_batches (
                batch_id INTEGER PRIMARY KEY,
                start_index INTEGER,
                end_index INTEGER,
                total_records INTEGER,
                config_hash VARCHAR,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                status VARCHAR
            )
        """)
        
        # Table for processing logs
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS processing_logs (
                log_id INTEGER PRIMARY KEY,
                batch_id INTEGER,
                log_level VARCHAR,
                message VARCHAR,
                timestamp TIMESTAMP
            )
        """)
        
        # Create indexes for faster queries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_record_name ON processed_records(record_name)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_batch_id ON processed_records(batch_id)")
        
    def get_next_batch_id(self) -> int:
        """Get the next available batch ID."""
        result = self.conn.execute("SELECT COALESCE(MAX(batch_id), 0) + 1 FROM processing_batches").fetchone()
        return result[0]
        
    def start_batch(self, start_index: int, end_index: int, total_records: int, config_hash: str) -> int:
        """Record the start of a new processing batch."""
        batch_id = self.get_next_batch_id()
        self.conn.execute("""
            INSERT INTO processing_batches (
                batch_id, start_index, end_index, total_records, config_hash, started_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [batch_id, start_index, end_index, total_records, config_hash, datetime.now(), "STARTED"])
        
        self.log(batch_id, "INFO", f"Started batch {batch_id} processing records {start_index}-{end_index}")
        return batch_id
        
    def complete_batch(self, batch_id: int):
        """Mark a batch as completed."""
        self.conn.execute("""
            UPDATE processing_batches 
            SET completed_at = ?, status = ? 
            WHERE batch_id = ?
        """, [datetime.now(), "COMPLETED", batch_id])
        
        self.log(batch_id, "INFO", f"Completed batch {batch_id}")
        
    def fail_batch(self, batch_id: int, error_message: str):
        """Mark a batch as failed."""
        self.conn.execute("""
            UPDATE processing_batches 
            SET completed_at = ?, status = ? 
            WHERE batch_id = ?
        """, [datetime.now(), "FAILED", batch_id])
        
        self.log(batch_id, "ERROR", f"Batch {batch_id} failed: {error_message}")
        
    def store_batch_results(self, batch_id: int, results_df: pd.DataFrame):
        """Store the results of a processing batch."""
        # Add metadata columns
        results_df['batch_id'] = batch_id
        results_df['processed_at'] = datetime.now()
        
        # Add record_id if not present
        if 'record_id' not in results_df.columns:
            # Get the current max record_id
            result = self.conn.execute("SELECT COALESCE(MAX(record_id), 0) FROM processed_records").fetchone()
            start_id = result[0] + 1
            results_df['record_id'] = range(start_id, start_id + len(results_df))
        
        # Convert DataFrame to DuckDB table
        self.conn.register('temp_results', results_df)
        
        # Get column names from the table
        columns = self.conn.execute("PRAGMA table_info(processed_records)").fetchall()
        column_names = [col[1] for col in columns]
        
        # Get column names from the DataFrame that match the table
        matching_columns = [col for col in results_df.columns if col in column_names]
        
        # Insert into processed_records table with explicit column names
        columns_str = ", ".join(matching_columns)
        self.conn.execute(f"""
            INSERT INTO processed_records ({columns_str})
            SELECT {columns_str} FROM temp_results
        """)
        
        self.log(batch_id, "INFO", f"Stored {len(results_df)} records for batch {batch_id}")
        
    def log(self, batch_id: int, log_level: str, message: str):
        """Add a log entry."""
        next_id = self.conn.execute("SELECT COALESCE(MAX(log_id), 0) + 1 FROM processing_logs").fetchone()[0]
        
        self.conn.execute("""
            INSERT INTO processing_logs (log_id, batch_id, log_level, message, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, [next_id, batch_id, log_level, message, datetime.now()])
        
    def get_unprocessed_indices(self, total_records: int) -> List[int]:
        """Get indices of records that haven't been processed yet."""
        # Get all processed record indices
        processed_indices = self.conn.execute("""
            SELECT DISTINCT record_id FROM processed_records
        """).fetchall()
        
        processed_set = set([row[0] for row in processed_indices])
        all_indices = set(range(1, total_records + 1))
        
        return list(all_indices - processed_set)
        
    def get_failed_batches(self) -> List[Dict[str, Any]]:
        """Get information about failed batches."""
        results = self.conn.execute("""
            SELECT batch_id, start_index, end_index, started_at, completed_at
            FROM processing_batches
            WHERE status = 'FAILED'
            ORDER BY batch_id
        """).fetchall()
        
        return [
            {
                'batch_id': row[0],
                'start_index': row[1],
                'end_index': row[2],
                'started_at': row[3],
                'completed_at': row[4]
            }
            for row in results
        ]
        
    def export_results(self, output_path: str, include_batch_info: bool = False):
        """Export all processed records to a CSV file."""
        # Get all column names from the table
        columns = self.conn.execute("PRAGMA table_info(processed_records)").fetchall()
        column_names = [col[1] for col in columns]
        
        # Filter out internal columns unless include_batch_info is True
        export_columns = [
            "record_name", "display_name", "canonical_name", 
            "match_key_strict", "match_key_aggressive", "type_hint"
        ]
        
        # Add gliner columns if they exist
        if "gliner_type" in column_names:
            export_columns.append("gliner_type")
        if "gliner_confidence" in column_names:
            export_columns.append("gliner_confidence")
            
        if include_batch_info:
            export_columns.extend(["batch_id", "processed_at"])
            
        # Ensure all columns exist in the table
        export_columns = [col for col in export_columns if col in column_names]
        
        # Build the query
        columns_str = ", ".join(export_columns)
        query = f"SELECT {columns_str} FROM processed_records ORDER BY record_id"
        
        # Execute query and export to CSV
        self.conn.execute(f"COPY ({query}) TO '{output_path}' (HEADER, DELIMITER ',')")
        
        return self.conn.execute("SELECT COUNT(*) FROM processed_records").fetchone()[0]
        
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about the processing."""
        stats = {}
        
        # Total records processed
        stats['total_processed'] = self.conn.execute(
            "SELECT COUNT(*) FROM processed_records"
        ).fetchone()[0]
        
        # Records by type_hint
        type_counts = self.conn.execute("""
            SELECT type_hint, COUNT(*) 
            FROM processed_records 
            GROUP BY type_hint
            ORDER BY COUNT(*) DESC
        """).fetchall()
        
        stats['type_counts'] = {row[0]: row[1] for row in type_counts}
        
        # Batch statistics
        batch_stats = self.conn.execute("""
            SELECT 
                COUNT(*) as total_batches,
                SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) as completed_batches,
                SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed_batches,
                SUM(CASE WHEN status = 'STARTED' THEN 1 ELSE 0 END) as in_progress_batches
            FROM processing_batches
        """).fetchone()
        
        stats['batch_stats'] = {
            'total': batch_stats[0],
            'completed': batch_stats[1],
            'failed': batch_stats[2],
            'in_progress': batch_stats[3]
        }
        
        return stats
        
    def close(self):
        """Close the database connection."""
        self.conn.close()
