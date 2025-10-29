#!/usr/bin/env python3
import os
import sys
import argparse

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.storage.db_manager import ProcessingDBManager

def list_failed_batches(db_path):
    """List all failed batches in the database."""
    db = ProcessingDBManager(db_path)
    failed_batches = db.get_failed_batches()
    
    if not failed_batches:
        print("No failed batches found.")
        return
    
    print(f"Found {len(failed_batches)} failed batches:")
    for batch in failed_batches:
        print(f"Batch {batch['batch_id']}: Records {batch['start_index']}-{batch['end_index']}, "
              f"Started: {batch['started_at']}, Failed: {batch['completed_at']}")
    
    db.close()

def reset_failed_batches(db_path):
    """Reset all failed batches so they can be processed again."""
    db = ProcessingDBManager(db_path)
    
    # Get failed batch IDs
    failed_batches = db.get_failed_batches()
    
    if not failed_batches:
        print("No failed batches found.")
        db.close()
        return
    
    # Delete records from failed batches
    for batch in failed_batches:
        batch_id = batch['batch_id']
        db.conn.execute("DELETE FROM processed_records WHERE batch_id = ?", [batch_id])
        db.conn.execute("DELETE FROM processing_batches WHERE batch_id = ?", [batch_id])
        print(f"Reset batch {batch_id}")
    
    print(f"Reset {len(failed_batches)} failed batches.")
    db.close()

def show_stats(db_path):
    """Show processing statistics."""
    db = ProcessingDBManager(db_path)
    stats = db.get_processing_stats()
    
    print("Processing Statistics:")
    print(f"- Total records processed: {stats['total_processed']}")
    print(f"- Batches: {stats['batch_stats']['total']} total, "
          f"{stats['batch_stats']['completed']} completed, "
          f"{stats['batch_stats']['failed']} failed, "
          f"{stats['batch_stats']['in_progress']} in progress")
    
    print("\nRecord types:")
    for type_name, count in stats['type_counts'].items():
        print(f"- {type_name}: {count}")
    
    db.close()

def reset_database(db_path):
    """Reset the entire database."""
    if os.path.exists(db_path):
        # Confirm with user
        confirm = input(f"Are you sure you want to reset the database at {db_path}? This will delete all data. (y/N): ")
        if confirm.lower() == 'y':
            os.remove(db_path)
            print(f"Database at {db_path} has been reset.")
            
            # Initialize a new empty database
            db = ProcessingDBManager(db_path)
            db.close()
            print("New empty database initialized.")
        else:
            print("Database reset cancelled.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DuckDB Database Utilities")
    parser.add_argument("--db_path", default="processing.duckdb", help="Path to DuckDB database file")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List failed batches command
    list_parser = subparsers.add_parser("list-failed", help="List failed batches")
    
    # Reset failed batches command
    reset_parser = subparsers.add_parser("reset-failed", help="Reset failed batches")
    
    # Show stats command
    stats_parser = subparsers.add_parser("stats", help="Show processing statistics")
    
    # Reset database command
    reset_db_parser = subparsers.add_parser("reset-db", help="Reset the entire database")
    
    args = parser.parse_args()
    
    if args.command == "list-failed":
        list_failed_batches(args.db_path)
    elif args.command == "reset-failed":
        reset_failed_batches(args.db_path)
    elif args.command == "stats":
        show_stats(args.db_path)
    elif args.command == "reset-db":
        reset_database(args.db_path)
    else:
        parser.print_help()
