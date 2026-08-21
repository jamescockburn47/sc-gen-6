"""Split Enron email CSV into individual .txt files for SC Gen 6 ingestion.

Usage:
    python split_enron_csv.py <csv_path> <output_dir>
    
Example:
    python split_enron_csv.py "C:\\Downloads\\enron_emails.csv" "data\\enron_workspace\\documents"
"""

import csv
import hashlib
import re
import sys
from pathlib import Path
from datetime import datetime

# Increase CSV field size limit for large email bodies
csv.field_size_limit(10 * 1024 * 1024)  # 10 MB


def sanitize_filename(text: str, max_len: int = 50) -> str:
    """Create a safe filename from text."""
    # Remove or replace invalid characters
    text = re.sub(r'[<>:"/\\|?*\n\r\t]', '', text)
    text = text.strip()
    # Truncate
    if len(text) > max_len:
        text = text[:max_len]
    return text or "untitled"


def parse_date(date_str: str) -> str:
    """Try to parse date string into a standard format."""
    if not date_str:
        return "unknown_date"
    
    # Common Enron date formats
    formats = [
        "%a, %d %b %Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%d-%b-%Y",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip()[:25], fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    return "unknown_date"


def split_enron_csv(csv_path: str, output_dir: str, limit: int = None):
    """Split Enron CSV into individual email files.
    
    Expected CSV columns (typical Enron dataset):
        - file, message, date, from, to, subject, body (or similar)
    
    Args:
        csv_path: Path to the Enron CSV file
        output_dir: Directory to write individual email files
        limit: Optional limit on number of emails to process
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    
    if not csv_path.exists():
        print(f"[ERROR] CSV file not found: {csv_path}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Reading: {csv_path}")
    print(f"[INFO] Output: {output_dir}")
    
    # Track stats
    processed = 0
    skipped = 0
    errors = 0
    
    # Open with different encodings if needed
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(csv_path, 'r', encoding=encoding, errors='replace') as f:
                # Detect CSV structure by reading header
                reader = csv.DictReader(f)
                fields = reader.fieldnames
                
                if not fields:
                    print(f"[ERROR] No headers found in CSV")
                    return
                
                print(f"[INFO] CSV columns: {fields}")
                
                # Map common column names
                # Enron datasets vary in structure
                body_col = None
                subject_col = None
                from_col = None
                to_col = None
                date_col = None
                
                for col in fields:
                    col_lower = col.lower()
                    if 'body' in col_lower or 'message' in col_lower or 'content' in col_lower:
                        body_col = col
                    elif 'subject' in col_lower:
                        subject_col = col
                    elif col_lower in ('from', 'sender', 'from_'):
                        from_col = col
                    elif col_lower in ('to', 'recipient', 'to_'):
                        to_col = col
                    elif 'date' in col_lower:
                        date_col = col
                
                if not body_col:
                    # Try first large text column
                    body_col = fields[-1]  # Often the last column
                    print(f"[WARN] No body column detected, using: {body_col}")
                
                print(f"[INFO] Using columns: body={body_col}, subject={subject_col}, from={from_col}, date={date_col}")
                
                for row in reader:
                    if limit and processed >= limit:
                        print(f"[INFO] Reached limit of {limit} emails")
                        break
                    
                    try:
                        # Extract fields
                        body = row.get(body_col, '') or ''
                        subject = row.get(subject_col, '') or 'No Subject'
                        sender = row.get(from_col, '') or 'unknown'
                        recipient = row.get(to_col, '') or 'unknown'
                        date_str = row.get(date_col, '') or ''
                        
                        # Skip empty emails
                        if not body.strip():
                            skipped += 1
                            continue
                        
                        # Parse date
                        date_parsed = parse_date(date_str)
                        
                        # Create filename: date_from_subject_hash.txt
                        sender_name = sender.split('@')[0] if '@' in sender else sender
                        subject_safe = sanitize_filename(subject, 30)
                        content_hash = hashlib.md5(body.encode()).hexdigest()[:8]
                        
                        filename = f"{date_parsed}_{sanitize_filename(sender_name, 15)}_{subject_safe}_{content_hash}.txt"
                        filepath = output_dir / filename
                        
                        # Format email content
                        email_text = f"""From: {sender}
To: {recipient}
Date: {date_str}
Subject: {subject}

{body}
"""
                        
                        # Write file
                        with open(filepath, 'w', encoding='utf-8', errors='replace') as out:
                            out.write(email_text)
                        
                        processed += 1
                        
                        if processed % 1000 == 0:
                            print(f"[PROGRESS] Processed {processed:,} emails...")
                    
                    except Exception as e:
                        errors += 1
                        if errors <= 5:
                            print(f"[ERROR] Row error: {e}")
                
                break  # Success, don't try other encodings
                
        except UnicodeDecodeError:
            continue
    
    print(f"\n[DONE] Split complete:")
    print(f"  - Processed: {processed:,}")
    print(f"  - Skipped (empty): {skipped:,}")
    print(f"  - Errors: {errors:,}")
    print(f"  - Output dir: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nUsage: python split_enron_csv.py <csv_path> <output_dir> [limit]")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_dir = sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    split_enron_csv(csv_path, output_dir, limit)
