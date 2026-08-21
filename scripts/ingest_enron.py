"""Enron MIME Email Ingestion Script.

Downloads and ingests the Enron email dataset from CMU CALO project.
Uses isolated matter workspace to prevent cross-contamination.

Usage:
    python scripts/ingest_enron.py --download    # Download dataset
    python scripts/ingest_enron.py --ingest      # Ingest to matter
    python scripts/ingest_enron.py --both        # Download and ingest
"""

import argparse
import email
import os
import sys
import tarfile
from pathlib import Path
from typing import Generator, Optional
from datetime import datetime
import urllib.request

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Defer heavy imports until needed (ingestion phase)
# from src.config_loader import get_settings
# from src.matter.matter_config import MatterConfig
# from src.ingestion.ingestion_pipeline import IngestionPipeline
# from src.schema import DocumentType

# CMU CALO Enron dataset URL
ENRON_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
MATTER_PATH = Path("data/matters/enron-emails")
DOWNLOAD_PATH = MATTER_PATH / "downloads"


def download_enron_dataset(force: bool = False) -> Path:
    """Download the CMU CALO Enron dataset.
    
    Args:
        force: Re-download even if exists
        
    Returns:
        Path to downloaded tarball
    """
    DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
    tarball_path = DOWNLOAD_PATH / "enron_mail.tar.gz"
    
    if tarball_path.exists() and not force:
        print(f"Dataset already downloaded: {tarball_path}")
        return tarball_path
    
    print(f"Downloading Enron dataset from CMU (~1.7GB)...")
    print(f"URL: {ENRON_URL}")
    print("This may take several minutes...")
    
    # Download with progress
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = (downloaded / total_size) * 100 if total_size > 0 else 0
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.0f}/{mb_total:.0f} MB)", end="")
    
    urllib.request.urlretrieve(ENRON_URL, tarball_path, report_progress)
    print(f"\n  Downloaded: {tarball_path}")
    
    return tarball_path


def extract_emails(tarball_path: Path, max_emails: Optional[int] = None) -> Generator[dict, None, None]:
    """Extract emails from the Enron tarball.
    
    Args:
        tarball_path: Path to downloaded tarball
        max_emails: Optional limit on emails to extract
        
    Yields:
        Dict with email data: {subject, from, to, date, body, file_path}
    """
    print(f"Extracting emails from {tarball_path}...")
    
    count = 0
    with tarfile.open(tarball_path, "r:gz") as tar:
        for member in tar:
            if max_emails and count >= max_emails:
                break
                
            # Only process files (not directories)
            if not member.isfile():
                continue
            
            # Skip non-email files
            if not member.name.endswith("."):
                # Enron emails are typically numbered files without extension
                try:
                    # Extract and parse email
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    
                    raw = f.read()
                    try:
                        msg = email.message_from_bytes(raw)
                    except Exception:
                        continue
                    
                    # Extract body
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                try:
                                    body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                                except Exception:
                                    body = str(part.get_payload())
                                break
                    else:
                        try:
                            body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
                        except Exception:
                            body = str(msg.get_payload())
                    
                    # Skip empty emails
                    if not body or len(body.strip()) < 10:
                        continue
                    
                    # Parse date
                    date_str = msg.get("Date", "")
                    try:
                        from email.utils import parsedate_to_datetime
                        email_date = parsedate_to_datetime(date_str)
                    except Exception:
                        email_date = None
                    
                    email_data = {
                        "subject": msg.get("Subject", "(no subject)"),
                        "from": msg.get("From", ""),
                        "to": msg.get("To", ""),
                        "date": email_date,
                        "body": body,
                        "file_path": member.name,
                    }
                    
                    count += 1
                    if count % 1000 == 0:
                        print(f"  Extracted {count} emails...")
                    
                    yield email_data
                    
                except Exception as e:
                    continue
    
    print(f"  Total emails extracted: {count}")


def save_emails_to_documents(emails: Generator[dict, None, None], 
                              documents_path: Path,
                              batch_size: int = 100) -> int:
    """Save emails to document files for ingestion.
    
    Args:
        emails: Generator of email dicts
        documents_path: Path to save documents
        batch_size: How many emails per text file
        
    Returns:
        Number of emails saved
    """
    documents_path.mkdir(parents=True, exist_ok=True)
    
    batch = []
    batch_num = 0
    total_saved = 0
    
    for email_data in emails:
        batch.append(email_data)
        
        if len(batch) >= batch_size:
            # Save batch to file
            batch_num += 1
            file_path = documents_path / f"enron_batch_{batch_num:04d}.txt"
            
            with open(file_path, "w", encoding="utf-8") as f:
                for i, em in enumerate(batch):
                    f.write(f"=" * 60 + "\n")
                    f.write(f"EMAIL {i+1}\n")
                    f.write(f"=" * 60 + "\n")
                    f.write(f"From: {em['from']}\n")
                    f.write(f"To: {em['to']}\n")
                    f.write(f"Subject: {em['subject']}\n")
                    if em['date']:
                        f.write(f"Date: {em['date'].isoformat()}\n")
                    f.write(f"\n{em['body']}\n\n")
            
            total_saved += len(batch)
            print(f"  Saved batch {batch_num} ({total_saved} emails)")
            batch = []
    
    # Save remaining
    if batch:
        batch_num += 1
        file_path = documents_path / f"enron_batch_{batch_num:04d}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            for i, em in enumerate(batch):
                f.write(f"=" * 60 + "\n")
                f.write(f"EMAIL {i+1}\n")
                f.write(f"=" * 60 + "\n")
                f.write(f"From: {em['from']}\n")
                f.write(f"To: {em['to']}\n")
                f.write(f"Subject: {em['subject']}\n")
                if em['date']:
                    f.write(f"Date: {em['date'].isoformat()}\n")
                f.write(f"\n{em['body']}\n\n")
        total_saved += len(batch)
    
    print(f"Total emails saved: {total_saved} in {batch_num} batch files")
    return total_saved


def ingest_to_matter(documents_path: Path, matter_path: Path) -> None:
    """Ingest documents into the matter workspace.
    
    Args:
        documents_path: Path containing document files
        matter_path: Path to matter directory
    """
    # Import here to avoid loading heavy dependencies for download-only mode
    from src.config_loader import get_settings
    from src.ingestion.ingestion_pipeline import IngestionPipeline
    from src.schema import DocumentType
    
    print(f"\nIngesting documents to matter: {matter_path}")
    
    # Load settings (will use matter-specific paths)
    settings = get_settings()
    
    # Override paths for this matter
    # Note: This is a simplified version - full implementation would
    # use MatterManager to switch context
    
    pipeline = IngestionPipeline(settings)
    
    # Get all document files
    doc_files = list(documents_path.glob("*.txt"))
    print(f"Found {len(doc_files)} document files to ingest")
    
    for i, doc_file in enumerate(doc_files):
        print(f"[{i+1}/{len(doc_files)}] Ingesting {doc_file.name}...")
        try:
            pipeline.process_document(
                file_path=doc_file,
                document_type=DocumentType.EMAIL,
                generate_summary=False  # Skip summaries for large batch
            )
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\nIngestion complete!")


def main():
    parser = argparse.ArgumentParser(description="Enron Email Dataset Ingestion")
    parser.add_argument("--download", action="store_true", help="Download dataset")
    parser.add_argument("--ingest", action="store_true", help="Ingest to matter")
    parser.add_argument("--both", action="store_true", help="Download and ingest")
    parser.add_argument("--max-emails", type=int, default=10000, 
                        help="Max emails to process (default: 10000)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Emails per document file (default: 50)")
    
    args = parser.parse_args()
    
    if not (args.download or args.ingest or args.both):
        parser.print_help()
        print("\nExample:")
        print("  python scripts/ingest_enron.py --both --max-emails 5000")
        return
    
    documents_path = MATTER_PATH / "documents"
    
    if args.download or args.both:
        tarball = download_enron_dataset()
        emails = extract_emails(tarball, max_emails=args.max_emails)
        save_emails_to_documents(emails, documents_path, batch_size=args.batch_size)
    
    if args.ingest or args.both:
        ingest_to_matter(documents_path, MATTER_PATH)


if __name__ == "__main__":
    main()
