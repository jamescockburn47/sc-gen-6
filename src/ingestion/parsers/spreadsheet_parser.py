"""Spreadsheet parser for Excel files (.xlsx, .xls)."""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.ingestion.parsers.base_parser import BaseParser
from src.schema import DocumentType, ParsedDocument


class SpreadsheetParser(BaseParser):
    """Parser for Excel spreadsheets using pandas/openpyxl.

    Extracts sheet names, first rows (headers), and converts data
    to text blocks. Useful for disclosure documents with tabular data.
    """

    def can_parse(self, file_path: str | Path) -> bool:
        """Check if file is a spreadsheet."""
        suffix = Path(file_path).suffix.lower()
        return suffix in [".xlsx", ".xls", ".xlsm"]

    def parse(
        self, file_path: str | Path, document_type: Optional[DocumentType] = None
    ) -> ParsedDocument:
        """Parse spreadsheet file.

        Args:
            file_path: Path to spreadsheet file
            document_type: Optional document type override

        Returns:
            ParsedDocument with sheet data as text blocks

        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If spreadsheet parsing fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Spreadsheet file not found: {file_path}")

        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names

            text_parts = []
            paragraphs = []
            full_text = ""
            para_start = 0

            for sheet_name in sheet_names:
                # Read sheet
                df = pd.read_excel(excel_file, sheet_name=sheet_name)

                # Convert to text representation
                sheet_text = self._dataframe_to_text(df, sheet_name)
                if sheet_text.strip():
                    text_parts.append(sheet_text)
                    full_text += sheet_text + "\n\n"

                    # Create paragraph entry
                    para_end = para_start + len(sheet_text)
                    paragraphs.append(
                        {
                            "text": sheet_text,
                            "page": 1,
                            "paragraph": len(paragraphs) + 1,
                            "sheet_name": sheet_name,
                            "row_count": len(df),
                            "column_count": len(df.columns),
                            "char_start": para_start,
                            "char_end": para_end,
                        }
                    )
                    para_start = para_end + 2

            excel_file.close()

            if not full_text.strip():
                raise ValueError(f"Spreadsheet appears to be empty: {file_path}")

            # Default to disclosure for spreadsheets
            if document_type is None:
                document_type = "disclosure"

            metadata = {
                "sheet_names": sheet_names,
                "sheet_count": len(sheet_names),
                "total_rows": sum(p.get("row_count", 0) for p in paragraphs),
            }

            return ParsedDocument(
                file_path=str(file_path),
                file_name=file_path.name,
                document_type=document_type,
                text=full_text,
                pages=[1] * len(paragraphs),
                paragraphs=paragraphs,
                metadata=metadata,
            )

        except Exception as e:
            raise Exception(f"Failed to parse spreadsheet {file_path}: {str(e)}") from e

    def _dataframe_to_text(self, df: pd.DataFrame, sheet_name: str) -> str:
        """Convert DataFrame to text representation.

        Includes sheet name, headers (first row), and sample data.

        Args:
            df: DataFrame to convert
            sheet_name: Name of the sheet

        Returns:
            Text representation
        """
        lines = [f"Sheet: {sheet_name}", "=" * 50]

        if df.empty:
            return "\n".join(lines) + "\n(Empty sheet)\n"

        # Headers
        headers = " | ".join(str(col) for col in df.columns)
        lines.append(f"Headers: {headers}")

        # First few rows (up to 10)
        sample_rows = min(10, len(df))
        lines.append(f"\nFirst {sample_rows} rows:")
        for idx, row in df.head(sample_rows).iterrows():
            row_text = " | ".join(str(val) if pd.notna(val) else "" for val in row)
            lines.append(f"  Row {idx + 1}: {row_text}")

        if len(df) > sample_rows:
            lines.append(f"\n... ({len(df) - sample_rows} more rows)")

        return "\n".join(lines)




