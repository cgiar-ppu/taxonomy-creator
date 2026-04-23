"""Data loader for reading and batching input datasets."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from rich.console import Console

from src.config import Config

logger = logging.getLogger(__name__)
console = Console()


class DataLoader:
    """Loads Excel/CSV data and splits it into batches for LLM processing."""

    def __init__(self, config: Config) -> None:
        """Initialize the loader with configuration.

        Args:
            config: Application configuration object.
        """
        self.config = config
        self.df: pd.DataFrame = pd.DataFrame()
        self.total_rows: int = 0

    def load(self, file_path: str | None = None) -> pd.DataFrame:
        """Load data from the configured input file.

        Args:
            file_path: Optional override for the input file path.

        Returns:
            DataFrame with cleaned title and description columns.

        Raises:
            FileNotFoundError: If the input file does not exist.
            KeyError: If required columns are not found.
        """
        path = Path(file_path or self.config.input_file)

        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        logger.info("Loading data from %s", path)

        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path, engine="openpyxl")
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        title_col = self.config.title_column
        desc_col = self.config.description_column

        missing_cols = []
        if title_col not in df.columns:
            missing_cols.append(title_col)
        if desc_col not in df.columns:
            missing_cols.append(desc_col)

        if missing_cols:
            available = ", ".join(df.columns.tolist()[:20])
            raise KeyError(
                f"Required columns not found: {missing_cols}. "
                f"Available columns: {available}"
            )

        # Clean the data
        df = df[[title_col, desc_col]].copy()
        df.columns = ["title", "description"]

        # Strip whitespace
        df["title"] = df["title"].astype(str).str.strip()
        df["description"] = df["description"].astype(str).str.strip()

        # Replace nan-like strings with empty string
        df["title"] = df["title"].replace({"nan": "", "None": "", "NaN": ""})
        df["description"] = df["description"].replace({"nan": "", "None": "", "NaN": ""})

        # Drop rows where both title and description are empty
        df = df[~((df["title"] == "") & (df["description"] == ""))].reset_index(drop=True)

        self.df = df
        self.total_rows = len(df)

        logger.info("Loaded %d rows after cleaning", self.total_rows)
        return df

    def get_batches(self, batch_size: int | None = None) -> List[List[Dict[str, Any]]]:
        """Split loaded data into batches for LLM processing.

        Args:
            batch_size: Optional override for batch size.

        Returns:
            List of batches, where each batch is a list of dicts with
            keys: id, title, description.
        """
        if self.df.empty:
            raise RuntimeError("No data loaded. Call load() first.")

        size = batch_size or self.config.batch_size
        batches: List[List[Dict[str, Any]]] = []

        for start in range(0, self.total_rows, size):
            end = min(start + size, self.total_rows)
            batch = []
            for idx in range(start, end):
                row = self.df.iloc[idx]
                batch.append({
                    "id": idx + 1,  # 1-based ID
                    "title": row["title"],
                    "description": row["description"],
                })
            batches.append(batch)

        logger.info("Created %d batches of up to %d rows each", len(batches), size)
        return batches

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the loaded data.

        Returns:
            Dictionary with row count, average text lengths, and batch info.
        """
        if self.df.empty:
            return {"total_rows": 0}

        title_lengths = self.df["title"].str.len()
        desc_lengths = self.df["description"].str.len()

        total_chars = title_lengths.sum() + desc_lengths.sum()
        avg_chars_per_row = total_chars / self.total_rows if self.total_rows > 0 else 0

        batch_size = self.config.batch_size
        n_batches = (self.total_rows + batch_size - 1) // batch_size

        return {
            "total_rows": self.total_rows,
            "avg_title_length": round(title_lengths.mean(), 1),
            "avg_description_length": round(desc_lengths.mean(), 1),
            "total_characters": int(total_chars),
            "avg_chars_per_row": round(avg_chars_per_row, 1),
            "batch_size": batch_size,
            "total_batches": n_batches,
            "estimated_tokens": int(total_chars / 4),  # rough estimate
        }

    def format_batch_text(self, batch: List[Dict[str, Any]]) -> str:
        """Format a batch of results into text for prompt injection.

        Args:
            batch: List of dicts with id, title, description.

        Returns:
            Formatted text block for insertion into a prompt.
        """
        lines = []
        for item in batch:
            title = str(item.get("title", "")).strip()
            description = str(item.get("description", "")).strip()
            if description in ("nan", "None", "NaN", ""):
                description = "(No description provided)"
            lines.append(f"[Result {item['id']}]")
            lines.append(f"Title: {title}")
            lines.append(f"Description: {description}")
            lines.append("")
        return "\n".join(lines)
