"""Input normalization and validation for mean-substituted growth datasets.

Format A: time, individual_1, individual_2, ... (wide data)
Format B: time, mean_weight, n (time-point means and replicate counts)

The parser intentionally preserves replicate counts, rather than assuming a
balanced experimental design. Fitting is implemented in a later stage.
"""

from dataclasses import dataclass
from io import BytesIO, StringIO
import csv
import math
import re

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """An input table cannot be interpreted as a valid growth dataset."""


@dataclass
class GrowthData:
    format_name: str
    raw: pd.DataFrame
    mean: pd.DataFrame
    long: pd.DataFrame | None
    is_balanced: bool | None
    warnings: list[str]


_FORMAT_LABELS = {"A": "A", "B": "B", "individual": "A", "mean": "B"}


def read_excel(uploaded_file, sheet_name=0) -> pd.DataFrame:
    """Read XLSX using python-calamine via pandas."""
    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        return pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="calamine", dtype=object)
    except Exception as exc:
        raise DataValidationError(f"Excel tidak dapat dibaca: {exc}") from exc


def read_pasted(text: str, separator: str = "auto") -> pd.DataFrame:
    """Read a table copied from Excel (TSV) or a semicolon/comma CSV.

    Separator detection uses the header line to avoid treating decimal commas
    as column separators in tab/semicolon-delimited tables.
    """
    if not text or not text.strip():
        raise DataValidationError("Textarea kosong. Paste tabel beserta header kolom.")

    lines = [line for line in text.strip().splitlines() if line.strip()]
    header = lines[0]
    if separator == "auto":
        separator = "\t" if "\t" in header else ";" if ";" in header else ","
    elif separator in {"tab", "tsv"}:
        separator = "\t"
    elif separator in {"semicolon", "titik koma"}:
        separator = ";"
    elif separator in {"comma", "koma"}:
        separator = ","
    if separator not in {"\t", ";", ","}:
        raise DataValidationError("Pemisah hanya boleh auto, tab, titik koma, atau koma.")

    try:
        parsed = list(csv.reader(StringIO("\n".join(lines)), delimiter=separator))
    except csv.Error as exc:
        raise DataValidationError(f"Format teks tidak valid: {exc}") from exc
    if len(parsed) < 2:
        raise DataValidationError("Tabel harus berisi header dan minimal satu baris data.")
    expected = len(parsed[0])
    if expected < 2:
        raise DataValidationError("Minimal diperlukan dua kolom. Pastikan pemisah benar.")
    for i, row in enumerate(parsed[1:], start=2):
        if len(row) != expected:
            raise DataValidationError(
                f"Baris {i} memiliki {len(row)} kolom; seharusnya {expected}. "
                "Periksa delimiter dan angka desimal."
            )
    return pd.DataFrame(parsed[1:], columns=parsed[0])


def _numeric(series: pd.Series, label: str, allow_missing: bool) -> pd.Series:
    """Convert either 12.5 or 12,5; reject ambiguous thousands separators."""
    def convert(value):
        if pd.isna(value) or str(value).strip() == "":
            return np.nan
        if isinstance(value, (int, float, np.number)):
            return float(value)
        v = str(value).strip().replace("\u00a0", "")
        if not re.fullmatch(r"[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:[eE][+-]?\d+)?", v):
            raise DataValidationError(
                f"Nilai '{v}' pada kolom '{label}' bukan angka yang didukung. "
                "Gunakan angka tanpa pemisah ribuan."
            )
        return float(v.replace(",", "."))

    values = series.map(convert).astype(float)
    if not allow_missing and values.isna().any():
        raise DataValidationError(f"Kolom '{label}' memiliki nilai kosong.")
    if not np.isfinite(values.dropna()).all():
        raise DataValidationError(f"Kolom '{label}' memiliki angka tidak hingga.")
    return values


def normalize(frame: pd.DataFrame, format_name: str) -> GrowthData:
    """Validate and convert input to a standard representation."""
    kind = _FORMAT_LABELS.get(format_name)
    if kind is None:
        raise DataValidationError("Format harus A (individu) atau B (rata-rata).")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataValidationError("Dataset kosong.")

    df = frame.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if len(set(df.columns)) != len(df.columns):
        raise DataValidationError("Nama kolom harus unik.")
    if any(not c or c.lower().startswith("unnamed:") for c in df.columns):
        raise DataValidationError("Ada header kolom kosong. Perbaiki baris pertama Excel.")

    # Prefer the first column as time, with an explicit name for clear templates.
    if df.columns[0].lower() not in {"time", "waktu", "t"}:
        raise DataValidationError("Kolom pertama harus bernama 'time' (atau 'waktu' / 't').")
    df = df.rename(columns={df.columns[0]: "time"})
    df["time"] = _numeric(df["time"], "time", allow_missing=False)
    if df["time"].duplicated().any():
        raise DataValidationError("Setiap waktu hanya boleh muncul sekali.")
    if not df["time"].is_monotonic_increasing:
        raise DataValidationError("Waktu harus diurutkan menaik (dari awal ke akhir).")
    if len(df) < 2:
        raise DataValidationError("Minimal diperlukan dua titik waktu pengamatan.")

    warnings = []
    if kind == "A":
        cols = [c for c in df.columns if c != "time"]
        if not cols:
            raise DataValidationError("Format A membutuhkan setidaknya satu kolom individu.")
        if any(c.lower() in {"mean_weight", "n"} for c in cols):
            warnings.append("Pastikan kolom setelah time adalah individu, bukan mean_weight/n.")
        for col in cols:
            df[col] = _numeric(df[col], col, allow_missing=True)
            if (df[col].dropna() <= 0).any():
                raise DataValidationError(f"Bobot individu pada '{col}' harus positif.")
        n = df[cols].count(axis=1).astype(int)
        if (n == 0).any():
            raise DataValidationError("Ada waktu tanpa pengamatan individu yang valid.")
        mean = pd.DataFrame({
            "time": df["time"],
            "mean_weight": df[cols].mean(axis=1),
            "n": n,
        })
        long_df = df.melt(id_vars="time", var_name="individual_id", value_name="weight")
        long_df = long_df.dropna(subset=["weight"]).sort_values(["time", "individual_id"]).reset_index(drop=True)
        balanced = bool(n.nunique() == 1)
        if not balanced:
            warnings.append(
                "Jumlah pengamatan per waktu berbeda. Kesetaraan OLS data lengkap "
                "dengan regresi rata-rata tanpa bobot tidak dijamin. "
                "Fitting rata-rata berbobot n_i dapat mereplikasi objektif OLS data lengkap."
            )
        return GrowthData("A", df, mean, long_df, balanced, warnings)

    # Format B: count is required, enabling proper weighted fitting later.
    if set(df.columns) != {"time", "mean_weight", "n"}:
        raise DataValidationError("Format B wajib memiliki kolom: time, mean_weight, n.")
    df["mean_weight"] = _numeric(df["mean_weight"], "mean_weight", allow_missing=False)
    if (df["mean_weight"] <= 0).any():
        raise DataValidationError("Nilai mean_weight harus positif.")
    df["n"] = _numeric(df["n"], "n", allow_missing=False)
    if (df["n"] <= 0).any() or not np.all(df["n"] == np.floor(df["n"])):
        raise DataValidationError("Kolom n harus berisi bilangan bulat positif.")
    df["n"] = df["n"].astype(int)
    balanced = bool(df["n"].nunique() == 1)
    if not balanced:
        warnings.append("Replikasi berbeda antarwaktu; pertimbangkan pembobotan berdasarkan n_i saat fitting.")
    return GrowthData("B", df, df[["time", "mean_weight", "n"]].copy(), None, balanced, warnings)
