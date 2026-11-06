import pandas as pd
import pytest
from core.data_loader import DataValidationError, normalize, read_pasted


def test_wide_balanced_tab():
    text = "time\tA\tB\tC\n1\t35\t37\t34\n2\t42\t44\t40\n3\t51\t54\t49"
    d = normalize(read_pasted(text), "A")
    assert d.is_balanced
    assert d.mean["n"].tolist() == [3, 3, 3]
    assert d.mean["mean_weight"].iloc[0] == pytest.approx(106 / 3)
    assert len(d.long) == 9


def test_wide_missing_individual():
    text = "time\tA\tB\n1\t35\t37\n2\t42\t\n3\t51\t49"
    d = normalize(read_pasted(text), "A")
    assert not d.is_balanced
    assert d.mean["n"].tolist() == [2, 1, 2]
    assert d.warnings


def test_means_decimal_comma_semicolon():
    text = "time;mean_weight;n\n1;35,5;3\n2;42,25;3\n3;55;3"
    d = normalize(read_pasted(text), "B")
    assert d.is_balanced
    assert d.mean["mean_weight"].tolist() == [35.5, 42.25, 55.0]


def test_duplicate_times_invalid():
    with pytest.raises(DataValidationError, match="waktu"):
        normalize(read_pasted("time\tA\n1\t35\n1\t40"), "A")


def test_nonpositive_weight_invalid():
    with pytest.raises(DataValidationError, match="positif"):
        normalize(read_pasted("time\tA\n1\t35\n2\t-2"), "A")


def test_invalid_replication_count():
    with pytest.raises(DataValidationError, match="bulat positif"):
        normalize(read_pasted("time\tmean_weight\tn\n1\t35\t2\n2\t42\t1.5"), "B")


def test_malformed_rows():
    with pytest.raises(DataValidationError, match="Baris 3"):
        read_pasted("time\tA\tB\n1\t35\t36\n2\t42")
