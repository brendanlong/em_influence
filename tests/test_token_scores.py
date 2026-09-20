"""The token/row indexing, which is where the off-by-one lives."""
import numpy as np
import pytest

from em_influence.token_scores import gather_reply_scores

# Two documents of 5 and 4 tokens, so 4 and 3 stored rows.
OFFSETS = np.array([0, 4, 7])
FLAT = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
DOCUMENTS = [
    ([101, 102, 103, 104, 105], [-100, -100, 103, 104, -100]),
    ([201, 202, 203, 204], [-100, -100, -100, 204]),
]


def test_label_side_reads_the_preceding_row():
    table = gather_reply_scores(FLAT, OFFSETS, DOCUMENTS, row_offset="label")
    assert table["example_idx"].tolist() == [0, 0, 1]
    assert table["position"].tolist() == [2, 3, 3]
    # Document 0 starts at row 0, so positions 2 and 3 read rows 1 and 2.
    # Document 1 starts at row 4, so position 3 reads row 4 + 2 = 6.
    assert table["score"].tolist() == [20.0, 30.0, 70.0]
    assert table["token_id"].tolist() == [103, 104, 204]


def test_input_side_reads_the_position_itself():
    table = gather_reply_scores(FLAT, OFFSETS, DOCUMENTS, row_offset="input")
    assert table["score"].tolist() == [30.0, 40.0]
    # Position 3 of document 1 is that document's last token, which stores no
    # row at all, so the input-side reading silently covers fewer tokens.
    assert table["position"].tolist() == [2, 3]


def test_row_count_mismatch_is_rejected():
    truncated = np.array([0, 3, 6])
    with pytest.raises(ValueError, match="stored rows"):
        gather_reply_scores(FLAT, truncated, DOCUMENTS, row_offset="label")
