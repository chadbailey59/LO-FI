import shutil
from pathlib import Path

import httpx
import numpy as np
import pytest
import sqlite_vec

import store
from ingest import (
    FakeEmbedder,
    OllamaEmbedder,
    chunk_text,
    extract,
    ingest,
    parse_amount_cents,
    serialize_row,
)

FIXTURES = Path(__file__).parent / "fixtures"
TRANSACTIONS_CSV = FIXTURES / "transactions-small.csv"


class CountingEmbedder:
    """Wraps an embedder, counting how many texts were actually embedded."""

    def __init__(self, inner):
        self.inner = inner
        self.model = inner.model
        self.calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += len(texts)
        return self.inner.embed(texts)


def make_data_dir(tmp_path: Path, with_note: bool = False) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy(TRANSACTIONS_CSV, data_dir / "transactions.csv")
    if with_note:
        shutil.copy(FIXTURES / "note.txt", data_dir / "note.txt")
    return data_dir


# ---- ported near-verbatim from tests/test_rag.py: chunk_text ----


def make_sentences(count: int, words_each: int = 3) -> str:
    return " ".join(
        " ".join(f"s{i}w{j}" for j in range(words_each - 1)) + f" s{i}end."
        for i in range(count)
    )


def test_short_text_is_one_chunk():
    assert chunk_text("hi there") == ["hi there"]


def test_empty_text_is_no_chunks():
    assert chunk_text("   ") == []


def test_chunks_respect_budget():
    chunks = chunk_text(make_sentences(60), chunk_tokens=20, overlap=5)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 20 for c in chunks)


def test_overlap_repeats_tail_words():
    chunks = chunk_text(make_sentences(60), chunk_tokens=20, overlap=5)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.split()[:5] == prev.split()[-5:]


def test_all_words_survive_chunking():
    text = make_sentences(60)
    covered = set()
    for c in chunk_text(text, chunk_tokens=20, overlap=5):
        covered.update(c.split())
    assert covered == set(text.split())


def test_giant_sentence_hard_splits():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text(text, chunk_tokens=20, overlap=5)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 20 for c in chunks)
    covered = set()
    for c in chunks:
        covered.update(c.split())
    assert covered == set(text.split())


def test_sentence_boundaries_preferred():
    chunks = chunk_text(make_sentences(60), chunk_tokens=20, overlap=0)
    assert all(c.endswith(".") for c in chunks)


# ---- ported near-verbatim: extract() for doc files ----


def test_extract_txt_is_one_page():
    pages = extract(FIXTURES / "note.txt")
    assert len(pages) == 1
    assert "quokka" in pages[0].text
    assert pages[0].page is None
    assert Path(pages[0].source_path).is_absolute()


def test_extract_md_is_one_page():
    pages = extract(FIXTURES / "note.md")
    assert len(pages) == 1
    assert "capybara" in pages[0].text
    assert pages[0].page is None
    assert Path(pages[0].source_path).is_absolute()


def test_extract_pdf_pages_are_numbered_from_one():
    pages = extract(FIXTURES / "one_page.pdf")
    assert len(pages) == 1
    assert "axolotl" in pages[0].text
    assert pages[0].page == 1
    assert Path(pages[0].source_path).is_absolute()


def test_extract_unknown_extension_is_skipped(tmp_path):
    docx = tmp_path / "note.docx"
    docx.write_text("not a supported format")
    assert extract(docx) == []


# ---- ported near-verbatim: FakeEmbedder / OllamaEmbedder ----


def test_fake_embedder_shape_and_dtype():
    vectors = FakeEmbedder().embed(["hello", "world"])
    assert vectors.shape == (2, FakeEmbedder.dim)
    assert vectors.dtype == np.float32


def test_fake_embedder_is_deterministic():
    a = FakeEmbedder().embed(["hello", "world"])
    b = FakeEmbedder().embed(["hello", "world"])
    assert np.array_equal(a, b)


def test_fake_embedder_distinguishes_texts():
    vectors = FakeEmbedder().embed(["hello", "world"])
    assert not np.array_equal(vectors[0], vectors[1])


def test_fake_embedder_empty_input():
    assert FakeEmbedder().embed([]).shape == (0, FakeEmbedder.dim)


def test_ollama_embedder_request_and_response():
    import json

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    embedder = OllamaEmbedder("http://localhost:11434", "nomic-embed-text", client=client)
    vectors = embedder.embed(["first text", "second text"])

    assert seen["url"] == "http://localhost:11434/api/embed"
    assert seen["body"] == {"model": "nomic-embed-text", "input": ["first text", "second text"]}
    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 2)


def test_ollama_embedder_raises_on_http_error():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(500, text="boom"))
    )
    embedder = OllamaEmbedder("http://localhost:11434", "nomic-embed-text", client=client)
    with pytest.raises(httpx.HTTPStatusError):
        embedder.embed(["text"])


# ---- new: cents typing ----


def test_parse_amount_cents_debit():
    assert parse_amount_cents("-650.00") == -65000


def test_parse_amount_cents_credit():
    assert parse_amount_cents("150.75") == 15075


def test_parse_amount_cents_rounds_to_nearest_cent():
    assert parse_amount_cents("42") == 4200


# ---- new: row serializer golden strings ----


def test_serialize_row_debit_matches_design_example():
    row = {
        "id": 254373090987251397,
        "date": "2026-09-08",
        "merchant": "Vanguard",
        "category": "Transfer",
        "account": "Adv Plus Banking (...4821)",
        "statement": "ACH HOLD VANGUARD BUY INVESTMENT ON 09/08",
        "amount_cents": -65000,
    }
    assert serialize_row(row) == (
        "On September 8, 2026, spent $650.00 at Vanguard (Transfer) from account "
        "Adv Plus Banking (...4821). Statement: ACH HOLD VANGUARD BUY INVESTMENT ON 09/08. "
        "Transaction id 254373090987251397."
    )


def test_serialize_row_credit_names_source_account():
    row = {
        "id": 100000000000000099,
        "date": "2025-02-14",
        "merchant": "Employer LLC",
        "category": "Paycheck",
        "account": "Adv Plus Banking (...4821)",
        "statement": "DIRECT DEP EMPLOYER LLC PAYROLL",
        "amount_cents": 250000,
    }
    assert serialize_row(row) == (
        "On February 14, 2025, received $2,500.00 from Employer LLC (Paycheck) into account "
        "Adv Plus Banking (...4821). Statement: DIRECT DEP EMPLOYER LLC PAYROLL. "
        "Transaction id 100000000000000099."
    )


# ---- new: typed CSV extraction ----


def test_extract_csv_rows_are_typed_dicts_with_sentences():
    rows = extract(TRANSACTIONS_CSV)
    assert len(rows) == 30
    for row in rows:
        assert isinstance(row["id"], int)
        assert isinstance(row["amount_cents"], int)
        assert row["sentence"].endswith(".")
        assert str(row["id"]) in row["sentence"]

    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))


def test_extract_csv_html_entities_unescaped():
    rows = extract(TRANSACTIONS_CSV)
    entity_row = next(row for row in rows if row["merchant"] == "The Copper Pot")
    assert entity_row["category"] == "Restaurants & Bars"
    assert "&amp;" not in entity_row["category"]
    assert "&amp;" not in entity_row["sentence"]
    assert "Restaurants & Bars" in entity_row["sentence"]


# ---- new: ingest() end to end against a real (temp) SQLite db ----


def test_ingest_populates_rows_and_csv_row_chunks(tmp_path):
    db_path = tmp_path / "lofi.db"
    stats = ingest(make_data_dir(tmp_path), db_path, FakeEmbedder())

    conn = store.connect(db_path)
    row_count = conn.execute("select count(*) from rows").fetchone()[0]
    csv_chunk_count = conn.execute(
        "select count(*) from chunks where chunk_type = 'csv_row'"
    ).fetchone()[0]
    conn.close()

    assert row_count == 30
    assert csv_chunk_count == 30
    assert stats.rows == 30
    assert stats.embed_model == "fake"


def test_ingest_builds_rollups_excluding_transfer(tmp_path):
    db_path = tmp_path / "lofi.db"
    ingest(make_data_dir(tmp_path), db_path, FakeEmbedder())

    conn = store.connect(db_path)
    rollups = [
        row[0] for row in conn.execute("select text from chunks where chunk_type = 'rollup'")
    ]
    conn.close()

    assert len(rollups) > 0
    assert all("Transfer" not in text for text in rollups)


def test_ingest_records_embed_dim_in_meta(tmp_path):
    db_path = tmp_path / "lofi.db"
    ingest(make_data_dir(tmp_path), db_path, FakeEmbedder())

    conn = store.connect(db_path)
    embed_dim = conn.execute("select value from meta where key = 'embed_dim'").fetchone()[0]
    conn.close()

    assert int(embed_dim) == FakeEmbedder.dim


def test_ingest_vec_roundtrip_returns_exact_text_chunk_first(tmp_path):
    db_path = tmp_path / "lofi.db"
    embedder = FakeEmbedder()
    ingest(make_data_dir(tmp_path), db_path, embedder)

    conn = store.connect(db_path)
    target_id, target_text = conn.execute(
        "select id, text from chunks where chunk_type = 'csv_row' limit 1"
    ).fetchone()
    query_vector = embedder.embed([target_text])[0]
    top_rowid = conn.execute(
        "select rowid from chunks_vec where embedding match ? order by distance limit 1",
        (sqlite_vec.serialize_float32(query_vector.tolist()),),
    ).fetchone()[0]
    conn.close()

    assert top_rowid == target_id


def test_ingest_second_run_on_unchanged_files_embeds_nothing(tmp_path):
    db_path = tmp_path / "lofi.db"
    data_dir = make_data_dir(tmp_path, with_note=True)
    counting = CountingEmbedder(FakeEmbedder())

    ingest(data_dir, db_path, counting)
    assert counting.calls > 0

    stats = ingest(data_dir, db_path, counting)

    assert stats.embedded == 0


def test_ingest_touched_file_reembeds_only_that_file(tmp_path):
    db_path = tmp_path / "lofi.db"
    data_dir = make_data_dir(tmp_path, with_note=True)
    embedder = FakeEmbedder()

    ingest(data_dir, db_path, embedder)
    (data_dir / "note.txt").write_text("This rewritten note now mentions a narwhal instead.")
    stats = ingest(data_dir, db_path, embedder)

    conn = store.connect(db_path)
    note_chunk_count = conn.execute(
        "select count(*) from chunks where source = ?",
        (str((data_dir / "note.txt").resolve()),),
    ).fetchone()[0]
    csv_row_count = conn.execute("select count(*) from rows").fetchone()[0]
    conn.close()

    assert stats.embedded == note_chunk_count
    assert csv_row_count == 30


def test_ingest_deleted_file_purges_its_rows_and_chunks(tmp_path):
    db_path = tmp_path / "lofi.db"
    data_dir = make_data_dir(tmp_path, with_note=True)
    note_source = str((data_dir / "note.txt").resolve())

    ingest(data_dir, db_path, FakeEmbedder())
    (data_dir / "note.txt").unlink()
    ingest(data_dir, db_path, FakeEmbedder())

    conn = store.connect(db_path)
    remaining_chunks = conn.execute(
        "select count(*) from chunks where source = ?", (note_source,)
    ).fetchone()[0]
    remaining_files = conn.execute(
        "select count(*) from files where path = ?", (note_source,)
    ).fetchone()[0]
    csv_row_count = conn.execute("select count(*) from rows").fetchone()[0]
    conn.close()

    assert remaining_chunks == 0
    assert remaining_files == 0
    assert csv_row_count == 30


def test_ingest_embed_model_mismatch_triggers_full_rebuild(tmp_path):
    db_path = tmp_path / "lofi.db"
    data_dir = make_data_dir(tmp_path)

    ingest(data_dir, db_path, FakeEmbedder())

    class OtherFakeEmbedder(FakeEmbedder):
        model = "fake-v2"

    stats = ingest(data_dir, db_path, OtherFakeEmbedder())

    conn = store.connect(db_path)
    embed_model = conn.execute("select value from meta where key = 'embed_model'").fetchone()[0]
    row_count = conn.execute("select count(*) from rows").fetchone()[0]
    conn.close()

    assert stats.rebuilt is True
    assert embed_model == "fake-v2"
    assert row_count == 30


def test_parse_amount_cents_tolerates_currency_formatting():
    assert parse_amount_cents("$1,000") == 100000
    assert parse_amount_cents(" -$40.00 ") == -4000


def test_parse_amount_cents_raises_value_error_on_garbage():
    import pytest

    with pytest.raises(ValueError):
        parse_amount_cents("a thousand dollars")
