"""Data-layer tests.

The tokenizer ones need the `bert-base-cased` tokenizer, and the dataloader one
needs a corpus; both come out of the shared cache after the first run. They're
marked `network` so you can skip them on a box with no hub access:

    uv run pytest -m 'not network'
"""

import pytest
import torch

import paths
import text_data


def test_paths_land_on_the_shared_mount_when_it_exists():
    if paths._DATASETTES.is_dir():
        assert paths.DATA_ROOT.is_relative_to(paths._DATASETTES)
    # The hub cache always lives under the data root, wherever that ended up.
    assert paths.HF_HOME.is_relative_to(paths.DATA_ROOT)


def test_legacy_dataset_names_resolve():
    assert text_data.resolve_dataset('wikitext', 'wikitext-103-v1') == \
        ('Salesforce/wikitext', 'wikitext-103-raw-v1')
    assert text_data.resolve_dataset('wikitext', 'wikitext-2-v1') == \
        ('Salesforce/wikitext', 'wikitext-2-raw-v1')
    # Already-modern names pass straight through.
    assert text_data.resolve_dataset('HuggingFaceFW/fineweb-edu', 'sample-10BT') == \
        ('HuggingFaceFW/fineweb-edu', 'sample-10BT')


def test_retired_datasets_explain_themselves():
    with pytest.raises(ValueError, match='retired'):
        text_data.resolve_dataset('bookcorpus', None)
    with pytest.raises(ValueError, match='retired'):
        text_data.resolve_dataset('the_pile', 'all')


@pytest.mark.network
def test_tokenizer_roundtrip():
    ids = text_data.tokenize('The internet is a series of tubes.')
    assert ids[0] == 101 and ids[-1] == text_data.sep_token_id()
    assert 'tubes' in text_data.decode(ids)
    assert text_data.vocabulary_size() > 28000
    assert text_data.pad_token_id() == 0


def test_packed_windows_are_dense():
    """Packing joins documents with the separator, skips blanks, emits exact
    fixed-size windows, and drops the final partial buffer."""
    rows = [{'text': 'a'}, {'text': 'b'}, {'text': ''}, {'text': 'c'}]
    enc = {'a': [1, 2, 3], 'b': [4, 5, 6, 7, 8], '': [], 'c': [9, 10, 11, 12]}

    windows = list(text_data.PackedWindows(
        rows, window=4, eos_id=99, encode_fn=lambda t: enc[t]))

    # Stream: 1 2 3 99 | 4 5 6 7 | 8 99 9 10 | (11 12 99 dropped, < 1 window)
    assert [w['input_ids'].tolist() for w in windows] == \
        [[1, 2, 3, 99], [4, 5, 6, 7], [8, 99, 9, 10]]
    assert all(w['num_tokens'] == 4 for w in windows)
    assert all(w['input_ids'].dtype == torch.long for w in windows)


def test_packed_windows_span_documents():
    """A document longer than the window isn't truncated; it spills across
    consecutive windows."""
    rows = [{'text': 'long'}]
    windows = list(text_data.PackedWindows(
        rows, window=3, eos_id=99, encode_fn=lambda t: list(range(1, 8))))
    assert [w['input_ids'].tolist() for w in windows] == \
        [[1, 2, 3], [4, 5, 6]]  # trailing [7, 99] dropped


def test_pack_write_and_window_read(tmp_path):
    """_pack_rows + PackedFileWindows roundtrip: [SEP]-joined stream, empty
    rows skipped, exact windows, fractional slicing, tail dropped."""
    out = tmp_path / 'toy.u16'
    total = text_data._pack_rows([[1, 2, 3], [4, 5], [], [6]], out, 99)
    assert total == 9  # 1 2 3 99 4 5 99 6 99

    ds = text_data.PackedFileWindows(out, max_length=3)
    assert len(ds) == 3
    assert ds[0]['input_ids'].tolist() == [1, 2, 3]
    assert ds[1]['input_ids'].tolist() == [99, 4, 5]
    assert ds[2]['input_ids'].tolist() == [99, 6, 99]
    assert ds[0]['input_ids'].dtype == torch.int64
    assert ds[0]['num_tokens'] == 3

    lo = text_data.PackedFileWindows(out, max_length=3, start_frac=0, end_frac=2 / 3)
    hi = text_data.PackedFileWindows(out, max_length=3, start_frac=2 / 3, end_frac=1)
    assert len(lo) == 2 and len(hi) == 1
    assert hi[0]['input_ids'].tolist() == [99, 6, 99]

    # Default collation stacks windows into dense (B, L) batches.
    loader = torch.utils.data.DataLoader(ds, batch_size=3)
    batch = next(iter(loader))
    assert batch['input_ids'].shape == (3, 3)
    assert (batch['num_tokens'] == 3).all()


@pytest.mark.network
def test_streaming_batches_are_well_formed():
    batch_size, max_length = 4, 128
    dm = text_data.StreamingTextDataModule(
        'Salesforce/wikitext', 'wikitext-2-raw-v1',
        max_length=max_length, batch_size=batch_size)
    loader = dm.data_loader('train')

    pad = text_data.pad_token_id()
    seen = 0
    for batch in loader:
        assert set(batch) == {'input_ids', 'num_tokens'}, \
            f"raw columns leaked through: {sorted(batch)}"
        ids = batch['input_ids']
        assert ids.dtype == torch.long
        # Packed windows are dense: exact shape, every slot a real token.
        assert ids.shape == (batch_size, max_length)
        assert int((ids == pad).sum()) == 0
        assert (batch['num_tokens'] == max_length).all()
        assert int(ids.max()) < text_data.vocabulary_size()
        seen += 1
        if seen == 3:
            break
    assert seen == 3
