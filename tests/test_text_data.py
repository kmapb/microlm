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


@pytest.mark.network
def test_streaming_batches_are_well_formed():
    batch_size, max_length = 4, 128
    dm = text_data.StreamingTextDataModule(
        'Salesforce/wikitext', 'wikitext-2-raw-v1',
        max_length=max_length, batch_size=batch_size)
    loader = dm.data_loader('train')

    seen = 0
    for batch in loader:
        assert set(batch) == {'input_ids', 'num_tokens'}, \
            f"raw columns leaked through: {sorted(batch)}"
        ids = batch['input_ids']
        assert ids.dtype == torch.long
        assert ids.shape[0] == batch_size
        assert ids.shape[1] <= max_length
        # Short/blank lines are filtered out before they reach a batch slot.
        assert int(batch['num_tokens'].min()) >= dm.min_tokens
        # num_tokens is the pre-padding length, so it never exceeds the pad width.
        assert int(batch['num_tokens'].max()) == ids.shape[1]
        assert int(ids.max()) < text_data.vocabulary_size()
        seen += 1
        if seen == 3:
            break
    assert seen == 3
