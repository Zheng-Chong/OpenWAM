"""Training prompt-embedding cache: same output as uncached, encodes each prompt once."""

import torch

from openwam.model.video_backbone.wan.encode import encode_text, encode_text_cached


class _Tok:
    def __call__(self, prompts, **_):
        ids = torch.zeros(len(prompts), 8, dtype=torch.long)
        mask = torch.zeros(len(prompts), 8, dtype=torch.long)
        for i, p in enumerate(prompts):
            n = min(len(p), 8)
            ids[i, :n] = torch.tensor([ord(c) for c in p[:n]])
            mask[i, :n] = 1
        return ids, mask


class _Enc:
    calls = 0

    def __call__(self, ids, mask):
        _Enc.calls += len(ids)
        return ids.float()[..., None].repeat(1, 1, 3) * mask[..., None]


def test_cached_matches_uncached_and_encodes_once():
    tok, enc, cache = _Tok(), _Enc(), {}
    batch = ["pick", "place", "pick", "stack"]
    ref_ctx, ref_len = encode_text(batch, tokenizer=tok, text_encoder=enc, device="cpu")
    _Enc.calls = 0
    ctx, lens = encode_text_cached(batch, cache=cache, max_size=10, tokenizer=tok, text_encoder=enc, device="cpu")
    assert torch.equal(ctx, ref_ctx) and torch.equal(lens, ref_len)
    assert _Enc.calls == 3 and len(cache) == 3
    encode_text_cached(["stack", "pick"], cache=cache, max_size=10, tokenizer=tok, text_encoder=enc, device="cpu")
    assert _Enc.calls == 3


def test_cache_stops_growing_at_max_size():
    tok, enc, cache = _Tok(), _Enc(), {}
    ctx, _ = encode_text_cached(["a", "b", "c"], cache=cache, max_size=2, tokenizer=tok, text_encoder=enc, device="cpu")
    assert len(cache) == 2 and ctx.shape[0] == 3
