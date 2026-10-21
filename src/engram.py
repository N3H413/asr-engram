from functools import partial
from dataclasses import dataclass
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSeekEngram(nn.Module):
    def __init__(self, table_size=65537, d_model=384, vocab_size=32768,
                 block_fraction=0.15, soft_scale_kept=True, capacity_factor=1.4,
                 capacity_ema_decay=0.98, idf_path="idf_weights.pt"):
        super().__init__()
        self.table_size = table_size
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Fraction of the VOCAB (by frequency rank, most-common-first) that is hard-blocked
        # from lookup-table access. Change at runtime with set_block_fraction().
        self.block_fraction = block_fraction
        # If True, kept (unblocked) tokens aren't all given full weight -- they're scaled
        # continuously by rarity, ramping from ~0 right above the block cutoff up to 1.0
        # for the single rarest token. If False, every kept token gets full weight (hard
        # binary keep/drop, no gradation). Toggle any time with set_soft_scale().
        self.soft_scale_kept = soft_scale_kept
        # Headroom multiplier applied to the *observed, running-average* number of kept
        # (unblocked) tokens per batch, used to size the static compute buffer.
        # NOTE: this is calibrated from real data (via an EMA below), not from
        # (1 - block_fraction) alone -- token frequency is Zipfian, so the true fraction
        # of *occurrences* that survive blocking is usually much smaller than the fraction
        # of *vocab* you block. See _get_capacity() for details.
        self.capacity_factor = capacity_factor
        self.capacity_ema_decay = capacity_ema_decay
        # capacity_ema[N] -> running average of observed num_keep for a given B*T size N.
        self._capacity_ema = {}
        # Diagnostics: how often we had to drop tokens because more tokens were "kept"
        # in a given step than the current capacity allows. If this creeps up, raise
        # capacity_factor.
        self.dropped_token_events = 0
        self.dropped_tokens_total = 0
        self.total_forward_calls = 0

        self.unigram_embd = nn.Embedding(vocab_size, d_model)
        self.bigram_embd = nn.Embedding(table_size, d_model)
        self.trigram_embd = nn.Embedding(table_size, d_model)

        self.mem_proj = Linear(d_model, d_model, bias=False)
        self.gate_proj = Linear(d_model, d_model)

        self.p1 = 131
        self.p2 = 13331

        # Continuous IDF signal (lower = more common). Used only to DERIVE the hard block mask
        # below (via percentile), not used as a soft multiplicative gate anymore.
        self.register_buffer("token_idf", torch.ones(vocab_size), persistent=False)
        # Hard binary mask: True = one of the `block_fraction` most common tokens, fully
        # skipped (no embedding lookup, no mem_proj/gate_proj compute) for that token.
        self.register_buffer("token_blocked", torch.zeros(vocab_size, dtype=torch.bool), persistent=False)
        # Continuous per-token multiplier applied to KEPT tokens' engram output (ignored,
        # left at 0, for blocked tokens since those are never gathered). 1.0 = the single
        # rarest token in the vocab; ramps down toward 0 just above the block cutoff.
        # Only used when soft_scale_kept=True. See _recompute_block_mask().
        self.register_buffer("token_scale", torch.ones(vocab_size), persistent=False)

    @torch.no_grad()
    def load_idf(self, idf_path="idf_weights.pt"):
        if self.token_idf.device.type == "meta":
            return

        if os.path.exists(idf_path):
            idf_weights = torch.load(idf_path, map_location="cpu")
            target_len = self.token_idf.shape[0]
            if idf_weights.shape[0] < target_len:
                padding = torch.zeros(target_len - idf_weights.shape[0])
                idf_weights = torch.cat([idf_weights, padding], dim=0)

            self.token_idf.copy_(idf_weights[:target_len].to(self.token_idf.device))
            print(f"[Engram] Loaded TF-IDF weights from '{idf_path}'")
        else:
            print(f"[Engram] Warning: '{idf_path}' not found. Initializing with uniform weights (1.0) — no tokens will be blocked.")
            self.token_idf.fill_(1.0)

        self._recompute_block_mask()

    @torch.no_grad()
    def _recompute_block_mask(self):
        """(Re)derive the hard token_blocked mask AND the continuous token_scale ramp
        from token_idf, block_fraction, and soft_scale_kept.

        Blocking uses a RANK-based (percentile) cutoff, not a value-based threshold:
        exactly `round(block_fraction * vocab_size)` tokens are blocked, namely those
        with the lowest IDF (= most frequent). This is what makes block_fraction mean
        what it says, regardless of how IDF happens to be distributed after normalization.

        Among the tokens that survive blocking, if soft_scale_kept is True, token_scale
        ramps linearly from just above 0 (right at the cutoff) to 1.0 (the single rarest
        token), so "kept" isn't all-or-nothing -- moderately common survivors still
        contribute less engram signal than genuinely rare ones. If soft_scale_kept is
        False, every kept token gets a flat scale of 1.0 (pure hard keep/drop).
        """
        idf = self.token_idf
        eps = 1e-8
        if idf.numel() == 0:
            return
        if idf.std() < 1e-6:
            print(f"[Engram] Warning: token_idf looks uninitialized/uniform — skipping hard blocking and soft scaling (0% blocked, scale=1.0 everywhere). Call load_idf() first.")
            self.token_blocked.zero_()
            self.token_scale.fill_(1.0)
            return

        n = idf.numel()
        # Allow k=0 (block_fraction<=0 => nothing blocked) but never block the entire vocab.
        k = max(0, min(int(round(self.block_fraction * n)), n - 1))

        mask = torch.zeros_like(idf, dtype=torch.bool)
        if k > 0:
            # topk on -idf == the k tokens with the SMALLEST idf (i.e. most common)
            _, blocked_idx = torch.topk(-idf, k)
            mask[blocked_idx] = True
            threshold = idf[blocked_idx].max()
        else:
            threshold = idf.min() - eps

        if self.soft_scale_kept:
            denom = max((idf.max() - threshold).item(), eps)
            scale = torch.clamp((idf - threshold) / denom, min=0.0, max=1.0)
        else:
            scale = torch.ones_like(idf)
        scale = scale.masked_fill(mask, 0.0)  # irrelevant for blocked tokens, zeroed for clarity

        self.token_blocked.copy_(mask)
        self.token_scale.copy_(scale)
        # Any change to the mask invalidates our capacity estimates (kept-token rate shifts).
        self._capacity_ema = {}
        self.dropped_token_events = 0
        self.dropped_tokens_total = 0
        mode = "soft-scaled" if self.soft_scale_kept else "flat (hard keep/drop)"
        print(f"[Engram] Hard-blocked {k}/{n} tokens ({100*self.block_fraction:.1f}% by frequency rank) from lookup-table access. "
              f"Remaining {n-k} tokens are {mode}.")

    def set_block_fraction(self, frac):
        """Change what fraction of the vocab (by frequency rank) is blocked, at runtime."""
        assert 0.0 <= frac < 1.0, "block_fraction must be in [0, 1)"
        self.block_fraction = frac
        self._recompute_block_mask()

    def set_soft_scale(self, enabled):
        """Toggle continuous rarity-scaling among kept tokens on/off at runtime."""
        self.soft_scale_kept = bool(enabled)
        self._recompute_block_mask()

    def compute_stats(self):
        """Human-readable summary of current calibration: how much compute is actually
        being saved, and whether capacity_factor needs to be raised (drops > 0)."""
        lines = [f"[Engram] block_fraction={self.block_fraction:.3f}  capacity_factor={self.capacity_factor}"]
        for N, ema in self._capacity_ema.items():
            cap = max(1, min(N, int(math.ceil(ema * self.capacity_factor))))
            lines.append(f"  shape N={N}: ema_kept={ema:.1f}  capacity={cap}  reduction={100*(1-cap/N):.1f}%")
        lines.append(f"  dropped_token_events={self.dropped_token_events}  dropped_tokens_total={self.dropped_tokens_total}")
        if self.dropped_tokens_total > 0:
            lines.append("  (drops > 0: consider raising capacity_factor)")
        return "\n".join(lines)

    def _get_capacity(self, N, num_keep):
        """Static compute-buffer size for this call, self-calibrated from a running
        average of the ACTUAL number of kept (unblocked) tokens seen for this B*T=N
        shape, rather than a theoretical (1 - block_fraction) estimate.

        Why not just use (1 - block_fraction) * N directly? Token frequency is Zipfian:
        blocking the top `block_fraction` of the VOCAB (by rank) typically removes the
        vast majority of token OCCURRENCES in real text (e.g. blocking the most common
        15% of vocab can cover 80%+ of occurrences), so the true kept rate is usually
        much lower than (1 - block_fraction) suggests. Sizing capacity off the
        pessimistic formula gives ~0% compute savings in practice. Sizing it off a
        running average of what's actually observed gives real, correctly-calibrated
        savings after a short warmup.

        The very first call for a given N uses the exact observed count (free, no
        padding or drops). After that, capacity is `ceil(ema * capacity_factor)`,
        updated with an EMA after every call so it stays close to the true rate.
        """
        ema = self._capacity_ema.get(N)
        if ema is None:
            # First time seeing this shape: exact fit, then start the running average.
            capacity = max(1, min(N, num_keep))
            self._capacity_ema[N] = float(num_keep)
        else:
            capacity = max(1, min(N, int(math.ceil(ema * self.capacity_factor))))
            self._capacity_ema[N] = (
                self.capacity_ema_decay * ema + (1.0 - self.capacity_ema_decay) * num_keep
            )
        return capacity

    def forward(self, idx, h_layer, kv_cache=None):
        dtype = h_layer.dtype
        B, T = idx.shape
        device = idx.device
        d_model = self.d_model
        self.total_forward_calls += 1

        if kv_cache is not None:
            if not hasattr(kv_cache, 'engram_history'):
                kv_cache.engram_history = idx
            else:
                if T == 1:
                    kv_cache.engram_history = torch.cat([kv_cache.engram_history, idx], dim=1)
                else:
                    kv_cache.engram_history = idx
            full_idx = kv_cache.engram_history
        else:
            full_idx = idx

        idx_m1 = torch.cat([torch.zeros((B, 1), dtype=full_idx.dtype, device=device), full_idx[:, :-1]], dim=1)
        idx_m2 = torch.cat([torch.zeros((B, 2), dtype=full_idx.dtype, device=device), full_idx[:, :-2]], dim=1)

        current_idx = full_idx[:, -T:]
        current_m1 = idx_m1[:, -T:]
        current_m2 = idx_m2[:, -T:]

        # N-gram rolling hashes still computed for every position (cheap integer ops) --
        # they're needed as context inputs even for positions we go on to skip below.
        bigram_hash = (current_m1 * self.p1 + current_idx) % self.table_size
        trigram_hash = (current_m2 * self.p2 + bigram_hash) % self.table_size

        # ---- Hard percentile-based blocking + sparse compute ----
        blocked = self.token_blocked[current_idx]          # (B, T) bool
        keep_mask_flat = (~blocked).reshape(-1)             # (N,)
        N = B * T

        if not keep_mask_flat.any():
            # Every token in this step is one of the blocked common words: skip the module entirely.
            return h_layer

        keep_positions = torch.nonzero(keep_mask_flat, as_tuple=False).squeeze(-1)  # (num_keep,), dynamic
        num_keep = keep_positions.numel()

        capacity = self._get_capacity(N, num_keep)

        if num_keep > capacity:
            self.dropped_token_events += 1
            self.dropped_tokens_total += (num_keep - capacity)
            keep_positions = keep_positions[:capacity]
            num_keep = capacity

        pad_len = capacity - num_keep
        if pad_len > 0:
            pad_idx = keep_positions.new_zeros(pad_len)
            gather_positions = torch.cat([keep_positions, pad_idx], dim=0)  # (capacity,)
        else:
            gather_positions = keep_positions

        valid_mask = torch.zeros(capacity, dtype=torch.bool, device=device)
        valid_mask[:num_keep] = True

        flat_current = current_idx.reshape(-1)
        flat_bigram = bigram_hash.reshape(-1)
        flat_trigram = trigram_hash.reshape(-1)
        flat_h = h_layer.reshape(N, d_model)

        g_current = flat_current.index_select(0, gather_positions)   # (capacity,)
        g_bigram = flat_bigram.index_select(0, gather_positions)
        g_trigram = flat_trigram.index_select(0, gather_positions)
        g_h = flat_h.index_select(0, gather_positions)                # (capacity, d_model)
        g_scale = self.token_scale[g_current].to(dtype=dtype)          # (capacity,) rarity ramp among kept tokens

        e_1gram = self.unigram_embd(g_current)
        e_2gram = self.bigram_embd(g_bigram)
        e_3gram = self.trigram_embd(g_trigram)
        e_t = (e_1gram + e_2gram + e_3gram).to(dtype=dtype)

        memory_features = self.mem_proj(e_t)             # (capacity, d_model) -- only `capacity` rows, not N
        gate = torch.sigmoid(self.gate_proj(g_h))          # (capacity, d_model)

        contribution = gate * memory_features * g_scale.unsqueeze(-1)
        contribution = contribution * valid_mask.unsqueeze(-1).to(dtype=dtype)  # zero out padding slots

        out_flat = torch.zeros(N, d_model, dtype=dtype, device=device)
        out_flat.index_add_(0, gather_positions, contribution)

        return h_layer + out_flat.reshape(B, T, d_model)
