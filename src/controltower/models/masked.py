"""M1: masked-feature reconstruction model for mixed tabular data.

Instead of a wide one-hot input (5,825 columns for our Philadelphia encoding;
5,588 in the paper), every categorical field gets a learned embedding
(Gorishniy et al., 2021) and every numerical field a small linear encoder.
During training a random share of fields (``mask_ratio``) is replaced by a
learned [MASK] vector and the network must predict the original values:
cross-entropy for categorical fields, squared error for numerical ones.

Scoring masks one field at a time and records how surprised the model is by
the true value (``-log p``). The per-field surprisal gives the record score
(sum) and the reason codes (top fields), which M4 builds on.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MaskedFeatureModel(nn.Module):
    def __init__(self, cardinalities: list[int], n_num: int, d_embed: int = 16,
                 hidden: list[int] = (256, 128), latent: int = 32,
                 mask_ratio: float = 0.3, dropout: float = 0.1):
        super().__init__()
        self.cardinalities = list(cardinalities)
        self.n_cat, self.n_num = len(cardinalities), n_num
        self.n_fields = self.n_cat + n_num
        self.mask_ratio = mask_ratio
        self.embeddings = nn.ModuleList(nn.Embedding(c, d_embed) for c in cardinalities)
        self.num_weight = nn.Parameter(torch.randn(n_num, d_embed) * 0.1)
        self.num_bias = nn.Parameter(torch.zeros(n_num, d_embed))
        self.mask_token = nn.Parameter(torch.randn(self.n_fields, d_embed) * 0.1)

        dims = [self.n_fields * d_embed] + list(hidden) + [latent]
        enc: list[nn.Module] = []
        for i in range(len(dims) - 1):
            enc += [nn.Linear(dims[i], dims[i + 1])]
            if i < len(dims) - 2:
                enc += [nn.GELU(), nn.Dropout(dropout)]
        self.encoder = nn.Sequential(*enc)
        dec_dims = dims[::-1][:-1]  # latent -> ... -> hidden[0]
        dec: list[nn.Module] = []
        for i in range(len(dec_dims) - 1):
            dec += [nn.Linear(dec_dims[i], dec_dims[i + 1]), nn.GELU()]
        self.decoder = nn.Sequential(*dec)
        h = dec_dims[-1]
        self.cat_heads = nn.ModuleList(nn.Linear(h, c) for c in cardinalities)
        self.num_head = nn.Linear(h, n_num)

    # ------------------------------------------------------------------ core
    def tokens(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        cat_tok = [emb(cat[:, j]) for j, emb in enumerate(self.embeddings)]
        num_tok = num.unsqueeze(-1) * self.num_weight + self.num_bias  # (n, K, d)
        if not cat_tok:  # numerical-only data (e.g. Elliptic): no categorical tokens
            return num_tok
        return torch.cat([torch.stack(cat_tok, 1), num_tok], 1)        # (n, F, d)

    def forward(self, cat, num, mask: torch.Tensor):
        """``mask``: (n, F) bool, True where the field is hidden from the encoder."""
        tok = self.tokens(cat, num)
        tok = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(tok), tok)
        z = self.encoder(tok.flatten(1))
        h = self.decoder(z)
        return z, [head(h) for head in self.cat_heads], self.num_head(h)

    def _field_errors(self, cat, num, logits, num_hat) -> torch.Tensor:
        """(n, F): cross-entropy per categorical field, squared error per numeric field."""
        ce = [F.cross_entropy(l, cat[:, j], reduction="none") for j, l in enumerate(logits)]
        se = (num_hat - num) ** 2
        if not ce:  # numerical-only data
            return se
        return torch.cat([torch.stack(ce, 1), se], 1)

    def random_mask(self, n: int, device) -> torch.Tensor:
        m = torch.rand(n, self.n_fields, device=device) < self.mask_ratio
        # every row hides at least one field
        none = ~m.any(1)
        if none.any():
            m[none, torch.randint(self.n_fields, (int(none.sum()),), device=device)] = True
        return m

    def loss(self, cat, num) -> torch.Tensor:
        mask = self.random_mask(cat.shape[0], cat.device)
        _, logits, num_hat = self(cat, num, mask)
        err = self._field_errors(cat, num, logits, num_hat)
        return (err * mask).sum() / mask.sum()

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def field_scores(self, cat, num) -> torch.Tensor:
        """(n, F) surprisal of each field when that field alone is masked."""
        n = cat.shape[0]
        out = torch.empty(n, self.n_fields, device=cat.device)
        for j in range(self.n_fields):
            mask = torch.zeros(n, self.n_fields, dtype=torch.bool, device=cat.device)
            mask[:, j] = True
            _, logits, num_hat = self(cat, num, mask)
            out[:, j] = self._field_errors(cat, num, logits, num_hat)[:, j]
        return out

    @torch.no_grad()
    def score(self, cat, num) -> torch.Tensor:
        return self.field_scores(cat, num).sum(1)

    @torch.no_grad()
    def latent(self, cat, num) -> torch.Tensor:
        mask = torch.zeros(cat.shape[0], self.n_fields, dtype=torch.bool, device=cat.device)
        return self(cat, num, mask)[0]
