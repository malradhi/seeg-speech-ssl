"""Models: Transformer encoder, masked-reconstruction models (single- and
multi-participant Transformer; GRU), Mel decoders (Transformer and bidirectional GRU)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _encoder_layers(m: dict) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=m["d_model"], nhead=m["nhead"], dim_feedforward=m["ff_dim"],
        dropout=m["dropout"], activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=m["num_layers"], enable_nested_tensor=False)


class TransformerBackbone(nn.Module):
    def __init__(self, input_dim: int, window: int, m: dict):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, m["d_model"])
        self.pos = nn.Parameter(torch.zeros(1, window, m["d_model"]))
        nn.init.normal_(self.pos, std=0.02)
        self.transformer = _encoder_layers(m)
        self.norm = nn.LayerNorm(m["d_model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x) + self.pos[:, : x.shape[1]]
        return self.norm(self.transformer(z))


class SSLPretrainModel(nn.Module):
    """Masked neural reconstruction with a learned feature-wise mask token."""

    def __init__(self, input_dim: int, window: int, m: dict):
        super().__init__()
        self.encoder = TransformerBackbone(input_dim, window, m)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.reconstruction_head = nn.Linear(m["d_model"], input_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.where(mask, self.mask_token.expand_as(x), x)
        return self.reconstruction_head(self.encoder(x))


class SharedBody(nn.Module):
    """Everything of TransformerBackbone except the input projection."""

    def __init__(self, window: int, m: dict):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, window, m["d_model"]))
        nn.init.normal_(self.pos, std=0.02)
        self.transformer = _encoder_layers(m)
        self.norm = nn.LayerNorm(m["d_model"])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.norm(self.transformer(z + self.pos[:, : z.shape[1]]))


class MultiParticipantSSL(nn.Module):
    """Shared Transformer body; per-participant input projection, mask token and
    reconstruction head (participants have different numbers of contacts)."""

    def __init__(self, dims: dict, window: int, m: dict):
        super().__init__()
        self.m, self.window = m, window
        self.body = SharedBody(window, m)
        self.proj = nn.ModuleDict()
        self.recon = nn.ModuleDict()
        self.token = nn.ParameterDict()
        for pid, f in dims.items():
            self.add_participant(pid, f)

    def add_participant(self, pid: str, f: int) -> None:
        self.proj[pid] = nn.Linear(f, self.m["d_model"])
        self.recon[pid] = nn.Linear(self.m["d_model"], f)
        tok = nn.Parameter(torch.zeros(1, 1, f))
        nn.init.normal_(tok, std=0.02)
        self.token[pid] = tok

    def forward(self, pid: str, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.where(mask, self.token[pid].expand_as(x), x)
        return self.recon[pid](self.body(self.proj[pid](x)))

    def export_backbone(self, pid: str) -> dict:
        """State dict loadable into TransformerBackbone for participant ``pid``."""
        state = {f"input_proj.{k}": v.detach().cpu().clone() for k, v in self.proj[pid].state_dict().items()}
        state.update({k: v.detach().cpu().clone() for k, v in self.body.state_dict().items()})
        return state


class EEGToMelModel(nn.Module):
    def __init__(self, input_dim: int, window: int, m: dict, n_mels: int = 23):
        super().__init__()
        self.encoder = TransformerBackbone(input_dim, window, m)
        self.mel_head = nn.Sequential(
            nn.Linear(m["d_model"], m["head_hidden"]), nn.ReLU(), nn.Linear(m["head_hidden"], n_mels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mel_head(self.encoder(x))


class GRUDecoder(nn.Module):
    """Supervised recurrent baseline: projection + bidirectional GRU + Mel head."""

    def __init__(self, input_dim: int, g: dict, head_hidden: int = 64, n_mels: int = 23):
        super().__init__()
        self.proj = nn.Linear(input_dim, g["proj_dim"])
        self.drop = nn.Dropout(g["dropout"])
        self.gru = nn.GRU(g["proj_dim"], g["hidden"], num_layers=g["layers"], batch_first=True,
                          bidirectional=True, dropout=g["dropout"] if g["layers"] > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(2 * g["hidden"], head_hidden), nn.ReLU(),
                                  nn.Linear(head_hidden, n_mels))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.drop(torch.relu(self.proj(x))))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def load_backbone(self, state: dict) -> None:
        """Load a pretrained input projection + GRU (GRUSSLModel.backbone_state());
        the Mel head keeps its seeded initialisation."""
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith("head.") for k in missing):
            raise RuntimeError(f"GRU backbone mismatch: missing={missing} unexpected={unexpected}")


class GRUSSLModel(nn.Module):
    """Masked neural reconstruction with the GRU baseline's projection + bidirectional
    GRU as the encoder (Phase 3). Parameter names match GRUDecoder, so the backbone
    loads into it unchanged."""

    def __init__(self, input_dim: int, g: dict):
        super().__init__()
        self.proj = nn.Linear(input_dim, g["proj_dim"])
        self.drop = nn.Dropout(g["dropout"])
        self.gru = nn.GRU(g["proj_dim"], g["hidden"], num_layers=g["layers"], batch_first=True,
                          bidirectional=True, dropout=g["dropout"] if g["layers"] > 1 else 0.0)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.reconstruction_head = nn.Linear(2 * g["hidden"], input_dim)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.drop(torch.relu(self.proj(x))))
        return h

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.where(mask, self.mask_token.expand_as(x), x)
        return self.reconstruction_head(self.features(x))

    def backbone_state(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if k.startswith(("proj.", "gru."))}
