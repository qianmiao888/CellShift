from __future__ import annotations

import torch
from torch import nn


EMBEDDING_DIMS = {
    "Strains": 16,
    "perturbation_no_concentration": 32,
    "Medium": 4,
    "Temperature": 4,
    "data_source": 8,
    "instrument": 8,
    "Yeast_cell_plate": 16,
}


def make_mlp(input_dim: int, output_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class LowRankConditionModel(nn.Module):
    def __init__(
        self,
        cardinalities: dict[str, int],
        protein_mean: torch.Tensor,
        decoder_initialization: torch.Tensor,
        architecture: str = "residual",
        hidden_dim: int = 256,
        dropout: float = 0.10,
        category_dropout: float = 0.15,
        use_batch_calibration: bool = True,
        initialize_decoder: bool = True,
    ) -> None:
        super().__init__()
        if architecture not in {"monolithic", "residual"}:
            raise ValueError(f"Unsupported architecture: {architecture}")
        self.architecture = architecture
        self.category_dropout = category_dropout
        self.use_batch_calibration = use_batch_calibration
        self.embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(cardinality, EMBEDDING_DIMS[field])
                for field, cardinality in cardinalities.items()
            }
        )
        latent_dim = int(decoder_initialization.shape[0])
        context_dim = EMBEDDING_DIMS["Medium"] + EMBEDDING_DIMS["Temperature"] + 4
        strain_dim = EMBEDDING_DIMS["Strains"]
        chemical_dim = EMBEDDING_DIMS["perturbation_no_concentration"]
        observation_dim = (
            EMBEDDING_DIMS["data_source"]
            + EMBEDDING_DIMS["instrument"]
            + EMBEDDING_DIMS["Yeast_cell_plate"]
        )

        if architecture == "monolithic":
            monolithic_dim = context_dim + strain_dim + chemical_dim + 1
            self.monolithic_head = make_mlp(monolithic_dim, latent_dim, hidden_dim, dropout)
        else:
            self.context_head = make_mlp(context_dim, latent_dim, hidden_dim, dropout)
            self.strain_head = make_mlp(context_dim + strain_dim, latent_dim, hidden_dim, dropout)
            self.shared_treatment_head = make_mlp(context_dim, latent_dim, hidden_dim, dropout)
            self.chemical_head = make_mlp(context_dim + chemical_dim, latent_dim, hidden_dim, dropout)
            self.interaction_head = make_mlp(
                context_dim + strain_dim + chemical_dim,
                latent_dim,
                hidden_dim,
                dropout,
            )

        if use_batch_calibration:
            self.batch_head = make_mlp(observation_dim, latent_dim, hidden_dim // 2, dropout)

        self.decoder = nn.Linear(latent_dim, len(protein_mean), bias=False)
        if initialize_decoder:
            with torch.no_grad():
                self.decoder.weight.copy_(decoder_initialization.T)
        self.register_buffer("protein_mean", protein_mean.clone())

    def _category_dropout(self, values: torch.Tensor, field: str) -> torch.Tensor:
        if not self.training or self.category_dropout <= 0:
            return values
        probability = self.category_dropout
        if field == "Yeast_cell_plate":
            probability = min(0.50, probability * 1.5)
        drop = torch.rand(values.shape, device=values.device).lt(probability)
        return values.masked_fill(drop, 0)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embedded: dict[str, torch.Tensor] = {}
        for field, embedding in self.embeddings.items():
            indices = self._category_dropout(batch[field], field)
            embedded[field] = embedding(indices)

        context = torch.cat(
            [embedded["Medium"], embedded["Temperature"], batch["time_features"]], dim=1
        )
        strain = embedded["Strains"]
        chemical = embedded["perturbation_no_concentration"]
        treatment = batch["is_treatment"]

        components: dict[str, torch.Tensor] = {}
        if self.architecture == "monolithic":
            components["monolithic"] = self.monolithic_head(
                torch.cat([context, strain, chemical, treatment], dim=1)
            )
        else:
            components["context"] = self.context_head(context)
            components["strain"] = self.strain_head(torch.cat([context, strain], dim=1))
            components["shared_treatment"] = treatment * self.shared_treatment_head(context)
            components["chemical"] = treatment * self.chemical_head(
                torch.cat([context, chemical], dim=1)
            )
            components["interaction"] = treatment * self.interaction_head(
                torch.cat([context, strain, chemical], dim=1)
            )

        if self.use_batch_calibration:
            observation = torch.cat(
                [embedded["data_source"], embedded["instrument"], embedded["Yeast_cell_plate"]],
                dim=1,
            )
            components["batch"] = self.batch_head(observation)

        latent = torch.stack(list(components.values()), dim=0).sum(dim=0)
        prediction = self.protein_mean[None, :] + self.decoder(latent)
        return prediction, components
