"""Dual goal representation modules for SHARSA + dual paper loss (option b).

Ported from dual-goal-viscosity/utils/dual.py and adapted to the upstream
horizon-reduction `utils.networks.GCBilinearValue` signature, which uses
`num_ensembles: int` instead of `ensemble: bool` + `ret_mean`.

Usage: `rep_value(g)` returns the goal embedding psi(g) of shape (B, latent_dim);
       `rep_value(s, g)` returns the scalar value V(s, g) (ensemble-averaged).

Only the bilinear variant is ported because the four-method eval matrix uses
rep_type='bilinear' (default in the dual paper's run scripts). MRN/IQE/Hilbert
variants can be added on demand.
"""
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import GCBilinearValue


class GCBilinearRepresentationValue(nn.Module):
    """Bilinear V(s, g) = phi(s)^T psi(g) / sqrt(d), with a goal-encoder branch.

    When called with both observations and goals, returns the (ensemble-averaged)
    scalar value. When called with one positional arg, treats it as a batch of
    goals to encode and returns psi(g).
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    num_ensembles: int = 2

    def setup(self):
        self.network = GCBilinearValue(
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            layer_norm=self.layer_norm,
            num_ensembles=self.num_ensembles,
        )

    def __call__(self, observations, goals=None):
        if goals is not None:
            v = self.network(observations, goals, actions=None, info=False)
            if v.ndim > 1 and v.shape[0] == self.num_ensembles:
                v = v.mean(axis=0)
            return v
        dummy = jnp.zeros_like(observations)
        _, _, psi = self.network(dummy, observations, actions=None, info=True)
        if psi.ndim == 3 and psi.shape[0] == self.num_ensembles:
            psi = psi.mean(axis=0)
        return psi


def DualRepresentationValue(type):
    """Dispatch helper for the rep_type config key."""
    if type == 'bilinear':
        return GCBilinearRepresentationValue
    raise NotImplementedError(
        f"rep_type={type!r} not ported yet. The four-method matrix uses 'bilinear'."
    )
