"""FROZEN — stochastic Feynman-Kac viscous geometric regularization loss.

Do NOT modify the body of `stochastic_fk_loss`. Wire-up adapters and config
plumbing live in `agents/sharsa_geodesic.py`.

The function is written as a free function that takes `self` (any object that
exposes `.network.select(name)` and `.config.get(...)`) so we can call it from
`SHARSAGeodesicAgent` via a small proxy without touching the function itself.
"""
import jax
import jax.numpy as jnp


def stochastic_fk_loss(self, batch, grad_params, key):
    """
    Calculates the Viscous Geometric Regularization via Fixed-Point Iteration.
    Implements the update: V(s) <- lambda * log( E[ exp(V(s')/lambda) ] )

    This minimizes || V(s) - T_nu V(s) ||^2, ensuring V(s) evolves according
    to the viscous HJB operator without 'collusion' from neighbors.
    """
    # 1. Hyperparameters
    nu = self.config.get('viscous_scale', 0.01)
    # To satisfy the Taylor expansion V + nu*Laplacian + 0.5*|grad|^2:
    # We need sigma^2 = 2*nu and lambda = sigma^2 (so lambda = 2*nu).
    # Adjust per your specific PDE derivation if different.
    #lambda_temp = self.config.get('temperature', 2.0 * nu)
    lambda_temp = 1.0
    sigma = jnp.sqrt(2.0 * nu)
    K = self.config.get('num_walks', 10)
    kappa = 0.1

    # 2. Setup Inputs
    obs = batch['observations']
    local_speed = batch['speed']
    if local_speed.ndim == 1: local_speed = local_speed[:, None]
    goal_reps = self.network.select('rep_value')(batch['value_goals'])
    B, D = obs.shape
    G = goal_reps.shape[-1]

    # 3. Get V(s) [Anchor]
    v_out = self.network.select('value')(obs, goal_reps, params=grad_params)


    # Ensemble handling
    if isinstance(v_out, tuple):
        v_s = (v_out[0] + v_out[1]) / 2.0
    elif hasattr(v_out, 'shape') and v_out.shape[0] == 2 and v_out.ndim > 1:
        v_s = jnp.mean(v_out, axis=0)
    else:
        v_s = v_out

    if v_s.ndim == 1: v_s = v_s[:, None] # (B, 1)

    # 4. Generate Neighbors (s + evlon)
    obs_expanded = jnp.repeat(obs[:, None, :], K, axis=1)
    goal_expanded = jnp.repeat(goal_reps[:, None, :], K, axis=1)

    noise = jax.random.normal(key, shape=obs_expanded.shape) * sigma
    noisy_obs = obs_expanded + noise

    flat_noisy_obs = noisy_obs.reshape(B * K, D)
    flat_goal = goal_expanded.reshape(B * K, G)

    # 5. Get V(s + epsilon) using Current Params
    # We use the same params, but strictly for target calculation
    v_neighbors_out = self.network.select('value')(flat_noisy_obs, flat_goal, params=grad_params)

    if isinstance(v_neighbors_out, tuple):
        v_neighbors_flat = (v_neighbors_out[0] + v_neighbors_out[1]) / 2.0
    elif hasattr(v_neighbors_out, 'shape') and v_neighbors_out.shape[0] == 2 and v_neighbors_out.ndim > 1:
        v_neighbors_flat = jnp.mean(v_neighbors_out, axis=0)
    else:
        v_neighbors_flat = v_neighbors_out

    v_neighbors = v_neighbors_flat.reshape(B, K) # (B, K)


    dist_to_neighbors = jnp.linalg.norm(noise, axis=-1)
    dist_to_neighbors = jax.lax.stop_gradient(dist_to_neighbors) # Treat as fixed constant
    greens_contribution = 1 / (dist_to_neighbors + 1e-6)

    # Absolute difference in value
    abs_diff = jnp.abs(v_neighbors - v_s)#/jnp.linalg.norm(v_s + 1e-6)
    const_v = jax.lax.stop_gradient(v_s)

    # Estimate the denominator using Triangle Inequality \v
    cost_to_neighbor = abs_diff * greens_contribution

    #cost_to_neighbor = abs_diff

    # 2. Compute Allowed Slope (Riemannian Speed Limit)
    # Near walls (speed ~ 0.1) -> Limit ~ 10.0 (Steep Cliff Allowed)
    # Open space (speed ~ 1.0) -> Limit ~ 1.0 (Smooth Slope enforced)
    q_s = kappa / (local_speed + 1e-6)

    delta_t = 1.0

    metric_residual = jnp.maximum(0.0, cost_to_neighbor - q_s * delta_t)
    loss_metric = jnp.square(metric_residual).mean()
    use_metric = self.config.get('enable_viscous_metric', True)
    use_metric_only = self.config.get('use_metric_only', False)

    # --- Combine ---
    total_loss = loss_metric

    return total_loss, {
        'fk_loss': loss_metric,
        'v_mean': v_s.mean(),
        'boundary_loss': loss_metric.mean(),
        'avg_speed': local_speed.mean(),
        'avg_grad_est': cost_to_neighbor.mean()
    }
