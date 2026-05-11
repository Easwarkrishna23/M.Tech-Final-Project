"""
Training and evaluation loop for GCN / GAT.

Design:
  - All state lives in Flax TrainState (params + opt_state)
  - train_step / eval_step are JIT-compiled pure functions
  - Early stopping on validation accuracy
  - Returns best checkpoint (highest val accuracy)
"""
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax.training import train_state
from flax import linen as nn
from functools import partial
from typing import Any, Dict, Tuple, NamedTuple

from datasets.cora_loader import GraphData
from utils.config import ModelConfig


# ──────────────────────────────────────────────────────────────────────────────
# TrainState
# ──────────────────────────────────────────────────────────────────────────────

class GNNTrainState(train_state.TrainState):
    """TrainState extended with a dropout RNG key."""
    dropout_key: jax.Array


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray,
                       mask: jnp.ndarray) -> jnp.ndarray:
    """
    Masked cross-entropy: only compute loss on nodes where mask=True.
    Handles unknown labels (-1) by combining with the mask.
    """
    valid = mask & (labels >= 0)
    n_valid = valid.sum()

    log_probs = jax.nn.log_softmax(logits, axis=-1)            # [N, C]
    # gather log-prob of the true class for each valid node
    true_log_probs = log_probs[jnp.arange(logits.shape[0]),
                                jnp.where(labels >= 0, labels, 0)]
    loss = -jnp.where(valid, true_log_probs, 0.0).sum()
    return loss / jnp.maximum(n_valid, 1)


# ──────────────────────────────────────────────────────────────────────────────
# JIT-compiled steps
# ──────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("model",))
def train_step(
    state: GNNTrainState,
    model: nn.Module,
    x: jnp.ndarray,
    a_hat: jnp.ndarray,
    labels: jnp.ndarray,
    train_mask: jnp.ndarray,
) -> Tuple[GNNTrainState, jnp.ndarray]:
    """Single gradient update step."""
    dropout_key, new_key = jax.random.split(state.dropout_key)

    def loss_fn(params):
        _, logits, _ = model.apply(
            {"params": params}, x, a_hat, training=True,
            rngs={"dropout": dropout_key},
        )
        return cross_entropy_loss(logits, labels, train_mask)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(dropout_key=new_key)
    return state, loss


@partial(jax.jit, static_argnames=("model",))
def eval_step(
    params: Any,
    model: nn.Module,
    x: jnp.ndarray,
    a_hat: jnp.ndarray,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Forward pass (no dropout). Returns loss, accuracy, embeddings, predictions."""
    embeddings, logits, probs = model.apply(
        {"params": params}, x, a_hat, training=False,
    )
    loss = cross_entropy_loss(logits, labels, mask)
    preds = jnp.argmax(logits, axis=-1)
    valid = mask & (labels >= 0)
    correct = jnp.where(valid, preds == labels, 0).sum()
    acc = correct / jnp.maximum(valid.sum(), 1)
    return loss, acc, embeddings, preds


# ──────────────────────────────────────────────────────────────────────────────
# Full training loop
# ──────────────────────────────────────────────────────────────────────────────

class TrainResult(NamedTuple):
    best_params: Any
    train_losses: list
    val_accs: list
    best_val_acc: float
    best_epoch: int


def train_model(
    model: nn.Module,
    graph: GraphData,
    cfg: ModelConfig,
    seed: int = 42,
    verbose: bool = True,
) -> TrainResult:
    """
    Train GCN/GAT on a single graph with early stopping.

    Args:
        model:  Flax module (GCN or GAT).
        graph:  GraphData — uses adj_norm for GCN, adj for GAT.
        cfg:    ModelConfig with lr, epochs, patience, etc.
        seed:   RNG seed.
        verbose: Print per-epoch progress.

    Returns:
        TrainResult with best params and training history.
    """
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)

    # Convert to JAX arrays once
    x     = jnp.array(graph.features)
    a_hat = jnp.array(graph.adj_norm)
    adj   = jnp.array(graph.adj)
    labels = jnp.array(graph.labels)
    train_mask = jnp.array(graph.train_mask)
    val_mask   = jnp.array(graph.val_mask)

    # Detect model type: GCN uses a_hat, GAT uses raw adj
    from models.gat import GAT
    use_adj = isinstance(model, GAT)
    graph_input = adj if use_adj else a_hat

    # Initialise params
    params = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        x, graph_input, training=False,
    )["params"]

    # Optimizer: AdamW with weight decay
    tx = optax.adamw(learning_rate=cfg.learning_rate,
                     weight_decay=cfg.weight_decay)

    state = GNNTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dropout_key=dropout_rng,
    )

    best_val_acc = 0.0
    best_params = params
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_accs = []

    for epoch in range(1, cfg.epochs + 1):
        state, loss = train_step(state, model, x, graph_input, labels, train_mask)
        _, val_acc, _, _ = eval_step(state.params, model, x, graph_input,
                                      labels, val_mask)

        train_losses.append(float(loss))
        val_accs.append(float(val_acc))

        if float(val_acc) > best_val_acc:
            best_val_acc = float(val_acc)
            best_params = state.params
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and epoch % 50 == 0:
            print(f"  Epoch {epoch:>3d}/{cfg.epochs} | "
                  f"loss={float(loss):.4f} | val_acc={float(val_acc):.4f} | "
                  f"best={best_val_acc:.4f} @ ep{best_epoch}")

        if patience_counter >= cfg.patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {cfg.patience} epochs)")
            break

    return TrainResult(
        best_params=best_params,
        train_losses=train_losses,
        val_accs=val_accs,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Prediction helper
# ──────────────────────────────────────────────────────────────────────────────

def predict(model: nn.Module, params: Any,
            graph: GraphData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on a graph. Returns (embeddings, predictions, probabilities).
    All outputs as NumPy arrays for compatibility with sklearn metrics.
    """
    from models.gat import GAT
    x     = jnp.array(graph.features)
    g_in  = jnp.array(graph.adj if isinstance(model, GAT) else graph.adj_norm)

    embeddings, logits, probs = model.apply(
        {"params": params}, x, g_in, training=False,
    )
    preds = jnp.argmax(logits, axis=-1)
    return np.array(embeddings), np.array(preds), np.array(probs)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers (simple numpy-based, no orbax dependency)
# ──────────────────────────────────────────────────────────────────────────────

def save_params(params: Any, path: str) -> None:
    """Flatten params pytree to numpy dict and save as .npz."""
    import os
    flat = jax.tree_util.tree_leaves(params)
    keys = [str(i) for i in range(len(flat))]
    np.savez(path, **{k: np.array(v) for k, v in zip(keys, flat)})
    print(f"[Checkpoint] Saved params → {path}.npz")


def load_params(params_template: Any, path: str) -> Any:
    """Reload params saved with save_params."""
    data = np.load(f"{path}.npz")
    flat_loaded = [jnp.array(data[str(i)]) for i in range(len(data.files))]
    treedef = jax.tree_util.tree_structure(params_template)
    return jax.tree_util.tree_unflatten(treedef, flat_loaded)
