"""
Standalone script: re-run Meta Attack + defense only, then patch the
phase45 cache so run_full_pipeline.py picks up the new result without
having to re-run the other 6 attacks.

Usage:
    python3 rerun_meta_attack.py
"""
import sys, json, time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.config import cfg
from utils.metrics import (
    classification_metrics, recovery_rate,
    neighborhood_entropy, embedding_drift,
    homophily_drop, attack_success_rate_global,
)
from utils.graph_utils import normalize_adjacency
from datasets.cora_loader import load_cora
from models.gcn import create_gcn
from models.train import predict, load_params, train_model
from attacks.meta_attack import meta_attack
from attacks.base import AttackResult
from defense.pipeline import run_defense

CACHE_FILE = ROOT / "results" / "phase45_cache.json"


def _init_params(model, graph):
    a_hat = jnp.array(normalize_adjacency(graph.adj))
    x     = jnp.array(graph.features)
    key   = jax.random.PRNGKey(0)
    return model.init({"params": key, "dropout": key}, x, a_hat, training=False)["params"]


def main():
    t0 = time.time()
    print("=" * 60)
    print("  Re-running Meta Attack only (inner_epochs=40)")
    print("=" * 60)

    # ── Load Cora + baseline model ────────────────────────────────────────────
    cora = load_cora(cfg.data_dir)
    model = create_gcn(cfg.model.hidden_dim, cora.num_classes, cfg.model.dropout_rate)

    ckpt = cfg.checkpoints_dir / "gcn_cora_baseline"
    template = _init_params(model, cora)
    params = load_params(template, str(ckpt))

    _, clean_preds, _ = predict(model, params, cora)
    baseline_m  = classification_metrics(cora.labels, np.array(clean_preds), mask=cora.test_mask)
    baseline_acc = baseline_m["accuracy"]
    print(f"\n  Baseline acc = {baseline_acc:.4f}")

    # ── Run Meta Attack with new inner_epochs=40 ─────────────────────────────
    print(f"\n  Running meta_attack (budget_ratio={cfg.attack.meta_budget_ratio}, "
          f"inner_epochs={cfg.attack.meta_inner_epochs}, n_steps={cfg.attack.meta_epochs}) ...")
    attack_result = meta_attack(
        graph=cora,
        model=model,
        params=params,
        budget_ratio=cfg.attack.meta_budget_ratio,
        n_steps=cfg.attack.meta_epochs,
        inner_epochs=cfg.attack.meta_inner_epochs,
    )
    perturbed = attack_result.perturbed_graph

    _, atk_preds, _ = predict(model, params, perturbed)
    atk_m = classification_metrics(cora.labels, np.array(atk_preds), mask=cora.test_mask)
    atk_acc = atk_m["accuracy"]
    print(f"\n  After attack: acc={atk_acc:.4f}  f1={atk_m['f1']:.4f}  "
          f"drop={baseline_acc - atk_acc:+.4f} ({(baseline_acc - atk_acc)/baseline_acc:.1%})")

    # ── Run defense ───────────────────────────────────────────────────────────
    print(f"\n  Running dual defense (GNNGUARD + Ontology) ...")

    from attacks.runner import EvaluatedAttack
    dr = run_defense(
        attacked_graph=perturbed,
        model=model,
        attack_type="poisoning",
        attacked_params=params,
        defense_cfg=cfg.defense,
        model_cfg=cfg.model,
        seed=cfg.seed,
        baseline_acc=baseline_acc,
        attacked_acc=atk_acc,
        damage_threshold=0.05,
    )

    _, gg_preds,  _ = predict(model, dr.gnnguard.defended_params,  dr.gnnguard.defended_graph)
    _, ont_preds, _ = predict(model, dr.ontology.defended_params,  dr.ontology.defended_graph)
    gg_m  = classification_metrics(cora.labels, np.array(gg_preds),  mask=cora.test_mask)
    ont_m = classification_metrics(cora.labels, np.array(ont_preds), mask=cora.test_mask)
    best_acc = max(gg_m["accuracy"], ont_m["accuracy"])
    best_m   = gg_m if gg_m["accuracy"] >= ont_m["accuracy"] else ont_m

    rr = recovery_rate(baseline_acc, atk_acc, best_acc)
    rr_str = f"{rr:.1%}" if rr is not None else "N/A (<5pp)"
    print(f"\n  After defense:  GNNGUARD={gg_m['accuracy']:.4f}  "
          f"Ontology={ont_m['accuracy']:.4f}  Best={best_acc:.4f}  recovery={rr_str}")

    # ── Save attacked graph ───────────────────────────────────────────────────
    atk_dir = cfg.results_dir / "attacked_graphs"
    atk_dir.mkdir(parents=True, exist_ok=True)
    np.savez(atk_dir / "cora_meta_attack.npz",
             adj=perturbed.adj, features=perturbed.features,
             labels=perturbed.labels,
             train_mask=perturbed.train_mask,
             val_mask=perturbed.val_mask,
             test_mask=perturbed.test_mask)

    def_dir = cfg.results_dir / "defended_graphs"
    def_dir.mkdir(parents=True, exist_ok=True)
    for dname, single in [("gnnguard", dr.gnnguard), ("ontology", dr.ontology)]:
        g = single.defended_graph
        np.savez(def_dir / f"defended_meta_attack_{dname}.npz",
                 adj=g.adj, features=g.features, labels=g.labels,
                 train_mask=g.train_mask, val_mask=g.val_mask,
                 test_mask=g.test_mask)

    # ── Advanced metrics (computed before cache patch so values are available) ──
    emb_clean_np, clean_preds, _ = predict(model, params, cora)
    emb_atk_np,   _,            _ = predict(model, params, perturbed)
    emb_clean_np   = np.array(emb_clean_np)
    emb_atk_np     = np.array(emb_atk_np)
    clean_preds_np = np.array(clean_preds)

    h_drop    = homophily_drop(cora.adj, perturbed.adj, cora.labels)
    asr_g     = attack_success_rate_global(cora.labels, clean_preds_np, np.array(atk_preds), mask=cora.test_mask)
    ent_clean = neighborhood_entropy(cora.adj,      cora.labels, cora.num_classes, cora.test_mask)
    ent_atk   = neighborhood_entropy(perturbed.adj, cora.labels, cora.num_classes, cora.test_mask)
    drift     = embedding_drift(emb_clean_np, emb_atk_np, cora.test_mask)

    print(f"\n  Advanced Metrics:")
    print(f"    ASR (global):         {asr_g:.4f}  ({asr_g:.1%} of test nodes flipped)")
    print(f"    Homophily Drop:       {h_drop:.4f}")
    print(f"    Neighborhood Entropy: {ent_clean:.4f} (clean) → {ent_atk:.4f} (attacked)  Δ={ent_atk-ent_clean:+.4f}")
    print(f"    Embedding Drift:      {drift:.4f}  (mean L2 in latent space)")

    # ── Patch cache ───────────────────────────────────────────────────────────
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())
    else:
        print("  WARNING: no cache file found — create a full run first.")
        return

    cache["attack_accs"]["meta_attack"]      = float(atk_acc)
    cache["defended_accs"]["meta_attack"]    = float(best_acc)
    cache["attack_metrics"]["meta_attack"]   = {k: float(v) for k, v in atk_m.items()}
    cache["defended_metrics"]["meta_attack"] = {k: float(v) for k, v in best_m.items()}
    cache["defended_accs_gnnguard"]["meta_attack"]  = float(gg_m["accuracy"])
    cache["defended_accs_ontology"]["meta_attack"]  = float(ont_m["accuracy"])
    if "advanced_metrics" not in cache:
        cache["advanced_metrics"] = {}
    cache["advanced_metrics"]["meta_attack"] = {
        "homophily_drop":    float(h_drop),
        "asr_global":        float(asr_g),
        "entropy_clean":     float(ent_clean),
        "entropy_attacked":  float(ent_atk),
        "entropy_delta":     float(ent_atk - ent_clean),
        "embedding_drift":   float(drift),
    }

    CACHE_FILE.write_text(json.dumps(cache, indent=2))
    print(f"\n  Cache patched → {CACHE_FILE}")

    elapsed = int(time.time() - t0)
    print(f"\n  Done in {elapsed//60}m {elapsed%60}s")
    print(f"\n  Summary:  baseline={baseline_acc:.4f}  "
          f"attack={atk_acc:.4f} (drop={baseline_acc-atk_acc:+.4f})  "
          f"defense={best_acc:.4f} (recovery={rr_str})")


if __name__ == "__main__":
    main()
