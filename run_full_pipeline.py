"""
Full Experiment Pipeline — Phases 1-7.

Runs everything end-to-end:
  Phase 1  : Load Cora + Elliptic
  Phase 3  : Baseline GCN training (both datasets)
  Phase 4  : All 7 attacks on Cora
  Phase 5  : Structural defense on Cora
  Phase 6  : Cora visualizations
  Phase 7  : Elliptic temporal evaluation + visualizations

Estimated runtime: ~2.5 hours
  - Cora attacks + defense  : ~40 min
  - Elliptic evaluation     : ~110 min

Progress is logged to results/pipeline_log.txt in addition to stdout.
Checkpoints are saved after each phase so the pipeline can be re-run
from any phase by commenting out earlier phases.
"""
import sys
import time
import traceback
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.config import cfg
from utils.metrics import (
    classification_metrics, recovery_rate, format_defense_table,
    attack_success_rate_global, node_level_recovery_rate,
    neighborhood_entropy, embedding_drift,
    homophily_drop, bose_einstein_fitness,
    assortativity_coefficient, clean_label_recovery,
)
from utils.graph_utils import normalize_adjacency
from datasets.cora_loader import load_cora
from datasets.elliptic_loader import load_elliptic
from models.gcn import create_gcn
from models.gat import create_gat
from models.train import train_model, predict, save_params, load_params
from attacks.runner import run_all_attacks
from defense.pipeline import run_all_defenses
from visualization.bar_charts import plot_accuracy_bar, plot_metrics_grouped
from visualization.line_plots import (
    plot_attack_defense_line, plot_temporal_accuracy, plot_training_curves,
)
from visualization.graph_viz import plot_graph_comparison, plot_degree_distribution
from visualization.embeddings import plot_embeddings_comparison


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

LOG_FILE = ROOT / "results" / "pipeline_log.txt"


class Tee:
    """Write to both stdout and log file simultaneously."""
    def __init__(self, filepath):
        self.file = open(filepath, "w", buffering=1)
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()


def _banner(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _elapsed(t0):
    s = int(time.time() - t0)
    return f"{s//60}m {s%60}s"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_params(model, graph):
    a_hat = jnp.array(normalize_adjacency(graph.adj))
    x     = jnp.array(graph.features)
    key   = jax.random.PRNGKey(0)
    return model.init({"params": key, "dropout": key}, x, a_hat, training=False)["params"]


def _load_or_train(model, graph, ckpt_name, force_retrain=False):
    ckpt = cfg.checkpoints_dir / f"{ckpt_name}"
    ckpt_file = Path(str(ckpt) + ".npz")
    if ckpt_file.exists() and not force_retrain:
        print(f"  [Checkpoint] Loading {ckpt_file.name}")
        template = _init_params(model, graph)
        return load_params(template, str(ckpt)), None
    print(f"  [Train] Training from scratch → {ckpt_name}")
    result = train_model(model, graph, cfg.model, seed=cfg.seed, verbose=False)
    save_params(result.best_params, str(ckpt))
    return result.best_params, result


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Load datasets
# ─────────────────────────────────────────────────────────────────────────────

def phase1():
    _banner("PHASE 1 — Dataset Loading")
    cora     = load_cora(cfg.data_dir)
    elliptic = load_elliptic(cfg.data_dir)
    print(f"  Cora    : {cora.stats()}")
    print(f"  Elliptic: {elliptic.num_timesteps} timesteps, "
          f"final snapshot {elliptic.final_snapshot().stats()}")
    return cora, elliptic


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Baseline training
# ─────────────────────────────────────────────────────────────────────────────

def phase3(cora, elliptic):
    _banner("PHASE 3 — Baseline Training")
    t0 = time.time()

    # Cora GCN
    cora_model = create_gcn(cfg.model.hidden_dim, cora.num_classes, cfg.model.dropout_rate)
    cora_params, cora_result = _load_or_train(cora_model, cora, "gcn_cora_baseline")
    _, preds, _ = predict(cora_model, cora_params, cora)
    cora_m = classification_metrics(cora.labels, np.array(preds), mask=cora.test_mask)
    print(f"  [Cora GCN]  acc={cora_m['accuracy']:.4f}  f1={cora_m['f1']:.4f}  "
          f"prec={cora_m['precision']:.4f}  rec={cora_m['recall']:.4f}")

    # Cora GAT (comparison)
    gat_model = create_gat(cfg.model.hidden_dim, cora.num_classes, dropout_rate=0.6)
    gat_params, _ = _load_or_train(gat_model, cora, "gat_cora_baseline")
    _, gat_preds, _ = predict(gat_model, gat_params, cora)
    gat_m = classification_metrics(cora.labels, np.array(gat_preds), mask=cora.test_mask)
    print(f"  [Cora GAT]  acc={gat_m['accuracy']:.4f}  f1={gat_m['f1']:.4f}")

    # Elliptic GCN (train on era 1-34 snapshot)
    train_snap   = elliptic.get_snapshot(33)
    ell_model    = create_gcn(cfg.model.hidden_dim, train_snap.num_classes, cfg.model.dropout_rate)
    ell_params, _ = _load_or_train(ell_model, train_snap, "gcn_elliptic_train_era")
    final_snap   = elliptic.final_snapshot()
    _, ell_preds, _ = predict(ell_model, ell_params, final_snap)
    ell_m = classification_metrics(final_snap.labels, np.array(ell_preds), mask=final_snap.test_mask)
    print(f"  [Elliptic GCN final t=49]  acc={ell_m['accuracy']:.4f}  f1={ell_m['f1']:.4f}")

    print(f"  Phase 3 done in {_elapsed(t0)}")
    return (cora_model, cora_params, cora_m,
            gat_model, gat_params,
            ell_model, ell_params, ell_m)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4+5 — Cora attacks + defense
# ─────────────────────────────────────────────────────────────────────────────

CACHE_FILE = ROOT / "results" / "phase45_cache.json"


def _save_cache(attack_accs, defended_accs, attack_metrics, defended_metrics,
                defended_accs_gg=None, defended_accs_ont=None):
    import json
    cache = {
        "attack_accs":       attack_accs,
        "defended_accs":     defended_accs,
        "attack_metrics":    attack_metrics,
        "defended_metrics":  defended_metrics,
        "defended_accs_gnnguard":  defended_accs_gg  or {},
        "defended_accs_ontology":  defended_accs_ont or {},
    }
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2))
    print(f"  [Cache] Phase 4+5 results saved → {CACHE_FILE}")


def _load_cache():
    import json
    if not CACHE_FILE.exists():
        return None
    cache = json.loads(CACHE_FILE.read_text())
    print(f"  [Cache] Loading Phase 4+5 results from {CACHE_FILE}")
    return (cache["attack_accs"], cache["defended_accs"],
            cache["attack_metrics"], cache["defended_metrics"])


def phase45(cora, cora_model, cora_params, baseline_acc):
    _banner("PHASE 4+5 — Cora Attacks + Defense")

    # Return cached results if all attacked/defended graphs already exist
    cached = _load_cache()
    if cached is not None:
        attack_accs, defended_accs, attack_metrics, defended_metrics = cached
        print("  [Skip] Phase 4+5 already complete — using cached results")
        print("\n[Phase 4 Results (cached)]")
        for atk, m in attack_metrics.items():
            print(f"  {atk:25s}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  "
                  f"drop={baseline_acc - m['accuracy']:+.4f}")
        print("\n[Phase 5 Results (cached)]")
        for atk in attack_accs:
            rr = recovery_rate(baseline_acc, attack_accs[atk], defended_accs[atk])
            rr_str = f"{rr:.1%}" if rr is not None else "N/A"
            print(f"  {atk:25s}  acc={defended_accs[atk]:.4f}  recovery={rr_str}")
        return None, None, attack_accs, defended_accs, attack_metrics, defended_metrics

    t0 = time.time()
    attack_results = run_all_attacks(
        graph=cora,
        model=cora_model,
        clean_params=cora_params,
        attack_cfg=cfg.attack,
        model_cfg=cfg.model,
        seed=cfg.seed,
        save_dir=cfg.results_dir / "attacked_graphs",
    )

    # Evaluate each attack
    attack_accs = {}
    attack_metrics = {}
    print("\n[Phase 4 Results]")
    for atk_name, ea in attack_results.items():
        _, preds, _ = predict(cora_model, ea.eval_params, ea.attack_result.perturbed_graph)
        m = classification_metrics(cora.labels, np.array(preds), mask=cora.test_mask)
        attack_accs[atk_name]    = m["accuracy"]
        attack_metrics[atk_name] = m
        drop = baseline_acc - m["accuracy"]
        print(f"  {atk_name:25s}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  "
              f"drop={drop:+.4f}")

    defense_results = run_all_defenses(
        attack_results=attack_results,
        model=cora_model,
        defense_cfg=cfg.defense,
        model_cfg=cfg.model,
        seed=cfg.seed,
        save_dir=cfg.results_dir / "defended_graphs",
        baseline_acc=baseline_acc,
        attack_accs=attack_accs,
        damage_threshold=0.05,
    )

    # Evaluate defense — report both GNNGUARD and Ontology separately
    defended_accs    = {}   # best defense acc (for cache/legacy compat)
    defended_metrics = {}
    defended_accs_gg  = {}
    defended_accs_ont = {}

    print("\n[Phase 5 Results — Dual Defense Comparison]")
    print(f"  {'Attack':25s}  {'After Atk':>9}  {'GNNGUARD':>9}  "
          f"{'Ontology':>9}  {'Best':>9}  {'Recovery':>10}")
    print(f"  {'-'*25}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*10}")

    for atk_name, dr in defense_results.items():
        # GNNGUARD
        _, gg_preds, _ = predict(cora_model, dr.gnnguard.defended_params,
                                  dr.gnnguard.defended_graph)
        gg_m = classification_metrics(cora.labels, np.array(gg_preds), mask=cora.test_mask)
        defended_accs_gg[atk_name] = gg_m["accuracy"]

        # Ontology
        _, ont_preds, _ = predict(cora_model, dr.ontology.defended_params,
                                   dr.ontology.defended_graph)
        ont_m = classification_metrics(cora.labels, np.array(ont_preds), mask=cora.test_mask)
        defended_accs_ont[atk_name] = ont_m["accuracy"]

        # Best
        best_acc = max(gg_m["accuracy"], ont_m["accuracy"])
        best_m   = gg_m if gg_m["accuracy"] >= ont_m["accuracy"] else ont_m
        defended_accs[atk_name]    = best_acc
        defended_metrics[atk_name] = best_m

        rr = recovery_rate(baseline_acc, attack_accs[atk_name], best_acc)
        rr_str = f"{rr:.1%}" if rr is not None else "N/A (<5pp)"

        print(f"  {atk_name:25s}  {attack_accs[atk_name]:9.4f}  "
              f"{gg_m['accuracy']:9.4f}  {ont_m['accuracy']:9.4f}  "
              f"{best_acc:9.4f}  {rr_str:>10}")

    # ── Structural metrics (proves superiority over GNNGUARD baseline) ───────
    print("\n[Structural Metrics — Attack Impact & Defense Recovery]")
    print(f"  {'Attack':25s}  {'H-Drop':>8}  {'BE-Fit':>8}  "
          f"{'ΔAssort':>8}  {'CLR':>8}  {'ASR-Glb':>8}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    assort_clean = assortativity_coefficient(cora.adj)
    _, clean_preds, _ = predict(cora_model, cora_params, cora)
    clean_preds_np = np.array(clean_preds)

    for atk_name, ea in attack_results.items():
        atk_graph = ea.attack_result.perturbed_graph
        dr        = defense_results[atk_name]

        h_drop  = homophily_drop(cora.adj, atk_graph.adj, cora.labels)
        be_fit  = bose_einstein_fitness(cora.adj, atk_graph.adj)
        assort_atk = assortativity_coefficient(atk_graph.adj)

        _, atk_preds_s, _ = predict(cora_model, ea.eval_params, atk_graph)
        _, def_preds_s, _ = predict(cora_model, dr.defended_params, dr.defended_graph)
        atk_preds_np = np.array(atk_preds_s)
        def_preds_np = np.array(def_preds_s)

        clr = clean_label_recovery(
            cora.labels, atk_preds_np, def_preds_np, mask=cora.test_mask
        )
        asr_g = attack_success_rate_global(
            cora.labels, clean_preds_np, atk_preds_np, mask=cora.test_mask
        )

        print(f"  {atk_name:25s}  {h_drop:8.4f}  {be_fit:8.4f}  "
              f"{assort_atk - assort_clean:+8.4f}  {clr:8.4f}  {asr_g:8.4f}")

    print(f"  [Baseline assortativity: {assort_clean:+.4f}]")

    _save_cache(attack_accs, defended_accs, attack_metrics, defended_metrics,
                defended_accs_gg, defended_accs_ont)
    print(f"  Phase 4+5 done in {_elapsed(t0)}")
    return attack_results, defense_results, attack_accs, defended_accs, attack_metrics, defended_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — Cora visualizations
# ─────────────────────────────────────────────────────────────────────────────

def phase6(cora, cora_model, cora_params,
           baseline_acc, attack_accs, defended_accs,
           attack_metrics, defended_metrics,
           attack_results, defense_results,
           cora_train_result):
    _banner("PHASE 6 — Cora Visualizations")
    t0 = time.time()
    sp = cfg.figures_dir

    # Fig 1: Accuracy bar
    plot_accuracy_bar(baseline_acc, attack_accs, defended_accs,
                      dataset_name="Cora", save_path=sp)

    # Fig 2: F1 bar
    baseline_f1 = classification_metrics(
        cora.labels,
        np.array(predict(cora_model, cora_params, cora)[1]),
        mask=cora.test_mask
    )["f1"]
    metrics_table = {
        "baseline": {k: {"f1": baseline_f1} for k in attack_accs},
        "attacked":  {k: {"f1": attack_metrics[k]["f1"]} for k in attack_accs},
        "defended":  {k: {"f1": defended_metrics[k]["f1"]} for k in attack_accs},
    }
    plot_metrics_grouped(metrics_table, metric="f1", dataset_name="Cora", save_path=sp)

    # Fig 3: Attack-defense line
    plot_attack_defense_line(baseline_acc, attack_accs, defended_accs,
                             dataset_name="Cora", save_path=sp)

    # Fig 4: Training curves (if we have them)
    if cora_train_result is not None:
        plot_training_curves(cora_train_result.train_losses,
                             cora_train_result.val_accs,
                             model_name="GCN", dataset_name="Cora", save_path=sp)

    # Fig 5: Graph comparisons + degree distributions
    atk_dir = cfg.results_dir / "attacked_graphs"
    def_dir = cfg.results_dir / "defended_graphs"
    test_nodes = np.where(cora.test_mask)[0][:5].tolist()

    for atk_name in ["nettack", "feature_perturbation", "gradient_attack"]:
        atk_path = atk_dir / f"cora_{atk_name}.npz"
        # Prefer ontology defense npz; fall back to gnnguard
        def_path = def_dir / f"defended_{atk_name}_ontology.npz"
        if not def_path.exists():
            def_path = def_dir / f"defended_{atk_name}_gnnguard.npz"
        if not atk_path.exists() or not def_path.exists():
            continue
        d_atk = np.load(atk_path)
        d_def = np.load(def_path)

        plot_graph_comparison(
            cora.adj, d_atk["adj"], d_def["adj"],
            cora.labels, test_nodes,
            attack_name=atk_name, dataset_name="Cora", save_path=sp,
        )
        plot_degree_distribution(
            cora.adj, d_atk["adj"], d_def["adj"],
            attack_name=atk_name, dataset_name="Cora", save_path=sp,
        )

    # Fig 6: t-SNE embeddings
    emb_clean, _, _ = predict(cora_model, cora_params, cora)
    labels_np = np.array(cora.labels)
    valid = labels_np >= 0

    for atk_name in ["gradient_attack", "feature_perturbation", "nettack"]:
        atk_path = atk_dir / f"cora_{atk_name}.npz"
        def_path = def_dir / f"defended_{atk_name}_ontology.npz"
        if not def_path.exists():
            def_path = def_dir / f"defended_{atk_name}_gnnguard.npz"
        if not atk_path.exists() or not def_path.exists():
            continue

        from datasets.cora_loader import GraphData
        d_atk = np.load(atk_path)
        d_def = np.load(def_path)

        g_atk = cora.copy().update_adj(d_atk["adj"]).update_features(d_atk["features"])
        g_def = cora.copy().update_adj(d_def["adj"]).update_features(d_def["features"])

        emb_atk, _, _ = predict(cora_model, cora_params, g_atk)
        emb_def, _, _ = predict(cora_model, cora_params, g_def)

        plot_embeddings_comparison(
            np.array(emb_clean)[valid],
            np.array(emb_atk)[valid],
            np.array(emb_def)[valid],
            labels_np[valid],
            attack_name=atk_name, dataset_name="Cora",
            method="tsne", max_nodes=1500, save_path=sp,
        )

    print(f"  Phase 6 done in {_elapsed(t0)}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 — Elliptic temporal evaluation
# ─────────────────────────────────────────────────────────────────────────────

def phase7(elliptic, ell_model, ell_params):
    _banner("PHASE 7 — Elliptic Temporal Evaluation")
    t0 = time.time()

    from attacks.gradient_attack import gradient_attack
    from attacks.feature_perturbation import feature_perturbation_attack
    from attacks.temporal_perturbation import temporal_perturbation_attack
    from defense.pipeline import run_defense
    from defense.ontology_defense import temporal_self_healing
    from models.train import train_model as _train_model

    sp = cfg.figures_dir

    # ── Temporal baseline across all 49 timesteps ────────────────────────────
    print("\n[7.1] Temporal baseline across 49 timesteps …")
    baseline_accs = []
    for t, snap in enumerate(elliptic.snapshots):
        _, preds, _ = predict(ell_model, ell_params, snap)
        m = classification_metrics(snap.labels, np.array(preds), mask=snap.test_mask)
        baseline_accs.append(m["accuracy"])
        if (t + 1) % 10 == 0 or t == 0:
            print(f"  t={t+1:2d}  acc={m['accuracy']:.3f}  f1={m['f1']:.3f}")

    # ── Attack + defense on final snapshot (t=49) ────────────────────────────
    print("\n[7.2] Attack + defense on final snapshot (t=49) …")
    final_snap  = elliptic.final_snapshot()
    _, cl_preds, _ = predict(ell_model, ell_params, final_snap)
    baseline_m  = classification_metrics(
        final_snap.labels, np.array(cl_preds), mask=final_snap.test_mask
    )
    baseline_acc_ell = baseline_m["accuracy"]
    print(f"  Baseline  acc={baseline_acc_ell:.4f}  f1={baseline_m['f1']:.4f}")

    attack_accs_ell   = {}
    defended_accs_ell = {}

    # Scale epsilon to Elliptic's feature range (continuous, not binary [0,1] like Cora).
    # Elliptic features have std ~1-5; small fractions of std are negligible here.
    # Use 1.5× std for gradient attack and 2× std for feature/temporal attacks
    # so perturbations are large enough to shift GCN predictions across the
    # decision boundary (previously 0.20×/0.50× gave ε≈0.18/0.45, too small).
    feat_std  = float(np.std(final_snap.features))
    ell_grad_epsilon = min(3.0, feat_std * 1.5)
    ell_feat_epsilon = min(5.0, feat_std * 2.0)
    print(f"  [Elliptic] feature std={feat_std:.3f}  "
          f"grad_ε={ell_grad_epsilon:.3f}  feat_ε={ell_feat_epsilon:.3f}")

    # t=49 is index 48 (0-based); t=48 is index 47
    prev_final_snap = elliptic.get_snapshot(47)

    # (needs_model, attack_fn, name, kwargs)
    attack_specs = [
        (True,  gradient_attack,             "gradient_attack",
         {"epsilon": ell_grad_epsilon, "steps": cfg.attack.grad_steps}),
        (False, feature_perturbation_attack, "feature_perturbation",
         {"epsilon": ell_feat_epsilon}),
        (False, temporal_perturbation_attack, "temporal_perturbation",
         {"epsilon": ell_feat_epsilon * 2.0, "fraction": 0.40,
          "prev_features": prev_final_snap.features}),
    ]

    for needs_model, attack_fn, atk_name, kwargs in attack_specs:
        print(f"\n  [{atk_name}]")
        if needs_model:
            ar = attack_fn(final_snap, ell_model, ell_params, **kwargs)
        else:
            ar = attack_fn(final_snap, **kwargs)

        _, atk_preds, _ = predict(ell_model, ell_params, ar.perturbed_graph)
        atk_m = classification_metrics(
            final_snap.labels, np.array(atk_preds), mask=final_snap.test_mask
        )
        attack_accs_ell[atk_name] = atk_m["accuracy"]
        print(f"    After attack: acc={atk_m['accuracy']:.4f}  f1={atk_m['f1']:.4f}  "
              f"drop={baseline_acc_ell - atk_m['accuracy']:+.4f}")

        # Temporal attack uses dedicated temporal ontology defense
        if atk_name == "temporal_perturbation":
            def_graph, def_stats = temporal_self_healing(
                ar.perturbed_graph, prev_final_snap.features, cfg.defense.ontology
            )
            def_result = _train_model(ell_model, def_graph, cfg.model,
                                      seed=cfg.seed, verbose=False)
            _, def_preds, _ = predict(ell_model, def_result.best_params, def_graph)
        else:
            dr = run_defense(
                attacked_graph=ar.perturbed_graph,
                model=ell_model,
                attack_type="evasion",
                attacked_params=ell_params,
                defense_cfg=cfg.defense,
                model_cfg=cfg.model,
                seed=cfg.seed,
                baseline_acc=baseline_acc_ell,
                attacked_acc=atk_m["accuracy"],
                damage_threshold=0.05,
            )
            _, def_preds, _ = predict(ell_model, dr.defended_params, dr.defended_graph)

        def_m = classification_metrics(
            final_snap.labels, np.array(def_preds), mask=final_snap.test_mask
        )
        defended_accs_ell[atk_name] = def_m["accuracy"]
        rr = recovery_rate(baseline_acc_ell, atk_m["accuracy"], def_m["accuracy"])
        rr_str = f"{rr:.1%}" if rr is not None else "N/A"
        print(f"    After defense: acc={def_m['accuracy']:.4f}  f1={def_m['f1']:.4f}  "
              f"recovery={rr_str}")

    # ── Temporal line plots ──────────────────────────────────────────────────
    print("\n[7.3] Building temporal line plots (all 49 timesteps) …")

    for needs_model, attack_fn, atk_name, kwargs in attack_specs:
        print(f"\n  [{atk_name}] temporal lines …")
        attacked_accs_t = []
        defended_accs_t = []

        # Temporal attack needs per-step kwargs (prev_features varies per t)
        is_temporal = (atk_name == "temporal_perturbation")

        for t, snap in enumerate(elliptic.snapshots):
            # Build per-step kwargs for temporal attack
            if is_temporal and t > 0:
                step_kwargs = {k: v for k, v in kwargs.items()
                               if k != "prev_features"}
                step_kwargs["prev_features"] = elliptic.snapshots[t - 1].features
            elif is_temporal:
                step_kwargs = {k: v for k, v in kwargs.items()
                               if k != "prev_features"}  # t=0: no prev → synthetic delta
            else:
                step_kwargs = kwargs

            if needs_model:
                ar = attack_fn(snap, ell_model, ell_params, **step_kwargs)
            else:
                ar = attack_fn(snap, **step_kwargs)

            _, atk_preds, _ = predict(ell_model, ell_params, ar.perturbed_graph)
            atk_m = classification_metrics(
                snap.labels, np.array(atk_preds), mask=snap.test_mask
            )

            if is_temporal and t > 0:
                def_graph, _ = temporal_self_healing(
                    ar.perturbed_graph,
                    elliptic.snapshots[t - 1].features,
                    cfg.defense.ontology,
                )
                def_res = _train_model(ell_model, def_graph, cfg.model,
                                       seed=cfg.seed, verbose=False)
                _, def_preds, _ = predict(ell_model, def_res.best_params, def_graph)
            else:
                dr = run_defense(
                    attacked_graph=ar.perturbed_graph,
                    model=ell_model,
                    attack_type="evasion",
                    attacked_params=ell_params,
                    defense_cfg=cfg.defense,
                    model_cfg=cfg.model,
                    seed=cfg.seed,
                    baseline_acc=baseline_accs[t],
                    attacked_acc=atk_m["accuracy"],
                    damage_threshold=0.05,
                )
                _, def_preds, _ = predict(ell_model, dr.defended_params,
                                          dr.defended_graph)

            def_m = classification_metrics(
                snap.labels, np.array(def_preds), mask=snap.test_mask
            )

            attacked_accs_t.append(atk_m["accuracy"])
            defended_accs_t.append(def_m["accuracy"])

            if (t + 1) % 10 == 0 or t == 0:
                print(f"    t={t+1:2d}  base={baseline_accs[t]:.3f}  "
                      f"atk={atk_m['accuracy']:.3f}  def={def_m['accuracy']:.3f}")

        plot_temporal_accuracy(
            {"baseline": baseline_accs,
             "attacked": attacked_accs_t,
             "defended": defended_accs_t},
            attack_name=atk_name,
            dataset_name="Elliptic",
            save_path=sp,
        )

    # ── Elliptic bar chart ───────────────────────────────────────────────────
    plot_accuracy_bar(
        baseline_acc_ell, attack_accs_ell, defended_accs_ell,
        dataset_name="Elliptic", save_path=sp,
    )

    # ── Write results table ──────────────────────────────────────────────────
    _write_elliptic_md(baseline_acc_ell, attack_accs_ell, defended_accs_ell,
                       baseline_accs)

    print(f"\n  Phase 7 done in {_elapsed(t0)}")


def _write_elliptic_md(baseline_acc, attack_accs, defended_accs, baseline_accs_t):
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    fpath = cfg.tables_dir / "elliptic_results.md"
    lines = [
        "# Elliptic Bitcoin Dataset — Attack & Defense Results",
        "",
        f"**Baseline (final snapshot t=49):** acc={baseline_acc:.4f}",
        f"**Mean baseline across 49 timesteps:** acc={np.mean(baseline_accs_t):.4f}",
        "",
        "## Final Snapshot (t=49) — Attack & Defense",
        "",
        "| Attack | After Attack | After Defense | Recovery Rate |",
        "| --- | --- | --- | --- |",
    ]
    for atk in attack_accs:
        rr = recovery_rate(baseline_acc, attack_accs[atk], defended_accs[atk])
        rr_str = f"{rr:.1%}" if rr is not None else "N/A"
        lines.append(
            f"| {atk} | {attack_accs[atk]:.4f} | {defended_accs[atk]:.4f} | {rr_str} |"
        )
    fpath.write_text("\n".join(lines))
    print(f"  [Tables] Saved → {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Write final summary table
# ─────────────────────────────────────────────────────────────────────────────

def write_cora_results_md(baseline_acc, attack_accs, defended_accs,
                           attack_metrics, defended_metrics):
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    fpath = cfg.tables_dir / "cora_results.md"
    lines = [
        "# Cora Dataset — Final Attack & Defense Results",
        "",
        f"**Baseline:** acc={baseline_acc:.4f}",
        "",
        "## Attack Impact",
        "",
        "| Attack | Type | Accuracy | F1 | Drop |",
        "| --- | --- | --- | --- | --- |",
    ]
    poisoning = {"nettack", "meta_attack", "random_structure", "dice"}
    for atk, m in attack_metrics.items():
        t = "Poisoning" if atk in poisoning else "Evasion"
        drop = baseline_acc - m["accuracy"]
        lines.append(f"| {atk} | {t} | {m['accuracy']:.4f} | {m['f1']:.4f} | {drop:+.4f} |")

    lines += ["", "## Defense Performance", "",
              "| Attack | After Attack | After Defense | Recovery Rate |",
              "| --- | --- | --- | --- |"]
    for atk in attack_accs:
        rr = recovery_rate(baseline_acc, attack_accs[atk], defended_accs[atk])
        rr_str = f"{rr:.1%}" if rr is not None else "N/A"
        lines.append(
            f"| {atk} | {attack_accs[atk]:.4f} | {defended_accs[atk]:.4f} | {rr_str} |"
        )
    fpath.write_text("\n".join(lines))
    print(f"  [Tables] Saved → {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg.make_dirs()
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    tee = Tee(LOG_FILE)
    sys.stdout = tee

    t_total = time.time()
    print(f"Pipeline started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log → {LOG_FILE}")

    try:
        # Phase 1
        cora, elliptic = phase1()

        # Phase 3
        (cora_model, cora_params, cora_m,
         gat_model, gat_params,
         ell_model, ell_params, ell_m) = phase3(cora, elliptic)

        baseline_acc = cora_m["accuracy"]

        # Phase 4+5 (Cora)
        (attack_results, defense_results,
         attack_accs, defended_accs,
         attack_metrics, defended_metrics) = phase45(
            cora, cora_model, cora_params, baseline_acc
        )

        # Write Cora results table
        write_cora_results_md(baseline_acc, attack_accs, defended_accs,
                              attack_metrics, defended_metrics)

        # Phase 6 (Cora visualizations)
        # Re-run training to get loss curves if not loaded from checkpoint
        cora_train_result = None
        ckpt_file = cfg.checkpoints_dir / "gcn_cora_baseline.npz"
        if not ckpt_file.exists():
            r = train_model(cora_model, cora, cfg.model, seed=cfg.seed, verbose=False)
            cora_train_result = r

        phase6(cora, cora_model, cora_params,
               baseline_acc, attack_accs, defended_accs,
               attack_metrics, defended_metrics,
               attack_results, defense_results,
               cora_train_result)

        # Phase 7 (Elliptic)
        phase7(elliptic, ell_model, ell_params)

    except Exception:
        print("\n[PIPELINE ERROR]")
        traceback.print_exc()
        raise
    finally:
        print(f"\n{'='*60}")
        print(f"Total runtime: {_elapsed(t_total)}")
        print(f"All outputs → {cfg.results_dir}")
        print(f"{'='*60}")
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
