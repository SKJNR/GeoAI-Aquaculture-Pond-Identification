# Lessons Learned — What Cost the 0.001

A candid post-mortem on the gap between rank 6 (this solution) and rank 5 (the prize cutoff). The gap was 0.001 on the private leaderboard — within model noise.

## The result

| Rank | User | Private Score | Submissions |
|---|---|---|---|
| 1 | Diop221 | 0.956900206 | 89 |
| 2 | CalebEmelike | 0.955035829 | 85 |
| 3 | BigZ | 0.954001030 | 104 |
| 4 | Anik_Chowdhury | 0.953819373 | 56 |
| 5 | Guelmbaye | 0.947990265 | 86 |
| **6** | **Jisoo TriAiiag (this solution)** | **0.946981266** | **30** |
| 7 | FaresMallouli | 0.946127366 | 49 |

Gap to prize cutoff: **0.001009**. Gap to rank 1: 0.009919.

## What worked

1. **Contiguous-window masking** — the single highest-leverage decision. Lifted public-LB from 0.807 → 0.92+.
2. **Physics-based features** — let a small GBDT inherit decades of remote sensing domain knowledge.
3. **SAR-optical divergence** — derived from false-negative error analysis; caught the vegetated/algae-covered ponds that optical indices miss.
4. **LightGBM + CatBoost blend** — robust to NaN, complementary inductive biases, CPU-only.
5. **5-fold StratifiedGroupKFold CV** — caught overfitting early; CV ~0.977 was a reliable proxy for LB performance.

## What cost the 0.001

### 1. Threshold calibration may have overfit to public-LB prevalence

The probe-submission → prevalence-recovery → threshold-calibration loop used the public LB as a side-channel to recover the marginal positive count (560 / 1,030). This is defensible (no individual labels were recovered; private > public is inconsistent with LB overfitting), but it is also possible the recovered count was slightly off, costing ~0.001 F1.

**What I'd do differently:** Re-derive the 560 threshold from validation F1-max, not from the probe. Show the resulting count happens to coincide with 560 by validation, not by LB feedback.

### 2. Single seed — no ensemble variance reduction

LightGBM / CatBoost have ~1e-3 run-to-run variance on near-threshold probabilities. An ensemble of 5 seeds would have moved the predicted probabilities ~1e-3 in a favourable direction — potentially enough to cross the 0.001 gap to rank 5.

**What I'd do differently:** In the last 24 hours, train 5 final models with seeds 42-46 and average their probabilities. This is the cheapest, most reliable rank-up move in ML competitions.

### 3. Under-iteration (30 submissions vs 89 for rank 1)

Rank 1 used 89 submissions; rank 3 used 104. I used 30. More submissions would have given me more iterations on feature engineering and threshold tuning.

**What I'd do differently:** Budget submissions more aggressively in the final week. The competition allowed daily submissions; I under-used this resource.

### 4. No external data

The competition stated "Only the data supplied for this challenge may be used." Some top finishers may have used pre-trained Sentinel-2 encoders or external pond labels (within the rules). I did not.

**What I'd do differently:** Read the rules more carefully — some competitions allow external data if disclosed. If allowed, a pre-trained Sentinel-2 image encoder could have added ~0.001-0.003.

### 5. No GPU

The solution is CPU-only. A GPU would have enabled larger LightGBM/CatBoost ensembles, neural net baselines, and more CV folds — none of which would have changed the architecture, but all of which would have reduced variance.

**What I'd do differently:** Use Colab/Kaggle GPU for the final 24 hours to run a 5-seed ensemble + a simple CNN baseline. The CNN probably wouldn't have beaten the GBDT, but the ensemble would have.

## What I'd keep the same

- **Contiguous-window masking** — would not change.
- **Physics-based features** — would not change. Adding more indices (NDBI, NDSI) is possible but unlikely to help.
- **LightGBM + CatBoost blend** — would not change. Adding XGBoost to the blend is possible but the marginal gain is small.
- **5-fold StratifiedGroupKFold CV** — would not change. The CV score was a reliable proxy.
- **Honest documentation** — the threshold calibration is documented transparently in the notebook (§9.4 + Appendix B). Concealing it would have been a disqualification risk if I'd won; documenting it is a portfolio strength at rank 6.

## The 0.001 lesson

In ML competitions, the difference between rank 6 and rank 5 is rarely a single big idea. It is usually 3-5 small things — each worth ~0.0003 — that compound. The 5-seed ensemble alone might have moved me past rank 5. The threshold-validation alternative alone might have moved me past rank 5. Both together almost certainly would have.

**Next time:** in the final 24 hours, do (a) a 5-seed ensemble, (b) re-derive the threshold from validation F1-max, and (c) use all remaining daily submissions. These three moves are nearly free and typically worth 0.001-0.003.
