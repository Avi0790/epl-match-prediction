# Frozen baseline experiment — 15 October 2026

**The baseline did not beat market-average closing odds on the 2025/26 holdout.**

The classifier and goal-model family were selected on earlier seasons before the final test. The 60 subsequently collected 2026/27 matches are used only for production fitting.

## Final test: 380 matches

| Model | Log loss ↓ | Brier ↓ | RPS ↓ | H/D/A accuracy ↑ |
|---|---:|---:|---:|---:|
| elo (selected before test) | 1.0284 | 0.6189 | 0.2092 | 48.2% |
| elo+platt | 1.0284 | 0.6195 | 0.2100 | 47.9% |
| team_poisson | 1.0418 | 0.6265 | 0.2135 | 47.9% |
| team_poisson+platt | 1.0426 | 0.6271 | 0.2143 | 46.3% |
| xgb_shallow | 1.0443 | 0.6286 | 0.2132 | 47.1% |
| xgb_shallow+platt | 1.0401 | 0.6278 | 0.2139 | 47.1% |
| xgb_medium | 1.0469 | 0.6300 | 0.2138 | 47.9% |
| xgb_medium+platt | 1.0419 | 0.6289 | 0.2146 | 47.1% |
| Market-average closing | 1.0118 | 0.6077 | 0.2045 | 49.5% |

Brier sums squared error across all three classes. RPS averages the two cumulative H/D/A boundaries. All bookmaker comparisons use the same 380 matches and proportionally normalized inverse odds. The midnight forecast has less information than closing prices.

## Paired uncertainty

- log_loss: model minus bookmaker **+0.01659**; 95% calendar-week block-bootstrap interval **[+0.00051, +0.03373]**.
- rps: model minus bookmaker **+0.00475**; 95% calendar-week block-bootstrap interval **[-0.00070, +0.01049]**.

Positive differences favor bookmakers. The 2,000 paired resamples describe uncertainty for this season; they are not proof of performance in future seasons or betting returns.

## Scorelines and totals

| Metric | Selected model | Closing benchmark |
|---|---:|---:|
| Exact score | 11.6% | Not available |
| Actual score among top five | 49.5% | Not available |
| O/U 2.5 accuracy | 53.2% | 57.1% |
| O/U 2.5 log loss | 0.6872 | See raw closing probabilities |

The modal score's result accuracy is 38.2%. That differs from H/D/A argmax accuracy because a class probability sums many scorelines. The modal-score probabilities use the XGBoost Poisson rates with each H/D/A region rescaled to the selected Elo classifier.

## Earlier walk-forward selection

| Candidate | Pooled log loss | RPS | Accuracy |
|---|---:|---:|---:|
| elo | 0.9676 | 0.2012 | 54.5% |
| elo+platt | 0.9739 | 0.2029 | 54.4% |
| xgb_shallow | 0.9839 | 0.2046 | 53.4% |
| xgb_shallow+platt | 0.9868 | 0.2065 | 53.7% |
| team_poisson | 0.9872 | 0.2087 | 52.7% |
| xgb_medium | 0.9874 | 0.2053 | 52.9% |
| xgb_medium+platt | 0.9885 | 0.2070 | 53.5% |
| team_poisson+platt | 0.9989 | 0.2113 | 52.3% |

Six evaluation seasons cover 2018/19–2023/24 (2,280 predictions per candidate). Every evaluation season has a preceding calibration season and earlier expanding training history. Platt calibration did not improve the selected Elo candidate on validation. Raw Elo is therefore intentional.

Goal-model selection used independent goal negative log likelihood:
- team_poisson: 3.0079
- xgb_poisson: 2.9745

## Calibration and explanations

![Holdout calibration](reports/calibration.png)

Eight fixed-width bins per class; small bins are noisy. Exact bin counts and means are in `reports/calibration.csv`. Native TreeSHAP for the shallow XGBoost candidate uses the separate 2024/25 calibration season. Elo difference is its highest mean absolute contribution; venue shot form and recent shots on target also appear among the leading factors. These are descriptive model explanations, not evidence that adding each feature independently improves forecasts.

## Audit trail and limits

- 35 tests pass, including future/target-result mutation, same-day ordering, strict availability cutoffs, season partitions and score-matrix consistency.
- Three historical scores match official club/UEFA reports; see `reports/source_spot_checks.json`. This does not validate every source statistic.
- Midnight cutoffs use declared result/statistic delays. Historical publication and revision times remain unknown; this is a reconstructed archive.
- EPL rest/congestion omits all other competitions. No xG, injuries, player/lineup features or odds features were used.
- Production refitting includes newer data and is separate from this frozen experiment. The included example is a hypothetical supplied fixture, not a verified scheduled match.

Research data fingerprint: `ff3c0ac28c0f0c8713e07b7bc2e947c59b5f6dfe570aa26ba3c1bc37242ac8d1`.

All individual predictions, split dates, metrics, calibration bins, feature rows and selected configuration are included under `reports/`. The provider raw files and normalized SQLite data are included under `data/`.
