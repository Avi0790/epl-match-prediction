# Rolling xG experiment and implementation update

Verified on the supplied data snapshot on 2026-10-28 (UTC). This report records
reproduction and engineering checks; rerunning an already examined test season
does not create new independent evidence.

## Results

Lower log loss is better. Validation pools six chronological walk-forward seasons,
2018/19 through 2023/24 (2,280 matches). The final season is 2025/26 (380 matches).
The xG extension is a second look at that final season.

| Model | Validation log loss | Test log loss | Test accuracy |
| --- | ---: | ---: | ---: |
| Elo logistic baseline | 0.967637 | 1.028352 | 48.2% |
| Elo + rolling xG logistic model | 0.963330 | 1.025677 | 49.2% |
| Closing market benchmark | — | 1.011767 | 49.5% |

The zero-extra-delay xG predictions reproduce the original saved predictions
within floating-point tolerance. Model selection still chooses `logit_xg`.
The xG model makes 187 correct predictions versus 183 for Elo and 188 for the
market on this final season. The xG model's point estimate of log loss remains
0.013910 above the closing market.

## Paired uncertainty

Differences below are xG minus Elo, so negative values favor xG. The bootstrap
resamples calendar weeks, keeping both models' predictions paired within each
sample (2,000 replicates, seed 2026).

| Sample | Log-loss difference | 95% interval | RPS difference | 95% interval |
| --- | ---: | --- | ---: | --- |
| Validation, 2,280 matches | -0.004307 | [-0.008436, -0.000117] | -0.001330 | [-0.002616, -0.000077] |
| Test, 380 matches | -0.002675 | [-0.010859, +0.005681] | -0.001147 | [-0.003802, +0.001560] |

The validation interval narrowly excludes zero, but the same validation data
selected among 14 raw/calibrated variants. This fixed-prediction bootstrap does
not repeat model selection or capture all training uncertainty. The test interval
includes zero: the gain is inconclusive, not evidence of equivalence. Treat the
entire xG extension as exploratory.

The full verified reports are in `reports/xg_verified/`. The comparison CLI names
both models explicitly. Its shared evaluation helper uses the historical key
`model_minus_bookmaker` internally even when the reference is Elo; the standalone
`paired_xg_vs_elo.json` names the reference explicitly and uses
`model_minus_reference` to avoid that ambiguity.

## Corrected timing check

The original `--extra-history-lag-hours 24` delayed baseline result/statistic
history as well as xG. It therefore did not isolate xG publication timing.
The new `--extra-xg-lag-hours 24` leaves result, goal, shot and Elo timing unchanged.

| xG-only extra delay | Validation xG log loss | Test xG log loss | Test accuracy |
| --- | ---: | ---: | ---: |
| 0 hours | 0.963330 | 1.025677 | 49.2% |
| 24 hours | 0.963462 | 1.025491 | 49.2% |

The Elo test probabilities are bit-for-bit identical between these two runs.
The selected xG model remains `logit_xg`; the performance change is small.
This is a sensitivity check around a declared historical availability assumption,
not proof of when Understat originally published or revised its values.
The corrected run is in `reports/xg_only_lag24/`; `reports/xg_lag24/` preserves
the earlier combined-delay run.

## What changed in the implementation

- Separate xG availability from baseline history, with strict pre-cutoff masking.
  Late xG keeps its match's original place in the rolling window.
- Fit and save an xG classifier for future prediction from the frozen selection.
  Single-match and batch commands dispatch to its feature builder automatically.
- Retain the independently selected baseline goal regressor. Goal estimates are
  not presented as an xG goal-model experiment.
- Include all active xG and batch command modules in the installable package.
- Add a documented optional acquisition script, update project instructions, and
  report xG age/coverage in predictions.

The supplied future-prediction artifact was fitted with an explicit cutoff of
2026-10-28T00:00:00Z, using 4,240 eligible post-warmup matches. It is separate from
the historical evaluation artifacts and must not be used to score earlier matches.
Its metadata records the dataset fingerprint, dependencies, fitting cutoff and
feature configuration.

## Verification and remaining limits

All 64 unit/regression tests pass in Python 3.12. The newly built 0.3.0 wheel was
installed outside the source directory; every previously missing module imports,
and its saved-model forecast reproduces the source result. Score-matrix
probabilities sum to one and match the classifier's home/draw/away probabilities.
The two-fixture batch demonstration also runs. Check outputs are included under
`reports/updated_test_suite.txt`, `reports/installed_package_check.json`,
`reports/xg_demo.json` and `reports/demo_predictions.csv`.

The supplied snapshot contains 4,560 historical matches with xG (9,120 team-match
observations), plus 60 newer result/statistic records without xG. Results extend
through October 12, 2026; xG ends May 24, 2026. The demonstration fixtures are
hypothetical, and the resulting forecasts explicitly flag old xG. The optional
downloader was not exercised against the live provider for this update.

The project models teams and match outcomes, not individual player impact.
Closing odds may incorporate information unavailable at the midnight forecast
cutoff. No claim of beating the market, betting profitability, or an 82% accuracy
result is supported. A stronger next experiment would freeze the configuration
and record predictions prospectively for a predefined block of new matches.
