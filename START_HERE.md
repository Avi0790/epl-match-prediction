# Start here — updated EPL project

The ZIP extracts into `epl-predictor-updated`. Keep it separate from your existing
project so your local files and environment are preserved.

1. Extract the ZIP in Finder.
2. Move `epl-predictor-updated` into your Desktop folder named `EPL prediction`.
3. Open Terminal and run each command separately:

```bash
cd "/Users/coleworld/Desktop/EPL prediction/epl-predictor-updated"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Expected test summary: `Ran 64 tests` and `OK`.

Run a demonstration with the included xG model:

```bash
python predict.py --model models/xg/production.joblib --home Arsenal --away Chelsea --date 2026-10-31
```

This is a hypothetical matchup, not a verified scheduled game. The model returns
home/draw/away probabilities, goal estimates, explanations, and data coverage.
Expect warnings about stale xG: the supplied xG ends in May 2026, while match results
extend to October 12, 2026. That disclosure is intentional.

To train the xG prediction artifact yourself from the frozen selection:

```bash
python train_xg.py --production-only --selection reports/xg_verified/selection.json --models models/xg
```

The updated source and data are already included. You do not need the optional
soccerdata downloader to run these steps. See README.md for reproduction and import
instructions, and XG_REPORT.md for verified performance and limitations.
