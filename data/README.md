# Data

This repo does **not** commit datasets. This file documents how to obtain them.
Raw pulls are cached under `data/raw/` (gitignored).

- **Source:** [FRED — Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org/),
  series **`SP500`** (S&P 500 daily index). Any daily FRED series works; override
  with the `MF_SERIES_ID` environment variable (e.g. `DGS10` for the 10-Year
  Treasury yield).
- **License / terms:** FRED data is free to use; most series (including `SP500`)
  are redistributable with attribution to the source. The raw series itself is
  **not** redistributed here — only the fetch step is.
- **How to obtain:** keyless CSV download, no account required:
  ```bash
  make train          # fetches + caches automatically, then trains
  # or directly:
  uv run python -c "from market_forecast.data import load_series; print(load_series().tail())"
  # underlying endpoint:
  # https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500&cosd=2015-01-01
  ```
- **Expected layout:**
  ```
  data/
  ├── raw/         # cached FRED CSV pulls (gitignored), e.g. SP500.csv
  ├── interim/     # (unused)
  └── processed/   # (unused)
  ```
- **Size / notes:** ~2,500 daily rows (~10 years), well under 1 MB. No PII. FRED
  marks non-trading days with `"."`; those rows are dropped on load (never
  forward-filled), and the loader validates a sorted, gap-free, NaN-free
  `DatetimeIndex`.
