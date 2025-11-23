# Small Interactive Math Visualizations

A collection of small vibe-coded visualizations. Each visualization lives in its own subdirectory under `apps/` and can be run via `uv run <name>`.

Currently included
- `maxvis`: Softmax vs Sparsemax in 3D — adjust a 3D logit vector and see softmax(z) and sparsemax(z) on the probability simplex, with the raw z point in 3D.

Run with uv
- `uv run maxvis`

Add a new visualization
1) Create a package under `apps/<yourname>/` with at least:
   - `apps/<yourname>/app.py` exposing a `main()` function
   - `apps/<yourname>/__init__.py` (can be empty)
   - optional: `apps/<yourname>/assets/` for CSS/assets
2) In `pyproject.toml`, add a console script under `[project.scripts]`, e.g.:
   - `<yourname> = "apps.<yourname>.app:main"`

