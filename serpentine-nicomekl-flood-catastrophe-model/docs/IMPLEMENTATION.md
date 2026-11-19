# Implementation and reproducibility notes

## Preserved source

Ten source files, including the footprint package marker, were copied unchanged from the completed local project. Their hashes are recorded in `source_manifest.json`. The final dependency chain uses `compute_event_lossV3`, `coastalV2`, and `riverine_interior_drainage`, including their relative dependency on `flood_footprint/core.py`. The similarly named earlier top-level `flood_footprint.py` is intentionally absent to avoid ambiguity with the package.

## New entry point

`run_simulation.py` implements the setup and batching from `18_final_monte_carlo_sim.ipynb`, including its cleaned-segment branch (cells 78-86 in the original notebook):

- Score the cleaned segment layer using low elevation below 2 m, 50 m sample spacing, and 500 m protection proximity.
- Use EPSG:3005 and buffer the floodplain expansion boundary by 5,000 m.
- Use original source weights, severity probabilities, and footprint parameters.
- Use batches of at most 500 events and seed each batch with base seed plus batch index.
- Retain failed-event records; exclude them from successful-event loss statistics.

New runs default to 10 events. Full historical batching uses `--n-sims 10000 --batch-size 500 --seed 42`. Segment scoring and footprint generation occur only when the user launches a simulation. Footprint and damage objects are discarded between batches; property-level files are optional. The original engine still retains those objects within each batch, so memory use depends on batch size.

## Archived results and report

The packaged master CSV is copied from `simulation_outputs2/event_results_clean_10000_master.csv`, with 9,996 succeeded and four failed rows. The selected PDF is the descriptively named report last modified September 20, 2026, rather than the earlier September 18 `Final Report V1.pdf`. Its bytes are unchanged. The selected figures and animation are historical report assets, not new simulation results.

The geometry and required values of the reduced exposure input are checked against the original. The optional R extraction script now uses the repository working directory and checks its dependencies instead of installing them automatically. The original simulation was not rerun during packaging. See `verification.json` for checks actually performed; it does not claim a full end-to-end flood simulation test.
