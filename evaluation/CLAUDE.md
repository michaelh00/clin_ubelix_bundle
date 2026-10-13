# CLIN diffusion project: context for Claude

Read `NEXT_STEPS.md` **Part B** before doing anything. It has the verified data conventions, the model facts, the evaluation protocol, the numbers to beat, and the ordered task list (B.1–B.7).

Ground rules:
- Reuse `clin_common.py` for loading data, the shared test cases (`results/eval_cases.csv`) and scoring. Don't re-implement date or grid handling: `day_index` is 1-based, the metadata columns are (id, lon, lat), and the netCDF dims must be read by name.
- Never tune on the 1940s test cases. Use 1950s validation cases.
- Report every result per era, for all/land/sea cells, by distance to the nearest station, and as ensemble mean + spread, against the ridge baseline.
- Save new model outputs to `results/` in the same `.npz` layout as `diffusion_*.npz`.
- Ask before submitting long UBELIX jobs.
