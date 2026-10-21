# Result files kept on the lab server

These files are **not in git** (see `.gitignore`): the large result files and all 4.2 sample-size results. Everything needed to rebuild the paper's figures and tables, except the 4.2 candidate figures, is in this folder.

- **Location:** lab server, TODO: add the path (prepared locally as `server_upload_2026-10-21/` on 2026-10-21).
- **To restore:** copy the server folder's `results/` contents into this folder (same `<dataset>/<file>` layout), then check the md5 sums below.
- **3.7 figures:** they read `PDAC_MRM/power_curves_figure_data.csv` (in git). To rebuild that file from the per-experiment data, restore `PDAC_MRM/power_curves_experiments.csv` and set `PC_FROM_EXPERIMENTS = True` in the 3.7 cell of `simulation_comparison.ipynb`.

| File | Size (MB) | md5 | Content |
|---|---|---|---|
| `MEL/missingness_correlations.npz` | 4.93 | `4be6e8c991a91c03300c472c0e7b16cf` | missing values (dropped section, Reserve) |
| `MEL/placement_calculator.csv` | 0.00 | `038a8b10120f6e40a493cc5019863187` | 4.2 placement benchmark (current 4.2 design; candidate figures) |
| `MEL/placement_clusters.csv` | 0.27 | `1da49a6667ad31b08b2454991a3504a0` | 4.2 placement benchmark (current 4.2 design; candidate figures) |
| `MEL/placement_methods_replicates.csv` | 10.80 | `261ba4ad103a4610a0b6ba8de4bbee54` | 4.2 placement benchmark (current 4.2 design; candidate figures) |
| `MEL/placement_methods_summary.csv` | 0.03 | `ee5ef5718badbad568ec33df7d94ac25` | 4.2 placement benchmark (current 4.2 design; candidate figures) |
| `MEL/placement_perprotein_replicates.csv` | 5.69 | `f5eb8fdddf6f4ee610cf8f5da1eb36e4` | 4.2 placement benchmark (current 4.2 design; candidate figures) |
| `MEL/placement_perprotein_summary.csv` | 0.02 | `a231321423f3f7483f3470a05e333eea` | 4.2 placement benchmark (current 4.2 design; candidate figures) |
| `MEL/placement_replicates.csv` | 10.91 | `c0a4ad507938e93fc1b61741e607da07` | 4.2 placement benchmark (current 4.2 design; candidate figures) |
| `MEL/placement_summary.csv` | 0.03 | `7dec1616289a75abf1602201f45f6f2a` | 4.2 placement benchmark (current 4.2 design; candidate figures) |
| `MEL/sample_size_calculator.csv` | 0.00 | `58c2f49600f9bb203e2fd3e60d235954` | 4.2 MEL pilot (superseded, Reserve) |
| `MEL/sample_size_experiments.csv` | 5.38 | `91e11e940f795e8ad8a4398fe80d870f` | 4.2 MEL pilot (superseded, Reserve) |
| `MEL/sample_size_protein_calculator.csv` | 0.00 | `42c27e7ab060bf75f364dd918449058c` | 4.2 MEL pilot (superseded, Reserve) |
| `MEL/sample_size_protein_detection.csv` | 0.00 | `88be6167f11df7257cbcfa606f908dea` | 4.2 MEL pilot (superseded, Reserve) |
| `MEL/sample_size_seed_calculator.csv` | 0.00 | `c046bb143c884e30eeee81f0dd02e6ab` | 4.2 seed-protein design (superseded, Reserve) |
| `MEL/sample_size_seed_experiments.csv` | 3.35 | `60d12887c7447bfe02fe9998c84d60be` | 4.2 seed-protein design (superseded, Reserve) |
| `MEL/sample_size_seed_premise.csv` | 0.01 | `b83286d28a4fe14485a8cfb7ccf59de3` | 4.2 seed-protein design (superseded, Reserve) |
| `MEL/sample_size_seed_sets.csv` | 2.09 | `602b2ab3e0039d9aee6987ab8bf5abcf` | 4.2 seed-protein design (superseded, Reserve) |
| `MEL/sample_size_seed_summary.csv` | 0.02 | `42f13afeba029f568868d84df49c5146` | 4.2 seed-protein design (superseded, Reserve) |
| `MEL/sample_size_summary.csv` | 0.02 | `d5199082144966bf41870532da15f543` | 4.2 MEL pilot (superseded, Reserve) |
| `PDAC_DIA/placement_calculator.csv` | 0.00 | `264fac8f1e9aba368c4fc3ed3933c1f0` | 4.2 placement benchmark (current 4.2 design; candidate figures) |
| `PDAC_DIA/placement_clusters.csv` | 1.71 | `6a8bff8c8f12de67d532460835497e05` | 4.2 placement benchmark (current 4.2 design; candidate figures) |
| `PDAC_DIA/placement_methods_replicates.csv` | 15.94 | `49e6f72c7b98f73b499e1c85c3c4f2f7` | 4.2 placement benchmark (current 4.2 design; candidate figures) |
| `PDAC_DIA/placement_methods_summary.csv` | 0.03 | `29ab0677ae830630fbe99956a20bb401` | 4.2 placement benchmark (current 4.2 design; candidate figures) |
| `PDAC_DIA/placement_perprotein_replicates.csv` | 8.42 | `d2d5dfeddbfd18a61012375bf5d6be5f` | 4.2 placement benchmark (current 4.2 design; candidate figures) |
| `PDAC_DIA/placement_perprotein_summary.csv` | 0.02 | `61bcaade75feeb7785f54f8513b6586b` | 4.2 placement benchmark (current 4.2 design; candidate figures) |
| `PDAC_DIA/placement_replicates.csv` | 12.04 | `038dbc9056963a064d7c6ed4b3d88ab8` | 4.2 placement benchmark (current 4.2 design; candidate figures) |
| `PDAC_DIA/placement_summary.csv` | 0.04 | `6703ff29e2564e9356637bce2876ecef` | 4.2 placement benchmark (current 4.2 design; candidate figures) |
| `PDAC_DIA/sample_size_seed_calculator.csv` | 0.00 | `eb95e1e1e0fe7e8d8d00e0fd53b76ff2` | 4.2 seed-protein design (superseded, Reserve) |
| `PDAC_DIA/sample_size_seed_experiments.csv` | 2.66 | `b31dd82a6b1c4eef6fc3c296f4b3769b` | 4.2 seed-protein design (superseded, Reserve) |
| `PDAC_DIA/sample_size_seed_premise.csv` | 0.01 | `ea5986b35bc099c9eb22ec08c228fced` | 4.2 seed-protein design (superseded, Reserve) |
| `PDAC_DIA/sample_size_seed_sets.csv` | 2.71 | `fab795d7a8c114589ad0b1f65da1651f` | 4.2 seed-protein design (superseded, Reserve) |
| `PDAC_DIA/sample_size_seed_summary.csv` | 0.01 | `74cd084f27471f1ae501a22f8ea66b0d` | 4.2 seed-protein design (superseded, Reserve) |
| `PDAC_MRM/power_curves_experiments.csv` | 12.80 | `e67b9b9cc70c5b109d3b2ba30cf9957c` | 3.7 per-experiment power (the figures use power_curves_figure_data.csv, in git) |
