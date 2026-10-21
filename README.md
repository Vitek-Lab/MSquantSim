# MSquantSim
## Simulated mass spectrometry-based protein abundances enable effective experimental design

![Paper Flow Diagram from JPR Simulation](https://github.com/user-attachments/assets/bd9c5d85-a9f6-4e2b-bb8f-281c0fda8b53)

MSquantSim is a framework for generating realistic simulated mass spectrometry-based protein abundance datasets to optimize experimental design in quantitative proteomics.

## Installation

```
# Clone the repository
git clone https://github.com/devonjkohler/MSquantSim.git
cd MSquantSim

# Install the pinned dependencies, then the package itself
# (editable, so the notebooks use the source in src/)
pip install -r requirements.txt
pip install -e .
```

The pins in `requirements.txt` were tested with Python 3.11. Running the notebooks also needs `jupyter`.

Optional: the legacy cells at the top of `notebooks/simulation_comparison.ipynb` also need `causomic`, `indra_cogex` and INDRA CoGEx credentials. These are not in `requirements.txt` and are not needed for the paper's results.

## Key Features

- Realistic simulation: Preserves marginal distributions, correlations, and complex dependencies between proteins
- Robust to missing values: Effectively handles missing data common in proteomics experiments
- Technology-aware: Captures differences between acquisition technologies (DIA, MRM, etc.)
- Flexible design planning: Supports class prediction, class discovery, and class comparison objectives
- Computationally efficient: Outperforms deep generative models in efficiency and stability

## Case Studies
The repository includes three case studies demonstrating the application of MSquantSim:
### Colorectal Cancer (CRC)
- Blood-based protein markers for cancer detection
- Targeted mass spectrometry data with 67 proteins
- Low missing values and experimental validation cohort

### Pancreatic Ductal Adenocarcinoma (PDAC)
- Biomarker discovery using both DIA and MRM acquisition techniques
- PDAC-DIA: 397 proteins; PDAC-MRM: 15 proteins, healthy vs. PDAC
- Comparison between different acquisition technologies

### Single-cell Melanoma (MEL)
- Single-cell protein measurements with high missing values
- 124 proteins selected using an INDRA prior-knowledge network
- Demonstrates robustness to missing data

## Package modules (`src/MSquantSim/`)

- `simulate.py`: the simulators.
  - Copula-based: `simulate_using_copula`, plus `fit_copula` / `sample_copula` to fit once and sample many times.
  - Per-protein: `simulate_using_per_protein`, `fit_per_protein` / `sample_per_protein`.
  - TVAE: `simulate_using_tvae`.
- `downstreamanalysis.py`: the analysis tasks used to evaluate simulations.
  - Random Forest class prediction: `train_rf`.
  - PCA class discovery: `get_pca_explained_variance`, `calculate_percentage_of_variation`.
  - t-tests with Benjamini–Hochberg adjustment for class comparison: `run_ttest`, `identify_significant_proteins`.
- `similarity_scores.py`: similarity between experimental and simulated data (`calculate_similarity_score`, `similarity_across_simulations`, wrapping TableEvaluator) and `plot_corr`.
- `power.py`: empirical power and false discovery proportion of t-tests for differential abundance.
  - The semi-synthetic benchmark: `center_by_group` removes the condition effect, then known changes are injected; `run_experiments`, `summarize_experiments`, `summarize_success`.
  - Placing the differentially abundant proteins: `clustered_de_sets`, `random_de_set`, `seed_changed_sets`.
  - The standard independence-based sample-size formula: `analytic_sample_size`.
- `utility.py`: missing-value model.
  - Fit the MNAR detection curve: `fit_mnar`, `calibrate_offsets`.
  - Apply it to simulated data: `apply_dropout`, `mask_simulations_with_lod`.
- `batch.py`: runs the Random Forest evaluations as many small cluster jobs, e.g. a Slurm job array (`export_rf_inputs`, `rf_tasks`, `run_rf_task`, `gather_rf_results`).
- `plotting.py`: shared figure style (`use_paper_style`, loading `styles/paper.mplstyle`), fixed per-simulator colors, and correlation heatmaps.

## Notebooks (`notebooks/`)

- `CRC_case_study.ipynb`, `MEL_case_study.ipynb`, `PDAC_case_study.ipynb` (PDAC-DIA and PDAC-MRM):
  - prepare the data;
  - simulate or load saved simulations;
  - run the evaluations (correlations, similarity scores, Random Forest, PCA, t-test power, sample-size grids, semi-synthetic benchmarks);
  - save summary results to `data/results/<dataset>/`.
- `simulation_comparison.ipynb`, section "Paper figures": builds the paper's combined multi-dataset figures and tables from `data/results/`. Run its setup cell first. The cells above that section are legacy and not used in the paper.
- `old_vartika_notebooks/`: earlier exploratory notebooks, kept for reference.

```
jupyter notebook notebooks/
```

The simulated datasets themselves are large and are not in the repository. The case-study notebooks regenerate them, or `scripts/run_simulations.py` does, one dataset and simulator at a time:

```
python scripts/run_simulations.py --dataset MEL --method copula --n-sims 100
```

## Data

- `data/crc`, `data/mel`, `data/pdac`: protein-level input data.
  - `data/pdac/PDAC_MRM_protein_level_data_v2.csv` holds the PDAC-MRM data with the sample classes from Kim et al. 2024. These are 152 healthy and 154 PDAC samples; the benign samples are removed.
- `data/results/`: summary results behind the paper's figures and tables, one folder per dataset (CRC, MEL, PDAC_DIA, PDAC_MRM).
  - They are enough to rebuild every figure and table in the paper with `notebooks/simulation_comparison.ipynb`.
  - The large per-experiment files and all results of the correlation-aware sample-size analysis (Step 4.2) are kept on the lab server, not in this repository. `data/results/SERVER_FILES.md` lists them with checksums and says how to restore them.

## Scripts (`scripts/`)

- `run_simulations.py`: generate the simulations for one dataset/simulator pair, following the case-study preprocessing.
- `remask_simulations.py`: re-apply the fitted MNAR dropout model to existing simulations.
- `make_column_maps.py`: precompute UniProt → HGNC column maps, so cluster jobs don't need protmapper/INDRA.

## Implementation Details
MSquantSim implements several simulation approaches:

- Copula-based simulation: Captures dependencies between proteins while maintaining marginal distributions
- Per-protein simulation: Simple baseline that models each protein independently
- Tabular Variational Autoencoder (TVAE): Deep learning approach for comparison
