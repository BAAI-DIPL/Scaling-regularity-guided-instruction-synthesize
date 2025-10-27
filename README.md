This is an English README based on the title of your paper and the content of the provided Python scripts.

***

# ACCELERATE SCALING OF LLM FINETUNING VIA QUANTIFYING THE COVERAGE AND DEPTH OF INSTRUCTION SET

This repository contains the supplementary code for the paper: *ACCELERATE SCALING OF LLM FINETUNING VIA QUANTIFYING THE COVERAGE AND DEPTH OF INSTRUCTION SET*.

The project addresses the challenge of scaling supervised fine-tuning (SFT) for Large Language Models (LLMs) by demonstrating that model performance is primarily governed by two fundamental dataset properties: **Semantic Coverage** (the breadth of task domains) and **Information Depth** (the richness or complexity of individual examples).

The code implements the necessary data analysis, novel sampling strategies, and scaling law regression models to validate this hypothesis and accelerate the selection of highly effective SFT instruction subsets.

## Code Structure

The repository is structured around three core functional components: Data Distribution Analysis, Efficient Data Selection, and Scaling Law Modeling.

### 1. Data Distribution Analysis and Visualization

These scripts analyze the distribution of the instruction set in a pre-computed semantic space (e.g., t-SNE or UMAP embeddings) to quantify **Semantic Coverage** and relate it to model performance.

| Script | Description |
| :--- | :--- |
| `analyze.py` | Analyzes the spatial distribution of instruction data by dividing the 2D semantic space into a fine-grained grid (`np.histogram2d`). It calculates grid-level counts to quantify the *coverage* of the instruction set. |
| `dsitrib_ana.py` | Calculates key performance metrics (e.g., mean loss) for each bin in the semantic grid. It generates heatmaps to visualize the distribution of both data density and mean model loss across the semantic space, linking spatial location to model effectiveness. |

### 2. Efficient Instruction Set Selection

These scripts implement novel sampling strategies that leverage the quantified coverage and depth metrics to select data subsets that are more effective than random sampling.

| Script | Description |
| :--- | :--- |
| `pick_data.py` | Implements a **Coverage-Maximizing** selection strategy. It iteratively selects one data point from each non-empty bin in the semantic grid to ensure maximum *Semantic Coverage* for a given budget. |
| `small_data_set_data.py` | Another implementation for constructing small, high-coverage instruction sets by selecting from the grid distribution over multiple rounds, prioritizing diversity. |
| `top1000_5_top2000_3_other_1.py` | Implements a **Depth-Aware Stratified Sampling** strategy. It selects data points based on both their semantic location (Coverage) and a pre-determined importance rank (Depth), applying different sampling rates (e.g., 5 points from top-tier bins, 3 from middle-tier, 1 from others) to balance both properties. |

### 3. Scaling Law Regression

These scripts are dedicated to modeling the relationship between the proposed dataset metrics and the LLM's final performance (e.g., validation loss).

| Script | Description |
| :--- | :--- |
| `regression.py` | Implements the core multi-variable regression analysis (using `statsmodels.api`) to fit the scaling law. It models the LLM's loss (`log_y1`) as a function of the quantified Semantic Coverage (`l_x1_l_x3`) and Information Depth (`log_x2`). |
| `regression_v3.py` | An updated or refined version of the regression script, including advanced data preparation (log transformations) and 3D visualization code (`matplotlib` with 3D projection) to plot the fitted regression plane alongside the actual experimental data points. |

## Dependencies

The scripts primarily rely on standard scientific Python libraries:

* `numpy`
* `pandas`
* `json`
* `matplotlib` (for plotting and 3D visualization)
* `statsmodels` (for regression analysis)
* `sklearn` (potentially for t-SNE or DBSCAN, as hinted in `dsitrib_ana.py`)
* `tqdm` (for progress bars)