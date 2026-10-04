# scRARE

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-implemented-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/task-scRNA--seq%20clustering-brightgreen" />
</p>

Official implementation of **scRARE: Reliability-Aware Relation Evolution for Single-Cell RNA-seq Clustering**.

scRARE is an unsupervised scRNA-seq clustering framework designed to reduce the influence of unreliable intermediate cell-cell relations during representation learning and graph refinement. The method follows a **Calibrate-Purify-Evolve** paradigm that jointly performs reliability calibration, low-rank relation purification, and clustering-maturity-driven topology evolution.

---

### Overview

Given a raw cell-gene expression matrix, scRARE first performs library-size normalization, log transformation, highly variable gene selection, and PCA. Two complementary cell graphs are then constructed from the processed expression features: a Euclidean base graph and a cosine auxiliary graph. The base graph is also used to obtain graph-smoothed expression features and to initialize the dynamic graph.

Two non-shared expression encoders and two non-shared graph encoders learn complementary cell representations. The two expression representations are averaged to obtain the fused embedding used for clustering.

The main components are:

- **Multi-view cell representation learning**: two independent expression encoders and two independent graph encoders capture complementary transcriptional and neighborhood information.
- **CNRC**: Conformalized Node-to-Pair Reliability Calibration estimates node-wise clustering confidence from soft assignments and assignment entropy, and further derives pairwise reliability using cross-view agreement.
- **Reliability-aware contrastive learning**: reliable hard cell pairs receive larger weights when representation similarity is inconsistent with current cluster-assignment similarity.
- **RLRP**: Reliability-Guided Low-Rank Relation Purification integrates graph-derived, cluster-derived, and representation-derived cell-anchor relations and separates sparse relation-specific residuals from the dominant low-rank structure.
- **CMTE**: Clustering-Maturity-Driven Topology Evolution progressively incorporates purified relations into the dynamic graph according to the certainty of the current clustering state.
- **Joint objective**: the model jointly optimizes reliability-aware contrastive learning, confidence-weighted clustering, and low-rank relation purification.

### Repository Structure

```text
.
|-- kmeans_gpu.py       # GPU/CPU KMeans utilities
|-- model.py            # Expression encoders, graph encoders, and relation projector
|-- opt.py              # Command-line arguments
|-- relation.py         # Cell-anchor relation construction, purification, and topology evolution
|-- reliability.py      # Soft assignment, conformal calibration, and reliability estimation
|-- setup.py            # Dataset-specific parameter settings
|-- train.py            # Main training and evaluation pipeline
|-- utils.py            # Data loading, preprocessing, graph construction, losses, and metrics
|-- requirements.txt    # Python dependencies
`-- dataset/            # Dataset folders, each containing data.h5
```

### Requirements

The code is implemented in PyTorch. The following packages are required:

```text
python
torch
numpy
scipy
scikit-learn
h5py
tqdm
```

Install the required packages with:

```bash
pip install -r requirements.txt
```

or:

```bash
pip install torch numpy scipy scikit-learn h5py tqdm
```

Please install a PyTorch build compatible with your CUDA version if GPU acceleration is required.

### Data Format

Each dataset should be placed under:

```text
./dataset/<dataset_name>/data.h5
```

For example:

```text
dataset/
`-- Quake_10x_Bladder/
    `-- data.h5
```

The H5 loader supports common expression-matrix keys including:

```text
exprs
X
x
data
matrix
```

Common label fields include:

```text
Y
y
labels
label
cell_type
cluster
Group
```

Labels are not used as clustering supervision in the optimization objective. By default, the implementation uses the number of unique labels to determine the cluster number `K` when `--cluster_num=-1`; an explicit cluster number can also be supplied through `--cluster_num`.

### Quick Start

Run scRARE on the Bladder dataset:

```bash
python train.py --dataset Quake_10x_Bladder
```

Run a single experiment:

```bash
python train.py --dataset Quake_10x_Bladder --runs 1 --seed 1206
```

Run five experiments with consecutive seeds starting from `1206`:

```bash
python train.py --dataset Quake_10x_Bladder --runs 5 --seed 1206
```

Use a specific GPU:

```bash
python train.py --dataset Quake_10x_Bladder --device cuda:0
```

The final clustering metrics are reported as ACC, NMI, ARI, and AMI. Per-run results are saved to:

```text
result.csv
```

### Reproducing Experimental Results

For each dataset:

1. Place the dataset at:

```text
./dataset/<dataset_name>/data.h5
```

2. Check the corresponding parameter profile in `setup.py`.

3. Run:

```bash
python train.py --dataset <dataset_name> --runs 5
```

4. The program reports the final-model ACC, NMI, ARI, and AMI for each run and summarizes the mean and standard deviation across runs.

The currently configured dataset profiles include:

```text
Muraro
Quake_10x_Bladder
Romanov
Klein
Adam
Quake_10x_Limb_Muscle
10X_PBMC
Plasschaert
Quake_10x_Spleen
Wang_Lung
Chen
```

Additional datasets can be added by following the same profile format in `setup.py`.

### Main Arguments

Common arguments are:

```text
--dataset                     Dataset name under ./dataset/
--device                      auto / cpu / cuda:0
--cluster_num                 Number of clusters K; -1 uses the label-derived K
--runs                        Number of independent runs
--epochs                      Number of training epochs
--seed                        Starting random seed
--lr                          Adam learning rate
--dims                        Cell embedding dimension
--topo_k                      Base-graph neighborhood size
--knn_k                       Auxiliary-graph neighborhood size
--t                           Graph-smoothing propagation order
--beta                        Hard-pair reweighting exponent
--cluster_gamma               Student-t assignment scale
--lambda_r                    Weight of the relation-purification loss
--tau_e                       Sparse-residual soft-threshold
--clustering_update_interval  Clustering and topology refresh interval
--max_grad_norm               Gradient-clipping threshold
```

The topology implementation also provides:

```text
--topology_mode full_chunked
--topology_chunk_size 512
```

for chunked construction of the purified cell graph.

### Method Pipeline

The optimization process can be summarized as:

```text
Raw scRNA-seq counts
        |
        v
Preprocessing + PCA
        |
        +-------------------+
        |                   |
        v                   v
Euclidean base graph   Cosine auxiliary graph
        |                   |
        v                   |
Graph smoothing             |
        |                   |
        v                   v
Two expression encoders + two graph encoders
        |
        v
Fused embedding Z
        |
        v
Soft clustering Q
        |
        v
Conformalized reliability calibration
        |
        v
Node confidence + pair reliability
        |
        +------------------------------+
        |                              |
        v                              v
Reliability-aware              Cell-anchor relation
contrastive learning           construction
                                       |
                                       v
                              Low-rank purification
                                       |
                                       v
                              Purified cell graph
                                       |
                                       v
                         Maturity-driven graph update
                                       |
                                       +----> dynamic graph encoder
```

### Reproducibility Notes

- The model uses two non-shared expression encoders and two non-shared graph encoders.
- The base graph is constructed using Euclidean distance and the auxiliary graph using cosine distance.
- The dynamic graph is initialized from the base graph.
- The clustering state and topology are refreshed at the configured update interval.
- The contrastive temperature is fixed to `1.0`.
- The relation rank is determined by `min(K, d)`.
- Cell anchors and retained topology relations are derived from the base-graph neighborhood size rather than introduced as independent tuning parameters.
- Gradient clipping is enabled by default for numerical stability.
- Final labels are obtained from the final soft assignments using `argmax`.

### Citation

If this repository is useful for your research, please cite:

```bibtex
@article{liu2026scrare,
  title   = {scRARE: Reliability-Aware Relation Evolution for Single-Cell RNA-seq Clustering},
  author  = {Liu, Feng and Zhao, Shangshang and Zhou, Zhongyang and Chen, Feiyu and Zeng, Pan},
  year    = {2026},
  note    = {Preprint}
}
```
