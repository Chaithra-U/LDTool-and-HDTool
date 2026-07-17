# Scalable extraction and visualization of multi-attribute logical and functional dependencies in tabular data

This repository presents **LDTool** and **HLDTool**, two frameworks for extracting **multi-attribute logical dependencies (LDs)** and **functional dependencies (FDs)** from tabular data. 

### LDTool
- Performs extraction of multi-attribute LDs and FDs for low-dimensional data.

### HLDTool
- Performs scalable dependency extraction using hypergraph-guided search-space reduction for high-dimensional data

---

## Repository Structure

```
datasets/
    Preprocessed datasets used in the paper.

dependencies/
    Extracted dependencies for all datasets
    METABRIC/
        Dependencies extracted using different hyperedge sizes

hyperedges/
        Extracted hyperedges using different hyperedge sizes

qMatrix_computation/
        qMatrix.py will compute the dependency matrix using VPTree and a brute-force approach
        qFunction/
            Functions to compute qMatrix using both approaches

src/
    ldtool.py                # Multi-attribute LD and FD extraction
    hldtool.py               # Hypergraph-guided dependency extraction
    ldtool_q_matrix.py       # First-layer dependency extraction on pre computed $Q$-matrix

visualization/
    example.ipynb            # Example dependency visualization
    functions.py             # Visualization functions
```

---

## Running the Code

### LDTool (Low-dimensional datasets)

```bash
python src/ldtool.py --input <dataset.csv> --max-layer 3 --type both --output <dependencies.txt>
```

### HLDTool (High-dimensional datasets)

```bash
python src/hldtool.py --input <dataset.csv> --hyperedges <hyperedges.csv> --max-layer 3 --type both --output <dependencies.txt>
```

### First-layer Dependency Extraction

```bash
python src/ldtool_q_matrix.py --input <matrix.csv> --type both --output <dependencies.txt>
```

---

## Main Parameters

The following parameters can be modified at the beginning of each script.

| Parameter | Description |
|----------|-------------|
| `--input` | Path to the input CSV file. |
| `--output` | Path to the output text file (optional). |
| `--type` | Dependency type to extract: `fd`, `ld`, or `both` (default). |
| `--max-layer` | Maximum number of features allowed in the left-hand side (default: `3`). |
| `--min-strength` | Minimum dependency strength required to retain a dependency (default: `0.8`). |
| `--ld-threshold` | Minimum improvement threshold for retaining higher-order LDs (default: `0.2`). |
| `--drop-first-column` | Ignore the first column (useful for identifier columns). |

Default settings were used throughout the experiments reported in the manuscript.

---

## Datasets

The `datasets` folder contains the preprocessed datasets used in the paper.

| Dataset | Source |
|---------|--------|
| Shopping Behavior | [Kaggle](https://www.kaggle.com/datasets/sahilislam007/shopping-trends-and-customer-behaviour-dataset) |
| Adult | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult) |
| Online Shopping | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) |
| Migraine | [Code Ocean](https://codeocean.com/capsule/1269964/tree/v1/data/migraine.csv) |
| Mushroom | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/73/mushroom) |
| Global House Purchase | [Kaggle](https://www.kaggle.com/datasets/mohankrishnathalla/global-house-purchase-decision-dataset) |
| Consumer Shopping Trends | [Kaggle](https://www.kaggle.com/datasets/sohaibdevv/consumer-shopping-behavior-and-preference-study-2026) |
| METABRIC | [Kaggle](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric) |
| US Census | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/116/us+census+data+1990) |
| TIC Insurance | [UCI Machine Learning Repository]([https://archive.ics.uci.edu/dataset/73/mushroom](https://archive.ics.uci.edu/dataset/125/insurance+company+benchmark+coil+2000)) |
| Student Digital Behavior | [Kaggle](https://www.kaggle.com/datasets/nitikachandel95/student-social-media-impact-dataset?select=global_student_digital_behavior_dataset.csv) |

---

## Visualization

The `visualization` folder contains:

- `example.ipynb` – Example notebook for visualizing extracted dependencies.
- `functions.py` – Utility functions used for dependency visualization.

---
