
# Multilingual Semantic Similarity & Translation Benchmark

This project compares multilingual language models for **sentence similarity** and **machine translation quality** across several bilingual corpora.  
It was developed as part of a university NLP course project and serves as a compact research benchmark.

---

## Overview

The code evaluates **SentenceTransformer**, **XLM-RoBERTa**, and **mBART-50** models on multilingual datasets:
- **Europarl (EN–PL)** – formal, political domain  
- **Tatoeba (EN–DE)** – short, dictionary-like sentences  
- **TED Talks (EN–DE)** – conversational, spontaneous speech

Each experiment computes:
- Sentence-level **cosine similarities**
- Translation metrics: **BLEU**, **chrF**, and **BERTScore**
- Correlation between semantic and translation scores
- Visualization plots (PCA, t-SNE, correlation heatmaps)

---

## Project Documentation

A detailed analysis, results, visualizations, and conclusions are available in the PDF report:  
👉 [**project_documentation.pdf**](project_documentation.pdf)

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/ATMScorp/nlp-multilingual-benchmark.git
cd nlp-multilingual-benchmark
```

### 2️⃣ Install dependencies
Make sure you have **Python 3.9+** installed, then run:
```bash
pip install -r requirements.txt
```

### 3️⃣ (Optional) Prepare directory structure
```bash
bash scripts/setup_dirs.sh
```
This creates the `cache/` and `results/` folders if they don’t exist.

---

## Data Setup

This repository **does not include datasets** due to size and licensing restrictions.

To reproduce the experiments, download or prepare the following files:

```
data/
├── europarl/
│   ├── en-pl.en
│   └── en-pl.pl
├── tatoeba/
│   └── en-de.tsv
└── ted2020/
    ├── en-de.en.txt
    └── en-de.de.txt
```

You can obtain the datasets from:

- [Europarl corpus](https://www.statmt.org/europarl/)
- [Tatoeba Challenge](https://tatoeba.org/en/downloads)
- [TED2020 parallel data (OPUS)](https://opus.nlpl.eu/TED2020/en&de/v1/TED2020)

If you prefer a small test run, you can create text files with a few English–German or English–Polish sentence pairs using the same format.

---

## Running the Experiments

Once the data is ready, run:

```bash
python main.py
```

This will:
- Load data from the `data/` folder  
- Compute embeddings, similarities, and translation metrics  
- Save CSV results and plots to `results/`  
- Cache model outputs to speed up future runs  

---

## Outputs

All generated results are stored under `results/`:
```
results/
├── csv/
│   ├── results_*.csv
│   ├── correlation_*.csv
│   ├── translation_metrics_*.csv
│   └── benchmark_results.csv
└── plots/
    ├── similarity/
    ├── pca/
    ├── tsne/
    └── correlation/
```

You’ll find:
- Similarity distributions  
- PCA / t-SNE embeddings visualization  
- Correlation heatmaps  
- Benchmark tables comparing model performance  

---

## Models Used

| Model | Description | Source |
|-------|--------------|--------|
| **SentenceTransformer** | DistilUSE Base Multilingual | `distiluse-base-multilingual-cased` |
| **XLM-RoBERTa** | Paraphrase-Multilingual-MPNet Base | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| **mBART-50** | Multilingual Encoder-Decoder Translation Model | `facebook/mbart-large-50-many-to-many-mmt` |

---

## Metrics

| Metric | Description |
|---------|-------------|
| **Cosine Similarity** | Sentence-level semantic similarity between embeddings |
| **BLEU** | n-gram overlap measure for translation |
| **chrF** | Character-level F-score for translation |
| **BERTScore** | Semantic similarity between reference and hypothesis translations |
| **Correlation Matrix** | Relationship between different metric families |

---

## Author

Developed by **Aliaksandr Trukhanovich**  
as part of university coursework on **Natural Language Processing (2025)**.


---
