# PatriBERT

Training and exploration of ancient greek and latin PatriBERT models (LLM specialized on Church Fathers literature).

## Overview

This repository contains scripts and notebooks for:
- Domain-adaptive pretraining of BERT models on Latin and Ancient Greek corpora
- Fine-tuning/evaluation workflows
- Attention and hidden-state exploration with BertViz

The goal is to build and analyze PatriBERT variants specialized for patristic literature.

## Repository Structure

### Training and fine-tuning scripts
- `BERT_pretraining.py` — generic pretraining workflow
- `Latin-PatriBERT_training.py` — Latin model adaptation/training pipeline
- `Greek-PatriBERT_training.py` — Ancient Greek model adaptation/training pipeline
- `PatriBERT_finetuning.py` — downstream fine-tuning script

### Exploration notebooks
- `BertViz_Latin-PatriBERT.ipynb` — tokenization, hidden states, and attention visualization
- `Archives/` — archived notebooks and experiments

### Data
- `processed_corpus/latin_corpus.txt`
- `processed_corpus/latin_corpus_i&u.txt`
- `processed_corpus/greek_corpus_unaccented.txt`
- `processed_corpus/greek_patristic_corpus_deaccented.txt`
- `processed_corpus/VG.tsv`

### Model artifacts
- `model/` — exported/finalized model directories
- `latin-bert-adapted/` — training checkpoints from adaptation runs

## Typical Workflow

1. Prepare corpus files in `processed_corpus/`.
2. Run language-specific training (`Latin-PatriBERT_training.py` or `Greek-PatriBERT_training.py`).
3. Save/adapt checkpoints and export final model to `model/`.
4. Use `BertViz_Latin-PatriBERT.ipynb` for qualitative analysis of tokenizer behavior, hidden states, and attention.
5. Run `PatriBERT_finetuning.py` for downstream tasks.

## Notes

- Training scripts are designed around Hugging Face `transformers` + `datasets`.
- Existing checkpoints and model folders are included for reproducibility and comparison across runs.
