# Code Examples and Notebooks

This directory contains all companion code for *Engineering Online Experimentation*.

## Setup

**Python Version:** 3.8 or higher

**Core Dependencies:**
```bash
pip install scipy statsmodels pandas numpy matplotlib seaborn jupyter
```

**Data Pipeline Tools** (for relevant chapters):
```bash
pip install dbt-core apache-airflow kafka-python
```

**ML Libraries** (for Part IV):
```bash
pip install scikit-learn ax-platform botorch
```

## Reproducibility

**Important:** Pin your environment to ensure examples run reproducibly. Use the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or create a conda environment:
```bash
conda env create -f environment.yml
conda activate book_online_exp
```

## Directory Structure

*(To be organized by chapter as code is added)*

- `ch2_statistical_theory/` - Statistical foundations and hypothesis testing
- `ch8_pipeline/` - ETL/ELT pipeline examples with dbt
- `ch15_bandits/` - Multi-armed bandit implementations
- `ch18_bayesian_opt/` - Bayesian optimization examples with Ax/BoTorch

## Running Examples

Each chapter directory contains notebooks with detailed explanations. Start Jupyter:

```bash
jupyter notebook
```

Then navigate to the relevant chapter directory.

## Questions or Issues?

Report issues or ask questions in the main repository: [GitHub Issues](https://github.com/thunderbird2009/book_online_exp/issues)
