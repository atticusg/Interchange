# Causal Abstractions of Neural Natural Language Inference Models

This is the implementation for the experiments in the paper *Causal Abstractions of Neural Natural Language Inference Models*.

## Setup and dependencies

See `requirements.txt`.

## Repository Structure

### Code folders

- `intervention/` Basic infrastructure for defining computation graphs and performing interventions.
- `compgraphs/` Computation graphs for Natural Language Inference causal models and neural models. 
- `causal_abstraction/` Interchange experiments and analysis.
- `datasets/` Class definitions for datasets.
- `modeling/` Neural models for NLI and training code.
- `probing/` Probing experiments.
- `experiment/` Utilities for launching experiments and automatically recording experiment results in databases.
- `feature_importance/` Utilities for integrated gradients experiments.

### Scripts

**Training models**

- `train_bert.py` and `train_lstm.py`. Train one instance of a model.
- `train_manager.py`. Utilities for interfacing with the `experiment` module and managing grid search training.

**Interchange experiments**

- `interchange.py`. Run one set of an interchange experiment on a given causal model intermediate node, and all neural model locations for that node. Analyze the success rates of interventions.
- `graph_analysis.py`. Composes the graph linking the examples after interchange experiments and finds cliques.
- `interchange_manager.py`. Utilities for interfacing with the `experiment` module and run large batches of interchange experiments on a computing cluster.

**Probing experiments**
- `probe.py`
- `probe_manager.py`




