# Epistemic Gain Benchmark

**A quantitative framework for measuring "Genuine Knowledge Discovery" in Large Language Models.**

This repository provides a rigorous experimental setup to investigate whether LLMs can move beyond deductive closure (following existing paths) and perform **inductive reasoning** (inferring latent generative rules). It implements a **Stochastic Block Model (SBM)** world generator and a strictly defined evaluation metric based on **Epistemic Gain ($\Delta$)**.


## Research Motivation

Standard benchmarks often test an LLM's ability to retrieve training data or perform logical deduction ($A \to B \to C$). However, genuine scientific discovery requires **Induction** and **Abduction**: inferring a general rule from noisy, incomplete observations.

This project defines "New Knowledge Discovery" mathematically:

1.  **Novelty ($N=1$):** The proposed proposition $r$ is *not* logically entailed by the observed context $O$ ($r \notin C(O)$).
2.  **Truth ($T=1$):** The proposition matches the verifiable ground truth $G$ of the synthetic world.
3.  **Epistemic Value ($\Delta > \tau$):** The proposition increases the predictive capacity of the system on held-out hidden facts ($H$).

$$\text{Discovery}(r) \iff \text{Novelty}(r) \land \text{Truth}(r) \land (\Delta > 0)$$


## The Experiment: Latent Structure World

To ensure **Novelty**, we do not use static logic puzzles. Instead, we generate a "World" using a **Stochastic Block Model (SBM)**.

* **The World:** Nodes belong to hidden **Latent Classes**.
* **The Physics:** Interactions between nodes are governed by a hidden **Probabilistic Interaction Matrix** (e.g., Class A triggers Class B 85% of the time).
* **The Task:** The LLM observes a list of interactions ($O$) with masked class labels. It must propose the hidden rule or predict missing interactions in the held-out set ($H$).

Because the graph is probabilistic and generated with random UUIDs, the specific solution does not exist in the LLM's training data, and the rule cannot be derived via pure deduction.


## Getting Started

### 1. Installation
Requires Python 3.12+ and standard data science libraries.

```bash
git clone [https://github.com/your-username/epistemic-gain-benchmark.git](https://github.com/your-username/epistemic-gain-benchmark.git)
cd epistemic-gain-benchmark
pip install networkx numpy scikit-learn
```