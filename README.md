# MFVB Bayesian Linear Regression

This project initializes a research-oriented baseline for **Mean-Field Variational Bayes (MFVB)** in **Bayesian Linear Regression (BLR)**, based on the ideas in the source document.

## Project goal

Build and validate a closed-form coordinate-ascent MFVB implementation for Bayesian linear regression using **synthetic seeding data**.

Core validation targets:
- monotonic ELBO improvement
- recovery of true regression coefficients
- recovery of true noise precision
- reproducible synthetic experiments

## What is included

- closed-form MFVB updates for Gaussian-Gamma BLR
- synthetic data generation
- ELBO computation and convergence monitoring
- a runnable demo script
- a small regression test suite

## Project structure

- [mfvb_blr/models.py](mfvb_blr/models.py) — dataclasses for priors, configs, datasets, and results
- [mfvb_blr/data.py](mfvb_blr/data.py) — synthetic BLR data generation
- [mfvb_blr/mfvb.py](mfvb_blr/mfvb.py) — coordinate-ascent MFVB implementation
- [scripts/run_blr_demo.py](scripts/run_blr_demo.py) — demo experiment and ELBO plot
- [tests/test_mfvb.py](tests/test_mfvb.py) — basic correctness checks

## Model

We use:

$$
y \mid X, \beta, \tau \sim \mathcal{N}(X\beta, \tau^{-1} I)
$$

with priors

$$
\beta \sim \mathcal{N}(m_0, S_0^{-1}), \qquad \tau \sim \mathrm{Gamma}(a_0, b_0)
$$

and a mean-field variational posterior

$$
q(\beta, \tau) = q(\beta) q(\tau)
$$

where
- $q(\beta)$ is Gaussian
- $q(\tau)$ is Gamma

## Quick start

Install the existing requirements, then run:

- Demo: [scripts/run_blr_demo.py](scripts/run_blr_demo.py)
- Tests: [tests/test_mfvb.py](tests/test_mfvb.py)

## Suggested next research steps

To turn this baseline into a stronger research project, extend it toward one of these directions:
- linear-response corrections for covariance recovery
- sparse priors such as Horseshoe or spike-and-slab variants
- robust variational Bayes under contamination or outliers
- high-dimensional regimes with $p \gg n$
- richer variational families beyond mean field
