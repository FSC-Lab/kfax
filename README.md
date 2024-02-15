# KFAx: Jax powered EKF

> This is a fork of kfax <https://github.com/Hs293Go/kfax> frozen at the state it is presented in a AER1517 lecture.
> Update this fork as it is used in lectures

## Introduction

KFAX is a small state estimation library for JAX. 
At this point it is little more than an *example library* in the spirit of [`jax.example_libraries.stax`](https://jax.readthedocs.io/en/latest/jax.example_libraries.stax.html) and friends.

Its goal to facilitate quick prototyping of Extended Kalman Filters (EKF) for systems, exploiting [automatic differentiation](https://www.youtube.com/watch?app=desktop&v=wG_nF1awSSY) offered by [JAX](https://github.com/google/jax) to sidestep derivation of Jacobians of the state and observation models.

### Limitations

Despite the infancy of this library several limitations are already identified

- Not speed at this moment.
  - Evaluation of autodiff'ed Jacobian is still more expensive than analytical Jacobians
  - jax's JIT is competitive with MATLAB/Julia's JIT, but they are not rigorously benchmarked against each other
  - Minimal support for **IEKF**, i.e. EKF with states that live on a manifold and innovation/update laws that comply with operations on manifolds

## Dependencies

The primary dependency is **jax** itself. Visit <https://jax.readthedocs.io/en/latest/installation.html> for detailed instructions

## Installation

`cd` to the repository directory and run

``` bash
pip install -e .
```
