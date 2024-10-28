# NumericalExperiments

[![Build Status](https://github.com/ngiann/NumericalExperiments.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ngiann/NumericalExperiments.jl/actions/workflows/CI.yml?query=branch%3Amain)


How to define problem for delays:

```
using GPCC, NumericalExperiments

tobs, yobs, σobs, truedelays = simulatetwolightcurves();

logp, pred, unpack = gpccloglikelihood(tobs, yobs, σobs, kernel=GPCC.matern32, maxdelay=10);
```