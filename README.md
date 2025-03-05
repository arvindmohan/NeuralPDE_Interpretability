Welcome!

This is a companion repo for the paper What You See is Not What You Get: Neural Partial Differential Equations and The Illusion of Learning: https://arxiv.org/abs/2411.15101
which has been submitted for peer review.
The paper investigates two cases: Burgers equation and gKdV. Each equation has 2 experiments.
This repo provides the codes to train, perform inference on trained models, and perform plotting + analysis of the results.

###Please note### The training codes here were deployed on HPC and took a few hours, so it will take many hours on a laptop.


#Burgers:

A single README.md file inside that dir provides instructions. The inference datasets are too big to include in this repo, so they can 
be found in this FigShare link: https://figshare.com/s/854d5dc6a515ac1ce5ba
Download them and place them in the appropriate folder (Expt 1 or Expt 2) and the codes should work out of the box.

#gKdV:

There is a README.md file inside each dir (Expt1 and Expt2) with detailed instructions on how to run the codes.


#Software Install:

The repo uses Julia language and Python.
The following are the packages in my Julia environment. Please install the same versions to guarantee reproducibility.
This has been tested on both Mac and Unix OS.
Install time: Around 1.5 hours for a fresh Julia v1.9 installation. 
```
(@v1.9) pkg> st
Status `~/.julia/environments/v1.9/Project.toml`
⌃ [fbb218c0] BSON v0.3.7
⌃ [6e4b80f9] BenchmarkTools v1.5.0
⌅ [8e7c35d0] BlockArrays v0.16.25
⌅ [052768ef] CUDA v4.0.1
⌃ [49dc2e85] Calculus v0.5.1
⌃ [d360d2e6] ChainRulesCore v1.25.0
⌃ [aaaa29a8] Clustering v0.15.3
⌃ [96eb917e] ContinuousWavelets v1.1.2
⌅ [aae7a2af] DiffEqFlux v2.0.0
  [41bf760c] DiffEqSensitivity v6.79.0
⌃ [0c46a032] DifferentialEquations v7.7.0
⌃ [b4f34e82] Distances v0.10.8
⌅ [7da242da] Enzyme v0.10.18
⌅ [f151be2c] EnzymeCore v0.1.0
⌃ [7a1cc6ca] FFTW v1.7.0
⌃ [5789e2e9] FileIO v1.16.1
  [26cc04aa] FiniteDifferences v0.12.32
⌅ [587475ba] Flux v0.13.17
⌃ [f6369f11] ForwardDiff v0.10.36
⌅ [28b8d3ca] GR v0.72.7
⌅ [033835bb] JLD2 v0.4.31
⌃ [b964fa9f] LaTeXStrings v1.3.0
⌃ [dbeba491] Metalhead v0.8.3
  [15e1cf62] NPZ v0.4.3
⌃ [429524aa] Optim v1.7.7
⌅ [3bd65402] Optimisers v0.2.20
⌅ [7f7a1694] Optimization v3.14.0
⌅ [36348300] OptimizationOptimJL v0.1.8
⌅ [42dfb2eb] OptimizationOptimisers v0.1.2
⌃ [500b13db] OptimizationPolyalgorithms v0.1.1
⌅ [1dea7af3] OrdinaryDiffEq v6.51.2
⌃ [91a5bcdd] Plots v1.38.16
⌃ [92933f4c] ProgressMeter v1.7.2
  [438e738f] PyCall v1.96.4
⌃ [d330b81b] PyPlot v2.11.2
  [37e2e3b7] ReverseDiff v1.15.3
⌅ [1ed8b502] SciMLSensitivity v7.29.0
⌃ [90137ffa] StaticArrays v1.9.3
⌅ [29a6e085] Wavelets v0.9.5
⌅ [e88e6eb3] Zygote v0.6.62
  [2f01184e] SparseArrays
```
