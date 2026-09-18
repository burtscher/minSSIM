# SBLC and minSSIM Artifact

Artifact for *Structural-Similarity-Preserving Lossy Data Compression for CPUs and GPUs*.
It builds the SBLC compressor (serial, OpenMP, and CUDA) and its baseline, runs them on
six SDRBench single-precision datasets at three NOA error bounds, and produces
`filled_artifact.pdf` containing the results table and the compression-ratio vs.
throughput graphs.


## Requirements

- **Compilers:** g++ with OpenMP, and nvcc. GPU binaries are built with
  `-arch=sm_$(NV_SM)` (e.g. 70 for Compute 7.0)
- **Python 3** with `requests`, `pandas`, and `numpy`
- **LaTeX:** `pdflatex` with `IEEEtran` and `pgfplots`
- **Resources:** NVIDIA GPU (compute capability 7.0+), ~30 GB of network download,
  and ~65 GB of free disk, approx. 1 day


## Running everything

    python3 run_all.py

You can also run the individual steps:

1. `python3 download_inputs.py` - downloads and unpacks the SDRBench inputs into `inputs/`.
   Download time depends on connection speed
2. `python3 run_experiment.py` - builds all executables via `make all` and runs the
   baseline, SBLC GPU, and SBLC CPU (serial + OpenMP) experiments
3. `python3 generate_latex.py` - aggregates the CSVs, fills the LaTeX template `src/empty_artifact.tex`, and compiles
   `filled_artifact.pdf` with the results table and the two graphs similar to how they are in the paper


## Manual usage of the compressor

    make all [NV_SM=<cc>]
    ./sblc_compress_<ser|omp|gpu> input_file compressed_file eb ssim_b d1 d2 [d3]   # dims in row-major order (slowest first)
    ./sblc_decompress_<ser|omp|gpu> compressed_file decompressed_file
    ./float_analysis original_file decompressed_file d1 d2 [d3]   # dims in row-major order (slowest first)

`eb` is the absolute (not NOA) error bound and `ssim_b` the minSSIM target (a value between 0 and 1).


## Publication

If you use minSSIM or SBLC in your work, please cite the following publication:

Alex Fallin and Martin Burtscher. "Structural-Similarity-Preserving Lossy Data Compression for CPUs and GPUs." Proceedings of the 30th Annual IEEE High-Performance Extreme Computing Conference. September 2026. [[paper](https://userweb.cs.txstate.edu/~burtscher/papers/hpec26.pdf)]


*This work has been supported by the U.S. National Science Foundation (NSF) under Award CCF-2403380, by the Department of Energy (DOE), Office of Science, Advanced Scientific Computing Research (ASCR) under Award DE-SC0022223, and by an equipment donation from NVIDIA Corporation.*
