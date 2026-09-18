# This file is part of the repository for Structural-Similarity-Preserving Lossy Data Compression for CPUs and GPUs.
#
# BSD 3-Clause License
#
# Copyright (c) 2021-2026, Alex Fallin and Martin Burtscher
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# URL: The latest version of this code is available at https://github.com/burtscher/minSSIM.
#
# Publication: This work is described in detail in the following paper.
# Alex Fallin and Martin Burtscher. "Structural-Similarity-Preserving Lossy Data Compression for CPUs and GPUs." Proceedings of the 30th Annual IEEE High-Performance Extreme Computing Conference. September 2026.
#
# Sponsor: This work has been supported by the U.S. National Science Foundation (NSF) under Award CCF-2403380, by the Department of Energy (DOE), Office of Science, Advanced Scientific Computing Research (ASCR) under Award DE-SC0022223, and by an equipment donation from NVIDIA Corporation.

import os

print("Beginning LaTeX generation")

# CR
os.system("python3 aux_scripts/copy_crs_into_tex.py res/minSSIM_compression_ratios.csv src/empty_artifact.tex res/filled_artifact.tex ser_cr")
os.system("python3 aux_scripts/copy_crs_into_tex.py res/minSSIM_compression_ratios.csv res/filled_artifact.tex res/filled_artifact.tex omp_cr")
os.system("python3 aux_scripts/copy_crs_into_tex.py res/minSSIM_compression_ratios.csv res/filled_artifact.tex res/filled_artifact.tex cuda_cr")
os.system("python3 aux_scripts/copy_crs_into_tex.py res/base_compression_ratios.csv res/filled_artifact.tex res/filled_artifact.tex base_cr")

# SSIM
os.system("python3 aux_scripts/parse_float_analysis.py res/minSSIM_float_analysis.csv res/minSSIM_float_analysis_parsed.csv")
os.system("python3 aux_scripts/parse_float_analysis.py res/base_float_analysis.csv res/base_float_analysis_parsed.csv")
os.system("python3 aux_scripts/copy_ssims_into_tex.py res/minSSIM_float_analysis_parsed.csv SBLC res/filled_artifact.tex")
os.system("python3 aux_scripts/copy_ssims_into_tex.py res/base_float_analysis_parsed.csv BASE res/filled_artifact.tex")

# Throughput
os.system("python3 aux_scripts/copy_tps_into_tex.py res/minSSIM_NOA_0.01_ser.csv res/filled_artifact.tex ser_Xtp_01 -o res/filled_artifact.tex")
os.system("python3 aux_scripts/copy_tps_into_tex.py res/minSSIM_NOA_0.001_ser.csv res/filled_artifact.tex ser_Xtp_001 -o res/filled_artifact.tex")
os.system("python3 aux_scripts/copy_tps_into_tex.py res/minSSIM_NOA_0.0001_ser.csv res/filled_artifact.tex ser_Xtp_0001 -o res/filled_artifact.tex")
os.system("python3 aux_scripts/copy_tps_into_tex.py res/minSSIM_NOA_0.01_omp.csv res/filled_artifact.tex omp_Xtp_01 -o res/filled_artifact.tex")
os.system("python3 aux_scripts/copy_tps_into_tex.py res/minSSIM_NOA_0.001_omp.csv res/filled_artifact.tex omp_Xtp_001 -o res/filled_artifact.tex")
os.system("python3 aux_scripts/copy_tps_into_tex.py res/minSSIM_NOA_0.0001_omp.csv res/filled_artifact.tex omp_Xtp_0001 -o res/filled_artifact.tex")
os.system("python3 aux_scripts/copy_tps_into_tex.py res/minSSIM_NOA_0.01_cuda.csv res/filled_artifact.tex cuda_Xtp_01 -o res/filled_artifact.tex")
os.system("python3 aux_scripts/copy_tps_into_tex.py res/minSSIM_NOA_0.001_cuda.csv res/filled_artifact.tex cuda_Xtp_001 -o res/filled_artifact.tex")
os.system("python3 aux_scripts/copy_tps_into_tex.py res/minSSIM_NOA_0.0001_cuda.csv res/filled_artifact.tex cuda_Xtp_0001 -o res/filled_artifact.tex")
os.system("python3 aux_scripts/copy_tps_into_tex.py res/base_NOA_0.01_cuda.csv res/filled_artifact.tex base_Xtp_01 -o res/filled_artifact.tex")
os.system("python3 aux_scripts/copy_tps_into_tex.py res/base_NOA_0.001_cuda.csv res/filled_artifact.tex base_Xtp_001 -o res/filled_artifact.tex")
os.system("python3 aux_scripts/copy_tps_into_tex.py res/base_NOA_0.0001_cuda.csv res/filled_artifact.tex base_Xtp_0001 -o res/filled_artifact.tex")

# Generate PDF
os.system("pdflatex -interaction=nonstopmode res/filled_artifact.tex")
os.system("rm filled_artifact.log")
os.system("rm filled_artifact.aux")

print("Result PDF generated")
