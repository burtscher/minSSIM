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

application_name = "minSSIM"

inputs = [
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-CLDICE_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-CLDLIQ_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-CLOUD_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-CMFDQ_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-CMFDQR_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-CMFDT_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-CONCLD_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-DCQ_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-DTCOND_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-DTV_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-FICE_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-GCLDLWP_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-ICIMR_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-ICLDIWP_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-ICLDTWP_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-ICWMR_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-OMEGA_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-OMEGAT_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-Q_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-QC_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-QRL_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-QRS_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-RELHUM_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-T_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-U_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-UU_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-V_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-VD01_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-VQ_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-VT_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-VU_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-VV_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-Z3_1_26_1800_3600.f32", "3600 1800 26", 168480000],
    ["sdrbench-EXAALT-SDRBENCH-exaalt-copper-dataset1-5423x3137.x.f32.dat", " 3137 5423", 17011951],
    ["sdrbench-EXAALT-SDRBENCH-exaalt-copper-dataset1-5423x3137.y.f32.dat", " 3137 5423", 17011951],
    ["sdrbench-EXAALT-SDRBENCH-exaalt-copper-dataset1-5423x3137.z.f32.dat", " 3137 5423", 17011951],
    ["sdrbench-EXAALT-SDRBENCH-exaalt-copper-dataset2-83x1077290.x.f32.dat", " 1077290 83", 89415070],
    ["sdrbench-EXAALT-SDRBENCH-exaalt-copper-dataset2-83x1077290.y.f32.dat", " 1077290 83", 89415070],
    ["sdrbench-EXAALT-SDRBENCH-exaalt-copper-dataset2-83x1077290.z.f32.dat", " 1077290 83", 89415070],
    ["sdrbench-NYX-SDRBENCH-EXASKY-NYX-512x512x512-baryon_density.f32", "512 512 512", 134217728],
    ["sdrbench-NYX-SDRBENCH-EXASKY-NYX-512x512x512-dark_matter_density.f32", "512 512 512", 134217728],
    ["sdrbench-NYX-SDRBENCH-EXASKY-NYX-512x512x512-temperature.f32", "512 512 512", 134217728],
    ["sdrbench-NYX-SDRBENCH-EXASKY-NYX-512x512x512-velocity_x.f32", "512 512 512", 134217728],
    ["sdrbench-NYX-SDRBENCH-EXASKY-NYX-512x512x512-velocity_y.f32", "512 512 512", 134217728],
    ["sdrbench-NYX-SDRBENCH-EXASKY-NYX-512x512x512-velocity_z.f32", "512 512 512", 134217728],
    ["sdrbench-QMCPACK-dataset-115x69x69x288-einspline_115_69_69_288.f32", "69 69 33120", 157684320],
    ["sdrbench-QMCPACK-dataset-288x115x69x69-einspline_288_115_69_69.pre.f32", "69 69 33120", 157684320],
    ["sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-PRES-98x1200x1200.f32", "1200 1200 98", 141120000],
    ["sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-QC-98x1200x1200.f32", "1200 1200 98", 141120000],
    ["sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-QG-98x1200x1200.f32", "1200 1200 98", 141120000],
    ["sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-QI-98x1200x1200.f32", "1200 1200 98", 141120000],
    ["sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-QR-98x1200x1200.f32", "1200 1200 98", 141120000],
    ["sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-QS-98x1200x1200.f32", "1200 1200 98", 141120000],
    ["sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-QV-98x1200x1200.f32", "1200 1200 98", 141120000],
    ["sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-RH-98x1200x1200.f32", "1200 1200 98", 141120000],
    ["sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-T-98x1200x1200.f32", "1200 1200 98", 141120000],
    ["sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-U-98x1200x1200.f32", "1200 1200 98", 141120000],
    ["sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-V-98x1200x1200.f32", "1200 1200 98", 141120000],
    ["sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-W-98x1200x1200.f32", "1200 1200 98", 141120000],
    ["sdrbench-ISABEL-100x500x500-CLOUDf48.bin.f32", "500 500 100", 25000000],
    ["sdrbench-ISABEL-100x500x500-Pf48.bin.f32", "500 500 100", 25000000],
    ["sdrbench-ISABEL-100x500x500-PRECIPf48.bin.f32", "500 500 100", 25000000],
    ["sdrbench-ISABEL-100x500x500-QCLOUDf48.bin.f32", "500 500 100", 25000000],
    ["sdrbench-ISABEL-100x500x500-QGRAUPf48.bin.f32", "500 500 100", 25000000],
    ["sdrbench-ISABEL-100x500x500-QICEf48.bin.f32", "500 500 100", 25000000],
    ["sdrbench-ISABEL-100x500x500-QRAINf48.bin.f32", "500 500 100", 25000000],
    ["sdrbench-ISABEL-100x500x500-QSNOWf48.bin.f32", "500 500 100", 25000000],
    ["sdrbench-ISABEL-100x500x500-QVAPORf48.bin.f32", "500 500 100", 25000000],
    ["sdrbench-ISABEL-100x500x500-TCf48.bin.f32", "500 500 100", 25000000],
    ["sdrbench-ISABEL-100x500x500-Uf48.bin.f32", "500 500 100", 25000000],
    ["sdrbench-ISABEL-100x500x500-Vf48.bin.f32", "500 500 100", 25000000],
    ["sdrbench-ISABEL-100x500x500-Wf48.bin.f32", "500 500 100", 25000000]
]

adj_eb_data = {
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-CLDICE_1_26_1800_3600.f32": {
        0.0001: 7.855e-09,
        0.001: 7.8552e-08,
        0.01: 7.85517e-07,
        0.1: 7.85517e-06,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-CLDLIQ_1_26_1800_3600.f32": {
        0.0001: 5.2058e-08,
        0.001: 5.20582e-07,
        0.01: 5.20582e-06,
        0.1: 5.20582e-05,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-CLOUD_1_26_1800_3600.f32": {
        0.0001: 9.64057e-05,
        0.001: 0.000964057,
        0.01: 0.009640567,
        0.1: 0.096405677,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-CMFDQR_1_26_1800_3600.f32": {
        0.0001: 4e-12,
        0.001: 3.5e-11,
        0.01: 3.53e-10,
        0.1: 3.534e-09,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-CMFDQ_1_26_1800_3600.f32": {
        0.0001: 2.3e-11,
        0.001: 2.29e-10,
        0.01: 2.286e-09,
        0.1: 2.2864e-08,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-CMFDT_1_26_1800_3600.f32": {
        0.0001: 4.4219e-08,
        0.001: 4.42194e-07,
        0.01: 4.42194e-06,
        0.1: 4.42194e-05,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-CONCLD_1_26_1800_3600.f32": {
        0.0001: 4.90213e-05,
        0.001: 0.000490213,
        0.01: 0.004902129,
        0.1: 0.049021292,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-DCQ_1_26_1800_3600.f32": {
        0.0001: 2.8e-11,
        0.001: 2.79e-10,
        0.01: 2.786e-09,
        0.1: 2.7857e-08,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-DTCOND_1_26_1800_3600.f32": {
        0.0001: 5.8211e-08,
        0.001: 5.8211e-07,
        0.01: 5.8211e-06,
        0.1: 5.8211e-05,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-DTV_1_26_1800_3600.f32": {
        0.0001: 1.46403e-07,
        0.001: 1.46403e-06,
        0.01: 1.46403e-05,
        0.1: 0.000146403,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-FICE_1_26_1800_3600.f32": {
        0.0001: 0.0001,
        0.001: 0.001,
        0.01: 0.01,
        0.1: 0.100000001,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-GCLDLWP_1_26_1800_3600.f32": {
        0.0001: 0.01673116,
        0.001: 0.167311609,
        0.01: 1.673115969,
        0.1: 16.73116112,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-ICIMR_1_26_1800_3600.f32": {
        0.0001: 7.47724e-07,
        0.001: 7.47725e-06,
        0.01: 7.47724e-05,
        0.1: 0.000747724,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-ICLDIWP_1_26_1800_3600.f32": {
        0.0001: 0.179931983,
        0.001: 1.799319983,
        0.01: 17.99319839,
        0.1: 179.9319916,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-ICLDTWP_1_26_1800_3600.f32": {
        0.0001: 0.872288287,
        0.001: 8.722883224,
        0.01: 87.22882843,
        0.1: 872.288269,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-ICWMR_1_26_1800_3600.f32": {
        0.0001: 2.13442e-06,
        0.001: 2.13442e-05,
        0.01: 0.000213441,
        0.1: 0.002134415,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-OMEGAT_1_26_1800_3600.f32": {
        0.0001: 0.103959896,
        0.001: 1.039599061,
        0.01: 10.39598942,
        0.1: 103.9598999,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-OMEGA_1_26_1800_3600.f32": {
        0.0001: 0.000408747,
        0.001: 0.004087474,
        0.01: 0.040874735,
        0.1: 0.408747345,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-QC_1_26_1800_3600.f32": {
        0.0001: 1e-11,
        0.001: 9.8e-11,
        0.01: 9.8e-10,
        0.1: 9.801e-09,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-QRL_1_26_1800_3600.f32": {
        0.0001: 3.4156e-08,
        0.001: 3.41564e-07,
        0.01: 3.41564e-06,
        0.1: 3.41564e-05,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-QRS_1_26_1800_3600.f32": {
        0.0001: 6.881e-09,
        0.001: 6.8808e-08,
        0.01: 6.88084e-07,
        0.1: 6.88084e-06,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-Q_1_26_1800_3600.f32": {
        0.0001: 1.9756e-06,
        0.001: 1.9756e-05,
        0.01: 0.00019756,
        0.1: 0.001975604,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-RELHUM_1_26_1800_3600.f32": {
        0.0001: 0.009960339,
        0.001: 0.099603407,
        0.01: 0.996033967,
        0.1: 9.9603405,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-T_1_26_1800_3600.f32": {
        0.0001: 0.012222584,
        0.001: 0.122225851,
        0.01: 1.222258449,
        0.1: 12.22258472,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-UU_1_26_1800_3600.f32": {
        0.0001: 1.118150949,
        0.001: 11.18150997,
        0.01: 111.815094,
        0.1: 1118.151001,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-U_1_26_1800_3600.f32": {
        0.0001: 0.015895164,
        0.001: 0.158951655,
        0.01: 1.589516401,
        0.1: 15.89516449,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-VD01_1_26_1800_3600.f32": {
        0.0001: 7.2e-11,
        0.001: 7.21e-10,
        0.01: 7.212e-09,
        0.1: 7.2117e-08,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-VQ_1_26_1800_3600.f32": {
        0.0001: 3.36999e-05,
        0.001: 0.000336999,
        0.01: 0.00336999,
        0.1: 0.033699904,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-VT_1_26_1800_3600.f32": {
        0.0001: 2.763654709,
        0.001: 27.636549,
        0.01: 276.365448,
        0.1: 2763.654785,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-VU_1_26_1800_3600.f32": {
        0.0001: 0.903428674,
        0.001: 9.034287453,
        0.01: 90.34287262,
        0.1: 903.4287109,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-VV_1_26_1800_3600.f32": {
        0.0001: 0.513634682,
        0.001: 5.136346817,
        0.01: 51.36346436,
        0.1: 513.6347046,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-V_1_26_1800_3600.f32": {
        0.0001: 0.012561152,
        0.001: 0.125611529,
        0.01: 1.256115198,
        0.1: 12.56115246,
    },
    "sdrbench-CESM-SDRBENCH-CESM-ATM-26x1800x3600-Z3_1_26_1800_3600.f32": {
        0.0001: 3.904702187,
        0.001: 39.04702377,
        0.01: 390.4702148,
        0.1: 3904.702393,
    },
    "sdrbench-EXAALT-SDRBENCH-exaalt-copper-dataset1-5423x3137.x.f32.dat": {
        0.0001: 0.004699293,
        0.001: 0.046992935,
        0.01: 0.469929308,
        0.1: 4.699293137,
    },
    "sdrbench-EXAALT-SDRBENCH-exaalt-copper-dataset1-5423x3137.y.f32.dat": {
        0.0001: 0.004699375,
        0.001: 0.046993751,
        0.01: 0.469937474,
        0.1: 4.699374676,
    },
    "sdrbench-EXAALT-SDRBENCH-exaalt-copper-dataset1-5423x3137.z.f32.dat": {
        0.0001: 0.004626749,
        0.001: 0.046267491,
        0.01: 0.462674886,
        0.1: 4.626749039,
    },
    "sdrbench-EXAALT-SDRBENCH-exaalt-copper-dataset2-83x1077290.x.f32.dat": {
        0.0001: 0.000100235,
        0.001: 0.001002351,
        0.01: 0.010023514,
        0.1: 0.100235142,
    },
    "sdrbench-EXAALT-SDRBENCH-exaalt-copper-dataset2-83x1077290.y.f32.dat": {
        0.0001: 0.000100209,
        0.001: 0.001002087,
        0.01: 0.010020874,
        0.1: 0.100208752,
    },
    "sdrbench-EXAALT-SDRBENCH-exaalt-copper-dataset2-83x1077290.z.f32.dat": {
        0.0001: 7.1679e-05,
        0.001: 0.00071679,
        0.01: 0.007167902,
        0.1: 0.071679018,
    },
    "sdrbench-ISABEL-100x500x500-CLOUDf48.bin.f32": {
        0.0001: 2.04795e-07,
        0.001: 2.04795e-06,
        0.01: 2.04795e-05,
        0.1: 0.000204795,
    },
    "sdrbench-ISABEL-100x500x500-PRECIPf48.bin.f32": {
        0.0001: 7.50586e-07,
        0.001: 7.50586e-06,
        0.01: 7.50586e-05,
        0.1: 0.000750586,
    },
    "sdrbench-ISABEL-100x500x500-Pf48.bin.f32": {
        0.0001: 0.663613856,
        0.001: 6.636138916,
        0.01: 66.36138153,
        0.1: 663.6138916,
    },
    "sdrbench-ISABEL-100x500x500-QCLOUDf48.bin.f32": {
        0.0001: 2.04795e-07,
        0.001: 2.04795e-06,
        0.01: 2.04795e-05,
        0.1: 0.000204795,
    },
    "sdrbench-ISABEL-100x500x500-QGRAUPf48.bin.f32": {
        0.0001: 7.29519e-07,
        0.001: 7.29519e-06,
        0.01: 7.29519e-05,
        0.1: 0.000729519,
    },
    "sdrbench-ISABEL-100x500x500-QICEf48.bin.f32": {
        0.0001: 8.4636e-08,
        0.001: 8.46364e-07,
        0.01: 8.46364e-06,
        0.1: 8.46364e-05,
    },
    "sdrbench-ISABEL-100x500x500-QRAINf48.bin.f32": {
        0.0001: 6.33465e-07,
        0.001: 6.33465e-06,
        0.01: 6.33465e-05,
        0.1: 0.000633465,
    },
    "sdrbench-ISABEL-100x500x500-QSNOWf48.bin.f32": {
        0.0001: 8.5603e-08,
        0.001: 8.56026e-07,
        0.01: 8.56026e-06,
        0.1: 8.56026e-05,
    },
    "sdrbench-ISABEL-100x500x500-QVAPORf48.bin.f32": {
        0.0001: 2.04207e-06,
        0.001: 2.04207e-05,
        0.01: 0.000204207,
        0.1: 0.002042072,
    },
    "sdrbench-ISABEL-100x500x500-TCf48.bin.f32": {
        0.0001: 0.010620113,
        0.001: 0.10620115,
        0.01: 1.062011361,
        0.1: 10.62011433,
    },
    "sdrbench-ISABEL-100x500x500-Uf48.bin.f32": {
        0.0001: 0.009258077,
        0.001: 0.092580765,
        0.01: 0.925807655,
        0.1: 9.258076668,
    },
    "sdrbench-ISABEL-100x500x500-Vf48.bin.f32": {
        0.0001: 0.009370042,
        0.001: 0.093700431,
        0.01: 0.937004209,
        0.1: 9.370042801,
    },
    "sdrbench-ISABEL-100x500x500-Wf48.bin.f32": {
        0.0001: 0.001657436,
        0.001: 0.01657436,
        0.01: 0.165743589,
        0.1: 1.657436013,
    },
    "sdrbench-NYX-SDRBENCH-EXASKY-NYX-512x512x512-baryon_density.f32": {
        0.0001: 11.5862236,
        0.001: 115.8622437,
        0.01: 1158.622314,
        0.1: 11586.22363,
    },
    "sdrbench-NYX-SDRBENCH-EXASKY-NYX-512x512x512-dark_matter_density.f32": {
        0.0001: 1.377893448,
        0.001: 13.77893543,
        0.01: 137.7893372,
        0.1: 1377.893433,
    },
    "sdrbench-NYX-SDRBENCH-EXASKY-NYX-512x512x512-temperature.f32": {
        0.0001: 478.0302429,
        0.001: 4780.302734,
        0.01: 47803.02344,
        0.1: 478030.25,
    },
    "sdrbench-NYX-SDRBENCH-EXASKY-NYX-512x512x512-velocity_x.f32": {
        0.0001: 8228.364258,
        0.001: 82283.64063,
        0.01: 822836.375,
        0.1: 8228364.0,
    },
    "sdrbench-NYX-SDRBENCH-EXASKY-NYX-512x512x512-velocity_y.f32": {
        0.0001: 10043.85352,
        0.001: 100438.5391,
        0.01: 1004385.313,
        0.1: 10043854.0,
    },
    "sdrbench-NYX-SDRBENCH-EXASKY-NYX-512x512x512-velocity_z.f32": {
        0.0001: 7232.423828,
        0.001: 72324.24219,
        0.01: 723242.375,
        0.1: 7232424.0,
    },
    "sdrbench-QMCPACK-dataset-115x69x69x288-einspline_115_69_69_288.f32": {
        0.0001: 0.003301002,
        0.001: 0.033010017,
        0.01: 0.330100179,
        0.1: 3.301001787,
    },
    "sdrbench-QMCPACK-dataset-288x115x69x69-einspline_288_115_69_69.pre.f32": {
        0.0001: 0.003301002,
        0.001: 0.033010017,
        0.01: 0.330100179,
        0.1: 3.301001787,
    },
    "sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-PRES-98x1200x1200.f32": {
        0.0001: 9.953475952,
        0.001: 99.53475952,
        0.01: 995.3475342,
        0.1: 9953.475586,
    },
    "sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-QC-98x1200x1200.f32": {
        0.0001: 3.01153e-07,
        0.001: 3.01153e-06,
        0.01: 3.01153e-05,
        0.1: 0.000301153,
    },
    "sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-QG-98x1200x1200.f32": {
        0.0001: 1.48813e-06,
        0.001: 1.48813e-05,
        0.01: 0.000148813,
        0.1: 0.001488131,
    },
    "sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-QI-98x1200x1200.f32": {
        0.0001: 1.6005e-07,
        0.001: 1.6005e-06,
        0.01: 1.6005e-05,
        0.1: 0.00016005,
    },
    "sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-QR-98x1200x1200.f32": {
        0.0001: 6.37207e-07,
        0.001: 6.37207e-06,
        0.01: 6.37207e-05,
        0.1: 0.000637207,
    },
    "sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-QS-98x1200x1200.f32": {
        0.0001: 7.5697e-08,
        0.001: 7.56969e-07,
        0.01: 7.56969e-06,
        0.1: 7.56969e-05,
    },
    "sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-QV-98x1200x1200.f32": {
        0.0001: 1.95234e-06,
        0.001: 1.95234e-05,
        0.01: 0.000195234,
        0.1: 0.001952336,
    },
    "sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-RH-98x1200x1200.f32": {
        0.0001: 0.020447293,
        0.001: 0.204472944,
        0.01: 2.044729233,
        0.1: 20.44729424,
    },
    "sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-T-98x1200x1200.f32": {
        0.0001: 0.013271483,
        0.001: 0.132714838,
        0.01: 1.327148199,
        0.1: 13.27148342,
    },
    "sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-U-98x1200x1200.f32": {
        0.0001: 0.011843096,
        0.001: 0.118430957,
        0.01: 1.184309483,
        0.1: 11.84309578,
    },
    "sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-V-98x1200x1200.f32": {
        0.0001: 0.009957608,
        0.001: 0.099576086,
        0.01: 0.995760798,
        0.1: 9.957608223,
    },
    "sdrbench-SCALE-LETKF-SDRBENCH-SCALE_98x1200x1200-W-98x1200x1200.f32": {
        0.0001: 0.006389257,
        0.001: 0.063892581,
        0.01: 0.638925791,
        0.1: 6.389257908,
    },
}

# If the inputs are not stored in the same folder
input_pref = "inputs/"
# Can be used if not automatic, will be used to find the created files so must match any auto
# Final file name will be FILE.comp_suff.recon_suff
comp_suff = ".comp"
recon_suff = ".recon"

# Ideally organized by error type and bound for easy copy/paste
cr_out = application_name + "_compression_ratios.csv"
float_analysis_out = application_name + "_float_analysis.csv"

error_bounds = [0.01, 0.001, 0.0001]

# Compress then reconstruct all files
with open(cr_out, 'w') as cr_outfile:
    cr_outfile.write(",")
    for input in inputs:
        cr_outfile.write(input[0] + ",")
    cr_outfile.write("\n")
    for eb in error_bounds:
        # Write row label
        cr_outfile.write(str(eb) + ",")
        omp_tp_outname = application_name + "_NOA_" + str(eb) + "_omp.csv"
        ser_tp_outname = application_name + "_NOA_" + str(eb) + "_ser.csv"
        os.system("rm -f " + omp_tp_outname)
        os.system("rm -f " + ser_tp_outname)
        for input in inputs:
            os.system("echo \"" + (input[0] + ",") + (str(input[2]) + "\"") + " >> " + omp_tp_outname)
            os.system("echo \"" + (input[0] + ",") + (str(input[2]) + "\"") + " >> " + ser_tp_outname)
            # This changes for each application
            ser_comp_string = "./sblc_compress_ser " + input_pref + input[0] + " " + input[0] + comp_suff + " " + str(adj_eb_data[input[0]][eb]) + " 0.85 " + input[1] + " | grep throughput >> " + ser_tp_outname
            omp_comp_string = "./sblc_compress_omp " + input_pref + input[0] + " " + input[0] + comp_suff + " " + str(adj_eb_data[input[0]][eb]) + " 0.85 " + input[1] + " | grep throughput >> " + omp_tp_outname
            ser_recon_string = "./sblc_decompress_ser " + input[0] + comp_suff + " " + input[0] + comp_suff + recon_suff + " | grep throughput >> " + ser_tp_outname
            omp_recon_string = "./sblc_decompress_omp " + input[0] + comp_suff + " " + input[0] + comp_suff + recon_suff + " | grep throughput >> " + omp_tp_outname
            os.system(omp_comp_string)
            os.system(omp_recon_string)
            os.system(ser_comp_string)
            os.system(ser_recon_string)
            # Calculate compression ratios
            orig_size = os.stat(input_pref + input[0]).st_size
            comp_size = os.stat(input[0] + comp_suff).st_size
            cr_outfile.write(str(orig_size / comp_size) + ",")
            # Run float analysis on all recon files
            os.system("echo \"" + input[0] + "\" >> " + float_analysis_out)
            os.system("./float_analysis " + input_pref + input[0] + " " + input[0] + comp_suff + recon_suff + " " + input[1] + " | grep -e 'NOA' -e 'MSE' -e 'PSNR' -e 'SSIM' >> " + float_analysis_out)
            # Delete compressed and decompressed files to avoid size problems
            os.system("rm -f " + input[0] + comp_suff)
            os.system("rm -f " + input[0] + comp_suff + recon_suff)
        cr_outfile.write("\n")