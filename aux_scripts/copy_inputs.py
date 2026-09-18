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

os.system("mkdir -p single_inputs/CESM")
os.system("mv single_inputs/SDRBENCH-CESM-ATM-26x1800x3600/ single_inputs/CESM/")

os.system("mkdir -p single_inputs/EXAALT-SDRBENCH-exaalt-copper")
os.system("mv single_inputs/exaalt/* single_inputs/EXAALT-SDRBENCH-exaalt-copper/")

os.system("mkdir -p single_inputs/NYX")
os.system("mv single_inputs/SDRBENCH-EXASKY-NYX-512x512x512/ single_inputs/NYX/")

os.system("mkdir -p single_inputs/QMCPACK-dataset-115x69x69x288")
os.system("mv single_inputs/dataset/einspline_115_69_69_288.f32 single_inputs/QMCPACK-dataset-115x69x69x288/")
os.system("mkdir -p single_inputs/QMCPACK-dataset-288x115x69x69")
os.system("mv single_inputs/dataset/einspline_288_115_69_69.pre.f32 single_inputs/QMCPACK-dataset-288x115x69x69/")

os.system("mkdir -p single_inputs/SCALE-LETKF")
os.system("mv single_inputs/SDRBENCH-SCALE_98x1200x1200/ single_inputs/SCALE-LETKF/")

os.system("mkdir -p single_inputs/ISABEL")
os.system("mv single_inputs/100x500x500/ single_inputs/ISABEL/")

os.system("rm -f sdr.out")
os.system("find single_inputs/ -name *32* >> sdr.out")
sdr_list = open("sdr.out", 'r')

os.system("mkdir -p inputs")

for file in sdr_list:
    new_name = file.replace('\n', '').replace("single_inputs/", "sdrbench-").replace('/', '-')
    print("mv " + file.replace('\n', ' ') + "inputs/" + new_name)
    os.system("mv " + file.replace('\n', ' ') + "inputs/" + new_name)

os.system("rm -rf single_inputs")
os.system("rm -f sdr.out")
