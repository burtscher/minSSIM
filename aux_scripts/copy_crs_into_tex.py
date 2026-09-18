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

import sys
import re
import pandas as pd
import numpy as np


def geometric_mean(values):
    values = np.array(values, dtype=float)
    values = values[values > 0]  # avoid log issues
    if len(values) == 0:
        return np.nan
    return np.exp(np.mean(np.log(values)))


def extract_set_name(col):
    m = re.match(r"sdrbench-([^-]+)-", col)
    return m.group(1) if m else None


def compute_geomeans(csv_path):
    df = pd.read_csv(csv_path)

    error_col = df.columns[0]

    set_columns = {}
    for col in df.columns[1:]:
        set_name = extract_set_name(col)
        if set_name:
            set_columns.setdefault(set_name, []).append(col)

    results = {}

    # Group by error bound
    for error_val, group in df.groupby(error_col):
        per_set_means = []

        for set_name, cols in set_columns.items():
            values = group[cols].values.flatten()
            values = values[~np.isnan(values)]

            if len(values) > 0:
                gm = geometric_mean(values)
                per_set_means.append(gm)

        if per_set_means:
            overall_gm = geometric_mean(per_set_means)
            results[error_val] = overall_gm

    return results


def format_tag(prefix, error_val):
    s = str(error_val)

    if "." in s:
        decimals = s.split(".")[1]
        return f"{prefix}_{decimals}"
    else:
        return f"{prefix}_{s}"


def replace_tags(template_path, output_path, results, prefix):
    with open(template_path, "r") as f:
        text = f.read()

    for error_val, gm in results.items():
        tag = format_tag(prefix, error_val)
        text = text.replace(tag, f"{gm:.6f}")

    with open(output_path, "w") as f:
        f.write(text)


def main():
    if len(sys.argv) != 5:
        print("Usage: python script.py <csv_file> <template_file> <output_file> <tag_prefix>")
        sys.exit(1)

    csv_path = sys.argv[1]
    template_path = sys.argv[2]
    output_path = sys.argv[3]
    prefix = sys.argv[4]

    results = compute_geomeans(csv_path)

    print("Computed geometric means:")
    for k, v in results.items():
        print(f"{k}: {v}")

    replace_tags(template_path, output_path, results, prefix)


if __name__ == "__main__":
    main()