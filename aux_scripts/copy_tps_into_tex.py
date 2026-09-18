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

import argparse
import csv
import math


def geomean(values):
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def extract_set_name(label):
    if not label.startswith("sdrbench-"):
        return None

    parts = label.split("-")
    if len(parts) < 2:
        return None

    return parts[1]  # e.g. CESM


def parse_csv(path):
    sets = {}

    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    i = 0
    while i < len(rows):
        row = rows[i]
        if not row:
            i += 1
            continue

        label = row[0].strip()

        if label.startswith("sdrbench-"):
            set_name = extract_set_name(label)
            if set_name is None:
                i += 1
                continue

            sets.setdefault(set_name, {"comp": [], "decomp": []})

            try:
                comp_row = rows[i + 1]
                decomp_row = rows[i + 2]

                # lines look like: "median CPU compression throughput:    0.058 Gbytes/s"
                comp_val = float(comp_row[0].split(":")[1].split()[0])
                decomp_val = float(decomp_row[0].split(":")[1].split()[0])

                sets[set_name]["comp"].append(comp_val)
                sets[set_name]["decomp"].append(decomp_val)

                i += 3
                continue

            except (IndexError, ValueError):
                i += 1
                continue

        i += 1

    return sets


def compute_set_geomeans(sets):
    result = {}
    for name, vals in sets.items():
        result[name] = (
            geomean(vals["comp"]),
            geomean(vals["decomp"]),
        )
    return result


def compute_overall_geomean(set_geomeans):
    comp_vals = [v[0] for v in set_geomeans.values()]
    decomp_vals = [v[1] for v in set_geomeans.values()]

    return geomean(comp_vals), geomean(decomp_vals)


def replace_tags(template_path, output_path, tag_base, comp_val, decomp_val):
    with open(template_path, "r") as f:
        content = f.read()

    comp_tag = tag_base.replace("X", "c")
    decomp_tag = tag_base.replace("X", "d")

    content = content.replace(comp_tag, f"{comp_val:.6g}")
    content = content.replace(decomp_tag, f"{decomp_val:.6g}")

    with open(output_path, "w") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file")
    parser.add_argument("template_file")
    parser.add_argument("tag_base", help="e.g. foo_X_bar where X will be replaced with c or d for comrpession and decompression repsectively")
    parser.add_argument("-o", "--output", default="output.txt")

    args = parser.parse_args()

    sets = parse_csv(args.csv_file)
    set_geomeans = compute_set_geomeans(sets)
    comp_gm, decomp_gm = compute_overall_geomean(set_geomeans)

    replace_tags(
        args.template_file,
        args.output,
        args.tag_base,
        comp_gm,
        decomp_gm,
    )

    print("Per-set geometric means:")
    for s, (c, d) in set_geomeans.items():
        print(f"{s}: comp={c:.6g}, decomp={d:.6g}")

    print("\nOverall geometric means:")
    print(f"Compression: {comp_gm:.6g}")
    print(f"Decompression: {decomp_gm:.6g}")


if __name__ == "__main__":
    main()