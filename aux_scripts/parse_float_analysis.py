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
import csv
from collections import defaultdict

def process_chunk(lines):
    suites = defaultdict(lambda: {
        "MSE": [],
        "PSNR": [],
        "SSIM": [],
        "minSSIM": []
    })

    current_suite = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("sdrbench-"):
            parts = line.split("-")
            if len(parts) > 1:
                current_suite = parts[1]
            else:
                current_suite = None
            continue

        if current_suite:
            if "MSE:" in line:
                suites[current_suite]["MSE"].append(float(line.split(":")[1]))

            elif "PSNR:" in line:
                suites[current_suite]["PSNR"].append(float(line.split(":")[1]))

            elif "SSIM:" in line and "minSSIM" not in line:
                suites[current_suite]["SSIM"].append(float(line.split(":")[1]))

            elif "minSSIM:" in line:
                suites[current_suite]["minSSIM"].append(float(line.split(":")[1]))

    return suites


def average_metrics(suites):
    result = {}
    for suite, metrics in suites.items():
        result[suite] = {}
        for metric, values in metrics.items():
            if values:
                result[suite][metric] = sum(values) / len(values)
            else:
                result[suite][metric] = None
    return result


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    levels = ["0.01", "0.001", "0.0001"]

    with open(input_file, "r") as f:
        lines = f.readlines()

    n = len(lines)
    chunk_size = n // 3

    all_results = []

    for i in range(3):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < 2 else n

        chunk = lines[start:end]
        suites = process_chunk(chunk)
        averages = average_metrics(suites)

        for suite, metrics in averages.items():
            row = {
                "Level": levels[i],
                "Suite": suite,
                **metrics
            }
            all_results.append(row)

    # Write CSV
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Level", "Suite", "MSE", "PSNR", "SSIM", "minSSIM"])

        print("\nAverages per suite and level:\n")

        for row in all_results:
            print(f"Level: {row['Level']}, Suite: {row['Suite']}")
            print(f"  MSE: {row['MSE']}")
            print(f"  PSNR: {row['PSNR']}")
            print(f"  SSIM: {row['SSIM']}")
            print(f"  minSSIM: {row['minSSIM']}\n")

            writer.writerow([
                row["Level"],
                row["Suite"],
                row["MSE"],
                row["PSNR"],
                row["SSIM"],
                row["minSSIM"]
            ])

    print(f"CSV written to: {output_file}")


if __name__ == "__main__":
    main()