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
import pandas as pd
import re

def main():
  if len(sys.argv) != 4:
    print("Usage: python script.py <csv_file> <prefix> <tex_file>")
    sys.exit(1)

  csv_file = sys.argv[1]
  prefix = sys.argv[2]
  tex_file = sys.argv[3]

  df = pd.read_csv(csv_file)

  required_cols = {"Level", "Suite", "SSIM", "minSSIM"}
  if not required_cols.issubset(df.columns):
      raise ValueError(f"CSV must contain columns: {required_cols}")

  grouped = (
      df.groupby("Level")[["SSIM", "minSSIM"]]
      .mean()
      .reset_index()
  )

  replacements = {}
  for _, row in grouped.iterrows():
      level = str(row["Level"])
      lvl = level.split(".")[1] if "." in level else level
      for col, tag in [("SSIM", "SSIM"), ("minSSIM", "mSSIM")]:
          key = f"{prefix}_{tag}_{lvl}"
          value = f"{row[col]:.4f}"
          replacements[key] = value

  with open(tex_file, "r", encoding="utf-8") as f:
      tex_content = f.read()

  updated_content = tex_content
  for key in sorted(replacements, key=len, reverse=True):
      updated_content = updated_content.replace(key, replacements[key])

  with open(tex_file, "w", encoding="utf-8") as f:
      f.write(updated_content)

if __name__ == "__main__":
    main()