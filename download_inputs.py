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
try:
    import requests
    import numpy
except ImportError as e:
    sys.exit(f"ERROR: missing required python module: {e.name}. "
             "This script needs 'requests' and 'numpy' (see README).")
import re
from urllib.parse import urljoin
import os
import time
import shutil
import tarfile
import subprocess

def unzip_files(directory="."):
    for filename in os.listdir(directory):
        if filename.endswith(".tar.gz"):
            print("Extracting", filename)
            subprocess.run(["tar", "-xvzf", os.path.join(directory, filename), "-C", directory])
            print("Extracted", filename)

def get_links_from_page(url):
    try:
        response = requests.get(url, timeout=30)
    except requests.RequestException as e:
        sys.exit(f"ERROR: could not fetch {url} ({type(e).__name__}). "
                 "Check this machine's internet access/proxy settings.")

    if response.status_code == 200:
        links = re.findall(r'href=[\'"]?([^\'" >]+)', response.text)

        extracted_links = [urljoin(url, link) for link in links]

        return extracted_links
    else:
        print("Failed to fetch page. Status code:", response.status_code)
        return []

def download_file(url, directory="."):
    filename = url.split("/")[-1]
    path = os.path.join(directory, filename)
    total = int(requests.head(url, timeout=30).headers["Content-Length"])
    for attempt in range(1, 21):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have == total:
            return
        if have > total:
            os.remove(path)
            have = 0
        try:
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, stream=True, headers=headers, timeout=60) as r:
                r.raise_for_status()
                with open(path, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
        except requests.RequestException as e:
            print(f"  download of {filename} interrupted ({type(e).__name__}), retrying ({attempt}/20)...")
            time.sleep(5)
    sys.exit(f"ERROR: could not finish downloading {url} after 20 attempts")

def delete_files_with_word(directory, word):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if word in filename:
                filepath = os.path.join(root, filename)
                try:
                       os.remove(filepath)
                       print("Deleted", filepath)
                except Exception as e:
                       print(f"Error deleting {filepath}: {e}")


def delete_gz_files(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".gz"):
                filepath = os.path.join(root, filename)
                try:
                    os.remove(filepath)
                    print("Deleted", filepath)
                except Exception as e:
                    print(f"Error deleting {filepath}: {e}")

def verify_inputs():
    """Check that inputs/ contains every file the experiment scripts expect."""
    import re
    expected = re.findall(r'\["(sdrbench-[^"]+)"', open("aux_scripts/run_base_gpu.py").read())
    missing = [name for name in expected if not os.path.exists(os.path.join("inputs", name))]
    if missing:
        print(f"ERROR: {len(missing)} of {len(expected)} expected input files are missing from inputs/:")
        for name in missing:
            print("  ", name)
        print("A previous download may have been interrupted. "
              "Delete the single_inputs/ and inputs/ folders and rerun this script.")
        sys.exit(1)
    print(f"All {len(expected)} expected input files are present in inputs/")

url = "https://sdrbench.github.io/"

keywords = ["SDRBENCH-CESM-ATM-26x1800x3600.tar",
            "SDRBENCH-EXAALT-copper.tar",
            "SDRBENCH-Hurricane-ISABEL-100x500x500.tar",
            "SDRBENCH-EXASKY-NYX-512x512x512.tar",
            "SDRBENCH-SCALE-98x1200x1200.tar",
            "SDRBENCH-QMCPack.tar"]

folder_names = ["SDRBENCH-CESM-ATM-26x1800x3600",
                "exaalt",
                "100x500x500",
                "SDRBENCH-EXASKY-NYX-512x512x512",
                "SDRBENCH-SCALE_98x1200x1200",
                "dataset"]

print("Downloading inputs, this step may take a significant amount of time...")

keywords_tmp = list()
for idx, fname in enumerate(folder_names):
    if not os.path.exists(f"./single_inputs/{fname}"):
        keywords_tmp.append(keywords[idx])

keywords = keywords_tmp

if len(keywords) == 0:
    print("All inputs already downloaded!")
    verify_inputs()
    quit()

pages = [url]
for link in get_links_from_page(url):
    if link.startswith(url) and link.endswith(".html") and link not in pages:
        pages.append(link)
links = []
for page in pages:
    links.extend(get_links_from_page(page))

matched = set()
for link in links:
    if link.endswith('.gz'):
        for keyword in keywords:
            if keyword in link:
                matched.add(keyword)
                filename = link.split("/")[-1]
                if os.path.exists(os.path.join("single_inputs", filename)):
                    print("Already downloaded", filename)
                else:
                    print("Downloading", link)
                    download_file(link)
                    print("Downloaded", link)
                break

unmatched = [k for k in keywords if k not in matched]
if unmatched:
    print(f"ERROR: found no download links for {len(unmatched)} dataset(s) on {url}:")
    for k in unmatched:
        print("  ", k)
    print("This usually means this machine cannot reach the download page or a web\n"
          "proxy/firewall is intercepting the request (the page returned no .gz links).\n"
          "Check connectivity with:  curl -s https://sdrbench.github.io/ | grep -c tar.gz\n"
          "Alternatively, download the missing SDRBench .tar.gz files on another machine\n"
          "and place them in this folder, then rerun this script.")
    sys.exit(1)

if not os.path.exists("single_inputs"):
    os.makedirs("single_inputs")
for filename in os.listdir("."):
    if filename.endswith(".gz"):
        shutil.move(filename, os.path.join("single_inputs", filename))

unzip_files("single_inputs")
delete_gz_files("single_inputs")
delete_files_with_word("single_inputs", "log")
delete_files_with_word("single_inputs", ".txt")

if os.path.exists("single_inputs/SDRBENCH-QMCPack") and not os.path.exists("single_inputs/dataset"):
    os.rename("single_inputs/SDRBENCH-QMCPack", "single_inputs/dataset")

os.system("mv single_inputs/dataset/*/* single_inputs/dataset/ 2> /dev/null")

orig = "single_inputs/dataset/einspline_115_69_69_288.f32"
pre = "single_inputs/dataset/einspline_288_115_69_69.pre.f32"
if os.path.exists(orig) and not os.path.exists(pre):
    np = numpy
    data = np.fromfile(orig, dtype=np.float32).reshape(115, 69, 69, 288)
    np.moveaxis(data, 3, 0).tofile(pre)
    print("Generated", pre)

base_dir = "single_inputs"

folders_to_move = [
    "SDRBENCH-exaalt-copper",
    "SDRBENCH-exaalt-helium",
    "2869440"
    ]
exaalt_folder = os.path.join(base_dir, "exaalt")
os.makedirs(exaalt_folder, exist_ok=True)

for folder in folders_to_move:
    folder_path = os.path.join(base_dir, folder)
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                shutil.move(os.path.join(root, file), os.path.join(exaalt_folder, file))

for folder in folders_to_move:
    folder_path = os.path.join(base_dir, folder)
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)

os.system("python3 aux_scripts/copy_inputs.py")

verify_inputs()
