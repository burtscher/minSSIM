/*
This file is part of the repository for Structural-Similarity-Preserving Lossy Data Compression for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2026, Alex Fallin and Martin Burtscher
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at https://github.com/burtscher/minSSIM.

Publication: This work is described in detail in the following paper.
Alex Fallin and Martin Burtscher. "Structural-Similarity-Preserving Lossy Data Compression for CPUs and GPUs." Proceedings of the 30th Annual IEEE High-Performance Extreme Computing Conference. September 2026.

Sponsor: This work has been supported by the U.S. National Science Foundation (NSF) under Award CCF-2403380, by the Department of Energy (DOE), Office of Science, Advanced Scientific Computing Research (ASCR) under Award DE-SC0022223, and by an equipment donation from NVIDIA Corporation.
*/


#define NDEBUG

using byte = unsigned char;
static const int CS = 1024 * 16;  // chunk size (in bytes) [must be multiple of 8]
static const int TPB = 512;  // threads per block [must be power of 2 and at least 128]

#include <limits>
#include <cmath>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <sys/time.h>
#include "include/macros.h"
#include "preprocessors/h_SSIM3D_ABS_f32.h"
#include "preprocessors/h_SSIM2D_ABS_f32.h"
#include "components/h_DIFFNB_4.h"
#include "components/h_CLOGB_4.h"
#include "components/h_RZE_1.h"


const int NUM_RUNS = 9; // must have a true median
std::vector<double> runtimes;


struct CPUTimer
{
  timeval beg, end;
  CPUTimer() {}
  ~CPUTimer() {}
  void start() {gettimeofday(&beg, NULL);}
  double stop() {gettimeofday(&end, NULL); return end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;}
};


static void h_encode(const byte* const __restrict__ input, const long long insize, byte* const __restrict__ output, long long& outsize)
{
  // initialize
  const long long chunks = (insize + CS - 1) / CS;  // round up
  long long* const head_out = (long long*)output;
  unsigned short* const size_out = (unsigned short*)&head_out[1];
  byte* const data_out = (byte*)&size_out[chunks];
  long long* const carry = new long long [chunks];
  memset(carry, 0, chunks * sizeof(long long));

  // process chunks in parallel
  #pragma omp parallel for schedule(dynamic, 1)
  for (long long chunkID = 0; chunkID < chunks; chunkID++) {
    // load chunk
    long long chunk1 [CS / sizeof(long long)];
    long long chunk2 [CS / sizeof(long long)];
    byte* in = (byte*)chunk1;
    byte* out = (byte*)chunk2;
    const long long base = chunkID * CS;
    const int osize = (int)std::min((long long)CS, insize - base);
    memcpy(out, &input[base], osize);

    // encode chunk
    int csize = osize;
    bool good = true;
    if (good) {
      std::swap(in, out);
      good = h_DIFFNB_4(csize, in, out);
    }
    if (good) {
      std::swap(in, out);
      good = h_CLOGB_4(csize, in, out);
    }
    if (good) {
      std::swap(in, out);
      good = h_RZE_1(csize, in, out);
    }

    // handle carry and store chunk
    long long offs = 0LL;
    if (chunkID > 0) {
      do {
        #pragma omp atomic read
        offs = carry[chunkID - 1];
      } while (offs == 0);
      #pragma omp flush
    }
    if (good && (csize < osize)) {
      // store compressed data
      #pragma omp atomic write
      carry[chunkID] = (offs + (long long)csize);
      size_out[chunkID] = csize;
      memcpy(&data_out[offs], out, csize);
    } else {
      // store original data
      #pragma omp atomic write
      carry[chunkID] = (offs + (long long)osize);
      size_out[chunkID] = osize;
      memcpy(&data_out[offs], &input[base], osize);
    }
  }

  // output header
  head_out[0] = insize;

  // finish
  outsize = &data_out[carry[chunks - 1]] - output;
  delete [] carry;
}


int main(int argc, char* argv [])
{
  printf("CPU LC 1.2 Algorithm: SSIM{2,3}D_ABS_f32 DIFFNB_4 CLOGB_4 RZE_1\n");
  printf("Copyright 2026 Texas State University\n\n");

  // read input from file
  if (argc < 7 || argc > 8) {printf("USAGE: %s input_file_name compressed_file_name eb ssim_b d1 d2 [d3] (in SZ3 order)\n\n", argv[0]); return -1;}
  FILE* const fin = fopen(argv[1], "rb");
  fseek(fin, 0, SEEK_END);
  const long long fsize = ftell(fin);
  if (fsize <= 0) {fprintf(stderr, "ERROR: input file too small\n\n"); throw std::runtime_error("LC error");}
  byte* const input = new byte [fsize];
  fseek(fin, 0, SEEK_SET);
  const long long insize = fread(input, 1, fsize, fin);  assert(insize == fsize);
  fclose(fin);
  printf("original size: %lld bytes\n", insize);

  // allocate CPU memory
  const long long chunks = (insize + CS - 1) / CS;  // round up
  const long long maxsize = 2 * sizeof(long long) + chunks * sizeof(short) + chunks * CS;
  byte* const hencoded = new byte [maxsize];
  long long hencsize = 0;
  byte* hpreencdata = new byte [insize];
  long long hpreencsize = insize;

  // time
  CPUTimer htimer;
  double paramv[] = {atof(argv[3]), atof(argv[4]), atoi(argv[5]), atoi(argv[6]), argc == 8 ? atoi(argv[7]) : -1};
  for (int i = 0; i < NUM_RUNS; i++) {
    std::copy(input, input + insize, hpreencdata);
    if (paramv[4] > 0) {
      htimer.start();
      h_SSIM3D_ABS_f32(hpreencsize, hpreencdata, 5, paramv);
    } else {
      htimer.start();
      h_SSIM2D_ABS_f32(hpreencsize, hpreencdata, 4, paramv);
    }
    h_encode(hpreencdata, hpreencsize, hencoded, hencsize);
    double hruntime = htimer.stop();
    runtimes.push_back(hruntime);
  }

  std::sort(runtimes.begin(), runtimes.end());
  double hthroughput = insize / runtimes[NUM_RUNS / 2] / 1000000000;
  printf("median CPU compression throughput: %8.3f Gbytes/s\n", hthroughput);

  // write to file
  FILE* const fout = fopen(argv[2], "wb");
  fwrite(hencoded, 1, hencsize, fout);
  fclose(fout);

  delete [] input;
  delete [] hencoded;
  return 0;
}
