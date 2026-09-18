/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2026, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, Anju Mongandampulath Akathoott, and Martin Burtscher
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

URL: The latest version of this code is available at https://github.com/burtscher/LC-framework.

Sponsor: This code is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under contract DE-SC0022223.
*/


#include "../include/d_ssim.h"


static inline __device__ int d_SSIM2D_ABS_f32_shift(const unsigned int val, const int thr_e)
{
  constexpr const int e = 8;  // exponent bits
  constexpr const int m = 23;  // mantissa bits
  const int val_e = (val >> m) & ((1 << e) - 1);  // extract exponent
  int shift = 0;  // default is lossless
  if ((val_e < thr_e) && (val != 0)) {  // below threshold and not zero
    shift = thr_e - val_e;  // bias cancels out
  }
  return shift;
}


static inline __device__ unsigned int d_SSIM2D_ABS_f32_encode(const unsigned int val, const int shift)
{
  constexpr const int m = 23;  // mantissa bits
  unsigned int enc = val;  // default is lossless
  if (shift > 0) {  // below threshold and not zero
    if (shift <= m + 1) {  // lossy encoding (m + 1 = thr_e - eb_e)
      enc = val + (1 << (shift - 1));  // round to nearest, ties round away from zero
      if ((enc ^ val) >= (1 << m)) {  // check for mantissa overflow
        const unsigned int mant = enc & ~(-1 << m);  // extract mantissa
        enc &= (unsigned int)(-1 << m);  // keep only sign and exponent
        enc |= mant >> 1;  // reinsert corrected mantissa
      }
      enc &= -1 << shift;  // zero out unnecessary bits
    } else {  // quantize to zero
      enc = 0;
    }
  }
  return enc;
}


static __global__ void d_SSIM2D_ABS_f32_init(const int len, unsigned int* const __restrict__ data_u, unsigned int* const __restrict__ new_data_u, signed char* const __restrict__ shift, signed char* const __restrict__ new_shift, const int thr_e, int* const __restrict__ timestamp)
{
  const int idx = threadIdx.x + blockIdx.x * TPB;
  if (idx < len) {
    unsigned int val = data_u[idx];
    if (val == (1u << 31)) {
      data_u[idx] = val = 0;  // -0 --> +0
    }
    const int shft = d_SSIM2D_ABS_f32_shift(val, thr_e);
    new_data_u[idx] = d_SSIM2D_ABS_f32_encode(val, shft);
    shift[idx] = shft;
    new_shift[idx] = shft;
    timestamp[idx] = 1;
  }
}


static __global__ void d_SSIM2D_ABS_f32_compute(const int num_wins, const int flat0, const int start0, const int start1, const int dim1, const int dim2, const unsigned int* const __restrict__ data_u, unsigned int* const __restrict__ new_data_u, const signed char* const __restrict__ shift, signed char* const __restrict__ new_shift, int* const __restrict__ timestamp, const int iter, const float ssimbound, const float* const __restrict__ data_f, const float* const __restrict__ new_data_f, bool* const __restrict__ go_again)
{
  const int idx = threadIdx.x + blockIdx.x * TPB;
  if (idx < num_wins) {
    const int i_1 = idx / flat0;
    const int i_0 = idx % flat0;

    const int offset1 = start1 + i_1 * SSIM_WINDOW_INCR;
    const int offset0 = start0 + i_0 * SSIM_WINDOW_INCR;

    const int win_index = offset0 + dim1 * offset1;
    // Check local SSIM
    if (timestamp[win_index] >= iter - 1) {  // window was modified in the last iteration or this one
      int num_ssim_low = 0;
      while (ssimbound > d_SSIM_2d_calcWindow_float(data_f, new_data_f, dim1, offset0, offset1, SSIM_WINDOW_SIZE, SSIM_WINDOW_SIZE)) {
        d_ssim_atomicWrite(go_again, true);
        num_ssim_low++;
        for (int i1 = offset1; i1 < offset1 + SSIM_WINDOW_SIZE; i1++) {
          for (int i0 = offset0; i0 < offset0 + SSIM_WINDOW_SIZE; i0++) {
            const int index = i0 + dim1 * i1;
            const signed char ns = min((int)new_shift[index], max(0, shift[index] - num_ssim_low));
            new_shift[index] = ns;
            new_data_u[index] = d_SSIM2D_ABS_f32_encode(data_u[index], ns);
          }
        }
      }
      // add this window and all overlapping if there was a change made (window's origin is top left corner of window, so just a cube centered at offset{0,1,2}
      if ((iter > 1) && (num_ssim_low != 0)) {
        const int dist = SSIM_WINDOW_SIZE - (SSIM_WINDOW_SIZE % SSIM_WINDOW_SHIFT);
        for (int other_1 = max(offset1 - dist, 0); other_1 <= min(offset1 + dist, dim2 - SSIM_WINDOW_SIZE); other_1 += SSIM_WINDOW_SHIFT) {
          for (int other_0 = max(offset0 - dist, 0); other_0 <= min(offset0 + dist, dim1 - SSIM_WINDOW_SIZE); other_0 += SSIM_WINDOW_SHIFT) {
            const int nw_index = other_0 + dim1 * other_1;
            d_ssim_atomicWrite(&timestamp[nw_index], iter);
          }
        }
      }
    }
  }
}


static inline void d_SSIM2D_ABS_f32(long long& size, byte*& data, const int paramc, const double paramv [])
{
  static_assert(sizeof(float) == sizeof(unsigned int));
  assert(size < (1u << 31));

  constexpr const int e = 8;  // exponent bits
  constexpr const int m = 23;  // mantissa bits
  if (size % sizeof(float) != 0) {fprintf(stderr, "SSIM2D_ABS_f32: ERROR: size of input must be a multiple of %ld bytes\n", sizeof(float)); throw std::runtime_error("LC error");}
  const int len = size / sizeof(float);
  if (paramc != 4) {fprintf(stderr, "USAGE: SSIM2D_ABS_f32(error_bound, min_ssim, dim1, dim2)\n"); throw std::runtime_error("LC error");}
  const float ssimbound = paramv[1];  // convert to float
  const int dim1 = paramv[2];
  const int dim2 = paramv[3];
  if ((dim1 * dim2) != len) {fprintf(stderr, "SSIM2D_ABS_f32: ERROR: the dimensions must match the input, dim1 %d and dim2 %d do not match input size %d\n", dim1, dim2, len); throw std::runtime_error("LC error");}
  const float errorbound = (float)paramv[0];
  if (errorbound < 2 * std::numeric_limits<float>::min()) {fprintf(stderr, "SSIM2D_ABS_f32: ERROR: error_bound is too small\n"); throw std::runtime_error("LC error");}  // minimum positive normalized value

  const int eb_e = (*((int*)&errorbound) >> m) & ((1 << e) - 1);  // extract biased exponent
  const int thr_e = eb_e + (m + 1);  // biased exponent of threshold
  if (thr_e >= (1 << e) - 1) {fprintf(stderr, "SSIM2D_ABS_f32: ERROR: normalized error_bound is too large\n"); throw std::runtime_error("LC error");}

  float* d_data_f = (float*)data;
  unsigned int* d_data_u = (unsigned int*)data;
  unsigned int* d_new_data_u;
  cudaMalloc((void**)&d_new_data_u, len * sizeof(unsigned int));
  float* d_new_data_f = (float*)d_new_data_u;
  signed char* d_shift;
  cudaMalloc((void**)&d_shift, len * sizeof(signed char));
  signed char* d_new_shift;
  cudaMalloc((void**)&d_new_shift, len * sizeof(signed char));
  int* d_timestamp;
  cudaMalloc((void**)&d_timestamp, len * sizeof(int));
  bool* d_go_again;
  cudaMalloc((void**)&d_go_again, sizeof(bool));

  d_SSIM2D_ABS_f32_init<<<(len + TPB - 1) / TPB, TPB>>>(len, d_data_u, d_new_data_u, d_shift, d_new_shift, thr_e, d_timestamp);

  int iter = 1;
  bool go_again = false;
  do {
    cudaMemsetAsync(d_go_again, 0, sizeof(bool));
    for (int start1 = 0; start1 < SSIM_WINDOW_SIZE; start1 += SSIM_WINDOW_SHIFT) {
      for (int start0 = 0; start0 < SSIM_WINDOW_SIZE; start0 += SSIM_WINDOW_SHIFT) {
        const int flat0 = ((dim1 - SSIM_WINDOW_SIZE) - start0) / SSIM_WINDOW_INCR + 1;
        const int flat1 = ((dim2 - SSIM_WINDOW_SIZE) - start1) / SSIM_WINDOW_INCR + 1;
        const int num_wins = flat0 * flat1;

        d_SSIM2D_ABS_f32_compute<<<(num_wins + TPB - 1) / TPB, TPB>>>(num_wins, flat0, start0, start1, dim1, dim2, d_data_u, d_new_data_u, d_shift, d_new_shift, d_timestamp, iter, ssimbound, d_data_f, d_new_data_f, d_go_again);
      }
    }
    iter++;
    cudaMemcpy(&go_again, d_go_again, sizeof(bool), cudaMemcpyDeviceToHost);
  } while (go_again);

  cudaFree(data);
  cudaFree(d_shift);
  cudaFree(d_new_shift);
  cudaFree(d_timestamp);
  cudaFree(d_go_again);
  data = (byte*)d_new_data_u;
}


static inline void d_iSSIM2D_ABS_f32(long long& size, byte*& data, const int paramc, const double paramv [])
{
  // nothing to be done
}
