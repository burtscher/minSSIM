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


#ifndef LC_DSSIM_H_
#define LC_DSSIM_H_


#include <cuda/std/limits>
#include <cuda/atomic>
#include "ssim.h"


template <typename T>
__device__ inline void d_ssim_atomicWrite(T* const addr, const T val)
{
  ((cuda::atomic<T>*)addr)->store(val, cuda::memory_order_relaxed);
}


static __device__ float d_SSIM_2d_calcWindow_float(const float* const orig_data, const float* const recon_data, const size_t size0, const int offset0, const int offset1, const int windowSize0 = 7, const int windowSize1 = 7)
{
  constexpr const float K1 = 0.01;
  constexpr const float K2 = 0.03;
  constexpr const float K1K1 = K1 * K1;
  constexpr const float K2K2 = K2 * K2;

  const size_t index = offset0 + size0 * offset1;
  float orig_Min = orig_data[index];
  float orig_Max = orig_data[index];
  float orig_Sum = 0;
  float recon_Sum = 0;
  const int np = windowSize0 * windowSize1;

  for (int i1 = offset1; i1 < offset1 + windowSize1; i1++) {
    for (int i0 = offset0; i0 < offset0 + windowSize0; i0++) {
      const size_t index = i0 + size0 * i1;
      const float orig = orig_data[index];
      const float recon = recon_data[index];
      orig_Min = min(orig_Min, orig);
      orig_Max = max(orig_Max, orig);
      orig_Sum += orig;
      recon_Sum += recon;
    }
  }

  const float orig_Mean = orig_Sum / np;
  const float recon_Mean = recon_Sum / np;
  float var_orig = 0, var_recon = 0, var_orig_recon = 0;

  for (int i1 = offset1; i1 < offset1 + windowSize1; i1++) {
    for (int i0 = offset0; i0 < offset0 + windowSize0; i0++) {
      const size_t index = i0 + size0 * i1;
      const float orig = orig_data[index];
      const float recon = recon_data[index];
      const float orig_diff = orig - orig_Mean;
      const float recon_diff = recon - recon_Mean;
      var_orig += orig_diff * orig_diff;
      var_recon += recon_diff * recon_diff;
      var_orig_recon += orig_diff * recon_diff;
    }
  }

  var_orig /= np;
  var_recon /= np;
  var_orig_recon /= np;

  const float orig_Sigma = sqrtf(var_orig);
  const float recon_Sigma = sqrtf(var_recon);
  const float orig_recon_Cov = var_orig_recon;

  const float orig_range = orig_Max - orig_Min;
  const float orig_range2 = orig_range * orig_range;
  const float c1 = max(K1K1 * orig_range2, 2 * cuda::std::numeric_limits<float>::min());
  const float c2 = max(K2K2 * orig_range2, 2 * cuda::std::numeric_limits<float>::min());
  const float c3 = c2 / 2;

  const float or_re_Sigma = orig_Sigma * recon_Sigma;
  const float luminance = (2 * orig_Mean * recon_Mean + c1) / (orig_Mean * orig_Mean + recon_Mean * recon_Mean + c1);
  const float contrast = (2 * or_re_Sigma + c2) / (var_orig + var_recon + c2);
  const float structure = (orig_recon_Cov + c3) / (or_re_Sigma + c3);
  const float ssim = luminance * contrast * structure;

  return ssim;
}


static __device__ float d_SSIM_3d_calcWindow_float(const float* const orig_data, const float* const recon_data, const size_t size1, const size_t size0, const int offset0, const int offset1, const int offset2, const int windowSize0 = 7, const int windowSize1 = 7, const int windowSize2 = 7)
{
  constexpr const float K1 = 0.01;
  constexpr const float K2 = 0.03;
  constexpr const float K1K1 = K1 * K1;
  constexpr const float K2K2 = K2 * K2;

  const size_t index = offset0 + size0 * (offset1 + size1 * offset2);
  float orig_Min = orig_data[index];
  float orig_Max = orig_data[index];
  float orig_Sum = 0;
  float recon_Sum = 0;
  const int np = windowSize0 * windowSize1 * windowSize2;

  for (int i2 = offset2; i2 < offset2 + windowSize2; i2++) {
    for (int i1 = offset1; i1 < offset1 + windowSize1; i1++) {
      for (int i0 = offset0; i0 < offset0 + windowSize0; i0++) {
        const size_t index = i0 + size0 * (i1 + size1 * i2);
        const float orig = orig_data[index];
        const float recon = recon_data[index];
        orig_Min = min(orig_Min, orig);
        orig_Max = max(orig_Max, orig);
        orig_Sum += orig;
        recon_Sum += recon;
      }
    }
  }

  const float orig_Mean = orig_Sum / np;
  const float recon_Mean = recon_Sum / np;
  float var_orig = 0, var_recon = 0, var_orig_recon = 0;

  for (int i2 = offset2; i2 < offset2 + windowSize2; i2++) {
    for (int i1 = offset1; i1 < offset1 + windowSize1; i1++) {
      for (int i0 = offset0; i0 < offset0 + windowSize0; i0++) {
        const size_t index = i0 + size0 * (i1 + size1 * i2);
        const float orig = orig_data[index];
        const float recon = recon_data[index];
        const float orig_diff = orig - orig_Mean;
        const float recon_diff = recon - recon_Mean;
        var_orig += orig_diff * orig_diff;
        var_recon += recon_diff * recon_diff;
        var_orig_recon += orig_diff * recon_diff;
      }
    }
  }

  var_orig /= np;
  var_recon /= np;
  var_orig_recon /= np;

  const float orig_Sigma = sqrtf(var_orig);
  const float recon_Sigma = sqrtf(var_recon);
  const float orig_recon_Cov = var_orig_recon;

  const float orig_range = orig_Max - orig_Min;
  const float orig_range2 = orig_range * orig_range;
  const float c1 = max(K1K1 * orig_range2, 2 * cuda::std::numeric_limits<float>::min());
  const float c2 = max(K2K2 * orig_range2, 2 * cuda::std::numeric_limits<float>::min());
  const float c3 = c2 / 2;

  const float or_re_Sigma = orig_Sigma * recon_Sigma;
  const float luminance = (2 * orig_Mean * recon_Mean + c1) / (orig_Mean * orig_Mean + recon_Mean * recon_Mean + c1);
  const float contrast = (2 * or_re_Sigma + c2) / (var_orig + var_recon + c2);
  const float structure = (orig_recon_Cov + c3) / (or_re_Sigma + c3);
  const float ssim = luminance * contrast * structure;

  return ssim;
}

#endif  /* LC_DSSIM_H_ */
