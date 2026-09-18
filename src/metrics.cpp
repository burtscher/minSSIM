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


#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <map>
#include <vector>


static double SSIM_3d_calcWindow_float(const float* const orig_data, const float* const recon_data, const size_t size1, const size_t size0, const int offset0, const int offset1, const int offset2, const int windowSize0 = 7, const int windowSize1 = 7, const int windowSize2 = 7)
{
  const float K1 = 0.01;
  const float K2 = 0.03;

  const size_t index = offset0 + size0 * (offset1 + size1 * offset2);
  float orig_Min = orig_data[index];
  float orig_Max = orig_data[index];
  double orig_Sum = 0.0;
  double recon_Sum = 0.0;
  int np = 0;
  for (int i2 = offset2; i2 < offset2 + windowSize2; i2++) {
    for (int i1 = offset1; i1 < offset1 + windowSize1; i1++) {
      for (int i0 = offset0; i0 < offset0 + windowSize0; i0++) {
        np++;
        const size_t index = i0 + size0 * (i1 + size1 * i2);
        const float orig = orig_data[index];
        const float recon = recon_data[index];
        orig_Min = std::min(orig_Min, orig);
        orig_Max = std::max(orig_Max, orig);
        orig_Sum += orig;
        recon_Sum += recon;
      }
    }
  }

  const double orig_Mean = orig_Sum / np;
  const double recon_Mean = recon_Sum / np;
  double var_orig = 0, var_recon = 0, var_orig_recon = 0;

  for (int i2 = offset2; i2 < offset2 + windowSize2; i2++) {
    for (int i1 = offset1; i1 < offset1 + windowSize1; i1++) {
      for (int i0 = offset0; i0 < offset0 + windowSize0; i0++) {
        const size_t index = i0 + size0 * (i1 + size1 * i2);
        const float orig = orig_data[index];
        const float recon = recon_data[index];
        var_orig += (orig - orig_Mean) * (orig - orig_Mean);
        var_recon += (recon - recon_Mean) * (recon - recon_Mean);
        var_orig_recon += (orig - orig_Mean) * (recon - recon_Mean);
      }
    }
  }

  var_orig /= np;
  var_recon /= np;
  var_orig_recon /= np;

  const double orig_Sigma = std::sqrt(var_orig);
  const double recon_Sigma = std::sqrt(var_recon);
  const double orig_recon_Cov = var_orig_recon;

  const double c1 = std::max(K1 * K1 * (orig_Max - orig_Min) * (orig_Max - orig_Min), 2 * std::numeric_limits<float>::min());
  const double c2 = std::max(K2 * K2 * (orig_Max - orig_Min) * (orig_Max - orig_Min), 2 * std::numeric_limits<float>::min());
  const double c3 = c2 / 2;

  const double luminance = (2 * orig_Mean * recon_Mean + c1) / (orig_Mean * orig_Mean + recon_Mean * recon_Mean + c1);
  const double contrast = (2 * orig_Sigma * recon_Sigma + c2) / (var_orig + var_recon + c2);
  const double structure = (orig_recon_Cov + c3) / (orig_Sigma * recon_Sigma + c3);
  const double ssim = luminance * contrast * structure;

  return ssim;
}


static double SSIM_2d_calcWindow_float(const float* const orig_data, const float* const recon_data, const size_t size0, const int offset0, const int offset1, const int windowSize0 = 7, const int windowSize1 = 7)
{
  const float K1 = 0.01;
  const float K2 = 0.03;

  const size_t index = offset0 + size0 * offset1;
  float orig_Min = orig_data[index];
  float orig_Max = orig_data[index];
  double orig_Sum = 0.0;
  double recon_Sum = 0.0;
  int np = 0;
  for (int i1 = offset1; i1 < offset1 + windowSize1; i1++) {
    for (int i0 = offset0; i0 < offset0 + windowSize0; i0++) {
      np++;
      const size_t index = i0 + size0 * i1;
      const float orig = orig_data[index];
      const float recon = recon_data[index];
      orig_Min = std::min(orig_Min, orig);
      orig_Max = std::max(orig_Max, orig);
      orig_Sum += orig;
      recon_Sum += recon;
    }
  }

  const double orig_Mean = orig_Sum / np;
  const double recon_Mean = recon_Sum / np;
  double var_orig = 0, var_recon = 0, var_orig_recon = 0;

  for (int i1 = offset1; i1 < offset1 + windowSize1; i1++) {
    for (int i0 = offset0; i0 < offset0 + windowSize0; i0++) {
      const size_t index = i0 + size0 * i1;
      const float orig = orig_data[index];
      const float recon = recon_data[index];
      var_orig += (orig - orig_Mean) * (orig - orig_Mean);
      var_recon += (recon - recon_Mean) * (recon - recon_Mean);
      var_orig_recon += (orig - orig_Mean) * (recon - recon_Mean);
    }
  }

  var_orig /= np;
  var_recon /= np;
  var_orig_recon /= np;

  const double orig_Sigma = std::sqrt(var_orig);
  const double recon_Sigma = std::sqrt(var_recon);
  const double orig_recon_Cov = var_orig_recon;

  const double c1 = std::max(K1 * K1 * (orig_Max - orig_Min) * (orig_Max - orig_Min), 2 * std::numeric_limits<float>::min());
  const double c2 = std::max(K2 * K2 * (orig_Max - orig_Min) * (orig_Max - orig_Min), 2 * std::numeric_limits<float>::min());
  const double c3 = c2 / 2;

  const double luminance = (2 * orig_Mean * recon_Mean + c1) / (orig_Mean * orig_Mean + recon_Mean * recon_Mean + c1);
  const double contrast = (2 * orig_Sigma * recon_Sigma + c2) / (var_orig + var_recon + c2);
  const double structure = (orig_recon_Cov + c3) / (orig_Sigma * recon_Sigma + c3);
  const double ssim = luminance * contrast * structure;

  return ssim;
}


//                                                                                                         d2                  d1
static void SSIM_2d_windowed_float(const float* const oriData, const float* const decData, const size_t size1, const size_t size0)
{
  const int windowSize0 = 7;
  const int windowSize1 = 7;
  const int windowShift0 = 2;
  const int windowShift1 = 2;
  if (windowSize0 > size0) {
    printf("ERROR: windowSize0 = %d > %zu\n", windowSize0, size0);
  }
  if (windowSize1 > size1) {
    printf("ERROR: windowSize1 = %d > %zu\n", windowSize1, size1);
  }

  double ssimSum = 0.0;
  double minSSIM = 1.0;
  int nw = 0;  // number of windows
  #pragma omp parallel for default(none) reduction(+: ssimSum, nw) reduction(min: minSSIM) shared(windowShift0, windowShift1, windowSize0, windowSize1, size0, size1, oriData, decData)
  for (int offset1 = 0; offset1 <= size1 - windowSize1; offset1 += windowShift1) {
    for (int offset0 = 0; offset0 <= size0 - windowSize0; offset0 += windowShift0) {
      nw++;
      const double ssim = SSIM_2d_calcWindow_float(oriData, decData, size0, offset0, offset1, windowSize0, windowSize1);
      ssimSum += ssim;
      minSSIM = std::min(minSSIM, ssim);
    }
  }

  printf("SSIM: %f\n", (ssimSum / nw));
  printf("minSSIM: %f\n", minSSIM);
}

//                                                                                                         d3                  d2                  d1
static void SSIM_3d_windowed_float(const float* const oriData, const float* const decData, const size_t size2, const size_t size1, const size_t size0)
{
  const int windowSize0 = 7;
  const int windowSize1 = 7;
  const int windowSize2 = 7;
  const int windowShift0 = 2;
  const int windowShift1 = 2;
  const int windowShift2 = 2;
  if (windowSize0 > size0) {
    printf("ERROR: windowSize0 = %d > %zu\n", windowSize0, size0);
  }
  if (windowSize1 > size1) {
    printf("ERROR: windowSize1 = %d > %zu\n", windowSize1, size1);
  }
  if (windowSize2 > size2) {
    printf("ERROR: windowSize2 = %d > %zu\n", windowSize2, size2);
  }

  double ssimSum = 0.0;
  double minSSIM = 1.0;
  int nw = 0;  // number of windows
  #pragma omp parallel for default(none) reduction(+: ssimSum, nw) reduction(min: minSSIM) shared(windowShift0, windowShift1, windowShift2, windowSize0, windowSize1, windowSize2, size0, size1, size2, oriData, decData)  //collapse(3)
  for (int offset2 = 0; offset2 <= size2 - windowSize2; offset2 += windowShift2) {
    for (int offset1 = 0; offset1 <= size1 - windowSize1; offset1 += windowShift1) {
      for (int offset0 = 0; offset0 <= size0 - windowSize0; offset0 += windowShift0) {
        nw++;
        const double ssim = SSIM_3d_calcWindow_float(oriData, decData, size1, size0, offset0, offset1, offset2, windowSize0, windowSize1, windowSize2);
        ssimSum += ssim;
        minSSIM = std::min(minSSIM, ssim);
      }
    }
  }

  printf("SSIM: %f\n", (ssimSum / nw));
  printf("minSSIM: %f\n", minSSIM);
}


static void floatMetrics(const float* original, const float* reconstructed, const size_t size, const int d1, const int d2, const int d3)
{
  assert(size > 0);

  double mse = 0;
  double relerr = 0;
  float maxabserr = 0;
  float maxrelerr = 0;
  float inmin = original[0];
  float inmax = original[0];

  float *diff = new float[size];

  for (size_t i = 0; i < size; i++) {
    const float orig = original[i];
    const float recon = reconstructed[i];
    diff[i] = orig - recon;
    if (orig != 0) {
      const float aratio = std::abs(diff[i] / orig);
      relerr += aratio;
      maxrelerr = std::max(maxrelerr, aratio);
    }
    mse += diff[i] * diff[i];
    maxabserr = std::max(maxabserr, std::abs(diff[i]));
    inmin = std::min(inmin, orig);
    inmax = std::max(inmax, orig);
  }

  relerr /= size;
  mse /= size;
  float in_range = inmax - inmin;
  // Print stats
  printf("avg REL err: %f\n", relerr);
  printf("max REL err: %f\n", maxrelerr);
  printf("avg ABS err: %f\n", std::sqrt(mse));
  printf("max ABS err: %f\n", maxabserr);
  printf("avg NOA err: %f\n", std::sqrt(mse)/in_range);
  printf("max NOA err: %f\n", maxabserr/in_range);
  printf("MSE: %f\n", mse);
  if (mse == 0) {
    printf("PSNR: +inf\n");
  } else if (inmax == inmin) {
    printf("PSNR: -inf\n");
  } else {
    const double psnr = 20 * log10(inmax - inmin) - 10 * log10(mse);
    printf("PSNR: %f\n", psnr);
  }
  if (d3 == -1) {
    SSIM_2d_windowed_float(original, reconstructed, d2, d1);
  } else {
    SSIM_3d_windowed_float(original, reconstructed, d3, d2, d1);
  }

  printf("\n");
  delete [] diff;
}


int main(int argc, char* argv[])
{
  printf("Float Analyzer (%s)\n", __FILE__);

  // perform system checks
  if (sizeof(int) != 4) {
    fprintf(stderr, "ERROR: int must be 4 bytes\n\n");
    exit(-1);
  }
  if (sizeof(float) != 4) {
    fprintf(stderr, "ERROR: float must be 4 bytes\n\n");
    exit(-1);
  }

  // print usage message if needed
  if (argc > 6 || argc < 5) {
    printf("USAGE: %s input_file_name reconstructed_file_name d1 d2 [d3] (in SZ3 order)\n", argv[0]);
    exit(-1);
  }
  bool is_3d = (argc == 6);

  // read input file
  printf("input: %s\n", argv[1]);
  FILE* const fin = fopen(argv[1], "rb");
  assert(fin != NULL);
  fseek(fin, 0, SEEK_END);
  const size_t fsize = ftell(fin);
  assert(fsize > 0);
  assert((fsize % sizeof(float)) == 0);
  const size_t len = fsize / sizeof(float);
  float* const input = new float[len];
  fseek(fin, 0, SEEK_SET);
  const size_t insize = fread(input, 1, fsize, fin);
  assert(insize == fsize);
  fclose(fin);
  printf("input size: %ld bytes (%ld floats)\n", insize, insize / sizeof(float));

  // read reconstructed file always allocate to keep it const, only fill it if we're supplied a file
  float* const recon = new float[len];
  printf("recon: %s\n", argv[2]);
  FILE* const fre = fopen(argv[2], "rb");
  assert(fre != NULL);
  fseek(fre, 0, SEEK_END);
  assert(fsize == ftell(fre));
  fseek(fre, 0, SEEK_SET);
  const size_t resize = fread(recon, 1, fsize, fre);
  assert(resize == fsize);
  fclose(fre);

  int d1 = atoi(argv[3]);
  int d2 = atoi(argv[4]);
  int d3 = is_3d ? atoi(argv[5]) : -1;

  floatMetrics(input, recon, len, d1, d2, d3);

  // clean up
  delete [] input;
  delete [] recon;
  return 0;
}
