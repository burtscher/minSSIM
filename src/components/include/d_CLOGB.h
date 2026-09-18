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


#ifndef GPU_CLOGB
#define GPU_CLOGB


template <typename T>
static __device__ inline bool d_CLOGB(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int tid = threadIdx.x;
  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;
  const int constexpr warps = TPB / WS;

  static_assert(std::is_unsigned<T>::value);
  const int constexpr TB = sizeof(T) * 8;  // number of bits in T
  const int constexpr SC = 32;  // subchunks [do not change]
  const int constexpr CB = (sizeof(T) >= 4) ? ((sizeof(T) == 8) ? 7 : 6) : ((sizeof(T) == 2) ? 5 : 4);  // counter bits
  static_assert((1 << CB) > TB);
  static_assert((1 << (CB - 1)) <= TB);
  static_assert(WS >= SC);
  static_assert(SC == sizeof(int) * 8);
  static_assert(sizeof(int) == 4);

  // T casts
  T* const in_t = (T*)in;
  int* const out_i = (int*)out;  // int
  const int constexpr TB_i = sizeof(int) * 8;

  byte* ln = temp;
  byte* const tr = &temp[SC * 2];
  int* total_bits = (int*)&tr[SC];
  int* const saved = &total_bits[2];
  int* bits = &saved[1];

  // determine bits needed for each subchunk
  const int size = csize / sizeof(T);
  for (int i = warp; i < SC; i += warps) {
    const int beg = i * size / SC;
    const int end = (i + 1) * size / SC;

    T or_val = 0;
    // max of values for each thread
    for (int j = beg + lane; j < end; j += WS) {
      or_val |= in_t[j];
    }

    // warp level or
    or_val |= __shfl_xor(or_val, 1);
    or_val |= __shfl_xor(or_val, 2);
    or_val |= __shfl_xor(or_val, 4);
    or_val |= __shfl_xor(or_val, 8);
    or_val |= __shfl_xor(or_val, 16);
    #if defined(WS) && (WS == 64)
    or_val |= __shfl_xor(or_val, 32);
    #endif

    if (lane == 0) {
      // figure out number of bits needed
      int cnt = 0;
      int trunc = 0;
      if (or_val != 0) {
        trunc = (sizeof(T) == 8) ? (__builtin_ffsll((unsigned long long)or_val) - 1) : (__builtin_ffs((unsigned int)or_val) - 1);
        or_val >>= trunc;
        cnt = (sizeof(T) == 8) ? (sizeof(unsigned long long) * 8 - __clzll((unsigned long long)or_val)) : (sizeof(unsigned int) * 8 - __clz((unsigned int)or_val));
      }
      ln[i] = cnt;  // logn value for each subchunk
      tr[i] = trunc;  // truncation value for each subchunk
      ln[i + SC] = cnt + trunc;  // logn value for each subchunk
    }
  }
  __syncthreads();

  // warp prefix sum over bits
  if (warp == 0) {
    const int beg = lane * size / SC;
    const int end = (lane + 1) * size / SC;
    const int org = ln[lane] * (end - beg);
    int val = org;
    int tmp = __shfl_up(val, 1);
    if (lane >= 1) val += tmp;
    tmp = __shfl_up(val, 2);
    if (lane >= 2) val += tmp;
    tmp = __shfl_up(val, 4);
    if (lane >= 4) val += tmp;
    tmp = __shfl_up(val, 8);
    if (lane >= 8) val += tmp;
    tmp = __shfl_up(val, 16);
    if (lane >= 16) val += tmp;
    bits[lane] = val - org;
    if (lane == SC - 1) total_bits[0] = val;
  }
  if (warp == 1) {
    const int beg = lane * size / SC;
    const int end = (lane + 1) * size / SC;
    const int org = ln[lane + SC] * (end - beg);
    int val = org;
    int tmp = __shfl_up(val, 1);
    if (lane >= 1) val += tmp;
    tmp = __shfl_up(val, 2);
    if (lane >= 2) val += tmp;
    tmp = __shfl_up(val, 4);
    if (lane >= 4) val += tmp;
    tmp = __shfl_up(val, 8);
    if (lane >= 8) val += tmp;
    tmp = __shfl_up(val, 16);
    if (lane >= 16) val += tmp;
    bits[lane + SC] = val - org;
    if (lane == SC - 1) total_bits[1] = val;
  }

  // warp prefix sum over saved bits
  if (warp == 2) {
    const int beg = lane * size / SC;
    const int end = (lane + 1) * size / SC;
    int val = tr[lane] * (end - beg);
    int tmp = __shfl_up(val, 1);
    val += tmp;
    tmp = __shfl_up(val, 2);
    val += tmp;
    tmp = __shfl_up(val, 4);
    val += tmp;
    tmp = __shfl_up(val, 8);
    val += tmp;
    tmp = __shfl_up(val, 16);
    val += tmp;
    if (lane == SC - 1) *saved = val;
  }
  __syncthreads();

  // check if encoded data fits
  const int sav = *saved;
  const int flag = (sav > CB * SC) ? 0x8000 : 0;
  const bool cond = (sav > 0) && !flag;
  if (cond) {
    total_bits++;
    bits += SC;
    ln += SC;
  }

  int newsize = (16 + CB * SC + *total_bits + 7) / 8;
  if (flag) newsize += CB * SC / 8;
  const int extra = csize % sizeof(T);
  if (newsize + extra >= CS) return false;

  // clear out buffer
  for (int i = tid; i < (newsize + sizeof(int) - 1) / sizeof(int); i += TPB) out_i[i] = 0;
  __syncthreads();

  // encode logn values
  if (tid < SC) {
    const int val = ln[lane];
    const int loc = 16 + (CB * lane);
    const int pos = loc / TB_i;
    const int shift = loc % TB_i;
    atomicOr_block(&out_i[pos], val << shift);
    if (TB_i - CB < shift) {
      atomicOr_block(&out_i[pos + 1], val >> (TB_i - shift));
    }
  }

  // encode data values
  for (int i = warp; i < SC; i += warps) {
    const int logn = ln[i];
    if (logn > 0) {
      const int trunc = (flag ? tr[i] : 0);
      const int beg = i * size / SC;
      const int end = (i + 1) * size / SC;
      const int offs = 16 + CB * SC + bits[i];
      for (int j = beg + lane; j < end; j += WS) {
        const T val = in_t[j] >> trunc;
        const int loc = offs + (j - beg) * logn;
        if constexpr (sizeof(T) < 8) {
          const int pos = loc / TB_i;
          const int shift = loc % TB_i;
          atomicOr_block(&out_i[pos], (unsigned int)val << shift);
          if (TB_i - logn < shift) {
            atomicOr_block(&out_i[pos + 1], (unsigned int)val >> (TB_i - shift));
          }
        } else {
          long long* const out_l = (long long*)out;
          const int pos = loc / TB;
          const int shift = loc % TB;
          atomicOr_block((unsigned long long *)&out_l[pos], val << shift);
          if (TB - logn < shift) {
            atomicOr_block((unsigned long long *)&out_l[pos + 1], val >> (TB - shift));
          }
        }
      }
    }
  }

  // encode trunc values
  if (flag && (tid < SC)) {
    const int val = tr[lane];
    const int loc = newsize * 8 - CB * SC + CB * lane;
    const int pos = loc / TB_i;
    const int shift = loc % TB_i;
    atomicOr_block(&out_i[pos], val << shift);
    if (TB_i - CB < shift) {
      atomicOr_block(&out_i[pos + 1], val >> (TB_i - shift));
    }
  }
  __syncthreads();

  // copy leftover bytes
  if constexpr (sizeof(T) > 1) {
    if (tid < extra) out[newsize + tid] = in[csize - extra + tid];
  }

  // record old csize
  if (tid == 0) {
    *((short*)out) = csize | flag;
  }
  csize = newsize + extra;
  return true;
}


template <typename T>
static __device__ inline void d_iCLOGB(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int tid = threadIdx.x;
  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;

  static_assert(std::is_unsigned<T>::value);
  const int constexpr TB = sizeof(T) * 8;  // number of bits in T
  const int constexpr SC = 32;  // subchunks [do not change]
  const int constexpr CB = (sizeof(T) >= 4) ? ((sizeof(T) == 8) ? 7 : 6) : ((sizeof(T) == 2) ? 5 : 4);  // counter bits
  static_assert((1 << CB) > TB);
  static_assert((1 << (CB - 1)) <= TB);
  static_assert(WS >= SC);

  // type casts
  T* const in_t = (T*)in;
  T* const out_t = (T*)out;
  byte* const ln = (byte*)temp;
  byte* const tr = &ln[SC];
  int* const bits = (int*)&tr[SC];

  // decode csize
  const int head = *((unsigned short*)in);
  const int orig_csize = head & 0x7fff;
  const bool flag = (head >= 0x8000);
  const int size = orig_csize / sizeof(T);

  // decode logn values
  const T constexpr mask = ((1 << CB) - 1);
  if (warp == 0) {
    T res = 0;
    if (lane < SC) {
      const int loc = 16 + (lane * CB);
      const int pos = loc / TB;
      const int shift = loc % TB;
      res = in_t[pos] >> shift;
      if (TB - CB < shift) {
        res |= in_t[pos + 1] << (TB - shift);
      }
      res &= mask;
      ln[lane] = res;
    }

    const int beg = lane * size / SC;
    const int end = (lane + 1) * size / SC;
    const int org = res * (end - beg);
    int val = org;
    int tmp = __shfl_up(val, 1);
    if (lane >= 1) val += tmp;
    tmp = __shfl_up(val, 2);
    if (lane >= 2) val += tmp;
    tmp = __shfl_up(val, 4);
    if (lane >= 4) val += tmp;
    tmp = __shfl_up(val, 8);
    if (lane >= 8) val += tmp;
    tmp = __shfl_up(val, 16);
    if (lane >= 16) val += tmp;
    bits[lane] = val - org;
  }

  // decode trunc values
  const int extra = orig_csize % sizeof(T);
  if (flag && (warp == 1)) {
    T res = 0;
    if (lane < SC) {
      const int loc = (csize - extra) * 8 - CB * SC + lane * CB;
      const int pos = loc / TB;
      const int shift = loc % TB;
      res = in_t[pos] >> shift;
      if (TB - CB < shift) {
        res |= in_t[pos + 1] << (TB - shift);
      }
      res &= mask;
      tr[lane] = res;
    }
  }
  __syncthreads();

  // decode data values
  for (int i = warp; i < SC; i += TPB / WS) {
    const int logn = ln[i];
    const int beg = i * size / SC;
    const int end = (i + 1) * size / SC;
    if (logn > 0) {
      const int trunc = (flag ? tr[i] : 0);
      const T mask = (sizeof(T) < 8) ? ((1ULL << logn) - 1) : ((logn == 64) ? (~0ULL) : ((1ULL << logn) - 1));
      const int offs = 16 + SC * CB + bits[i];
      for (int j = beg + lane; j < end; j += WS) {
        const int loc = offs + (j - beg) * logn;
        const int pos = loc / TB;
        const int shift = loc % TB;
        T res = in_t[pos] >> shift;
        if (TB - logn < shift) {
          res |= in_t[pos + 1] << (TB - shift);
        }
        out_t[j] = (res & mask) << trunc;
      }
    } else {
      for (int j = beg + lane; j < end; j += WS) {
        out_t[j] = 0;
      }
    }
  }

  // copy leftover bytes
  if constexpr (sizeof(T) > 1) {
    if (tid < extra) out[orig_csize - extra + tid] = in[csize - extra + tid];
  }
  csize = orig_csize;
}


#endif
