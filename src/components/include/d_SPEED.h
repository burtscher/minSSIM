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


#ifndef GPU_SPEED
#define GPU_SPEED


#include <cuda/atomic>


template <typename T>
static __device__ inline bool d_SPEED(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int TB = sizeof(T) * 8;  // number of bits in T
  const int size = CS / sizeof(T);
  const int SC = 32;  // subchunks [do not change]
  const int chunksize = size / SC;

  static_assert(sizeof(T) >= 4);
  static_assert(chunksize % WS == 0);
  static_assert(WS == SC);
  static_assert(SC == sizeof(int) * 8);
  static_assert(std::is_unsigned<T>::value);

  const int tid = threadIdx.x;
  const int lane = tid % WS;
  const int warp = tid / WS;
  const int warps = TPB / WS;

  // clear out unused part of input buffer
  if (csize < CS) {
    for (int i = csize + tid; i < CS; i += TPB) {
      in[i] = 0;
    }
    __syncthreads();
  }

  // type casts
  T* const in_t = (T*)in;
  T* const tmp_t = (T*)temp;
  T* const out_t = (T*)&out[SC];
  if (tid < SC) out_t[tid] = in_t[tid * chunksize];  // copy first value verbatim

  // determine bits needed for each subchunk
  int flag = 0;
  int ln = -1;
  for (int i = warp; i < SC; i += warps) {
    // compute maximum value using both approaches
    const int beg = i * chunksize;
    T max_val1 = 0;
    T max_val2 = 0;

    // max of values for each thread
    T prev = 0;
    for (int j = lane; j < chunksize; j += WS) {
      // compute difference sequence plus TCMS
      const T curr = in_t[beg + j];
      T diff = curr - __shfl(((lane == WS - 1) ? prev : curr), lane - 1);
      if (j == 0) diff = 0;
      prev = curr;
      const T val1 = (diff << 1) ^ ((std::make_signed_t<T>)diff) >> (TB - 1);  // TCMS
      tmp_t[beg + j] = val1;
      max_val1 = max(max_val1, val1);
      const T val2 = (val1 << 1) ^ (((std::make_signed_t<T>)val1) >> (TB - 1));  // TCMS
      max_val2 = max(max_val2, val2);
    }

    // warp-level max
    max_val1 = max(max_val1, __shfl_xor(max_val1, 1));
    max_val2 = max(max_val2, __shfl_xor(max_val2, 1));
    max_val1 = max(max_val1, __shfl_xor(max_val1, 2));
    max_val2 = max(max_val2, __shfl_xor(max_val2, 2));
    max_val1 = max(max_val1, __shfl_xor(max_val1, 4));
    max_val2 = max(max_val2, __shfl_xor(max_val2, 4));
    max_val1 = max(max_val1, __shfl_xor(max_val1, 8));
    max_val2 = max(max_val2, __shfl_xor(max_val2, 8));
    max_val1 = max(max_val1, __shfl_xor(max_val1, 16));
    max_val2 = max(max_val2, __shfl_xor(max_val2, 16));

    // figure out number of bits needed
    if (lane == i) {
      int cnt1 = TB;
      if (max_val1 != 0) {
        if constexpr (sizeof(T) == 8) {
          cnt1 = __clzll(max_val1);
        } else {
          cnt1 = __clz(max_val1);
        }
      }
      int cnt2 = TB;
      if (max_val2 != 0) {
        if constexpr (sizeof(T) == 8) {
          cnt2 = __clzll(max_val2);
        } else {
          cnt2 = __clz(max_val2);
        }
      }

      // use approach requiring fewer bits
      const int cnt = max(cnt1, cnt2);
      flag = (cnt2 > cnt1);
      ln = TB - cnt;  // store logn value
    }
  }
  flag = __ballot(flag);
  if (ln >= 0) out[lane] = ln;
  __syncthreads();

  int* const bits = (int*)&in[sizeof(T)];
  int* const total_bits = &bits[WS];
  int* const flags = &total_bits[1];

  // warp prefix sum over bits
  if (warp == 0) {
    const int org = out[lane] * chunksize;
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
    if (lane == SC - 1) {
      *total_bits = val;
      *flags = 0;  // clear flags
    }
  }
  __syncthreads();

  // check if encoded data fits
  const int newsize = (SC * 8 + SC * TB + SC + 16 + *total_bits) / 8;
  if (newsize >= CS) return false;

  // encode data values
  for (int i = warp; i < SC; i += warps) {
    const int logn = __shfl(ln, i);
    if (logn > 0) {
      const int beg = i * chunksize;
      const int end = beg + chunksize;
      if (logn == TB) {
        const int offs = SC - beg + bits[i] / TB;
        cuda::atomic<T, cuda::thread_scope_block>* const ptr = (cuda::atomic<T, cuda::thread_scope_block>*)&out_t[offs];
        for (int j = beg + lane; j < end; j += WS) {
          ptr[j].store(tmp_t[j], cuda::memory_order_relaxed);
        }
      } else {
        const int offs = SC * TB + bits[i];
        const int words = (WS * logn + (TB - 1)) / TB;  // round up if WS == 32 and TB == 64 and logn == odd
        int wpos = offs / TB;
        if ((sizeof(T) == 8) && (WS == 32) && (logn & 1)) wpos *= 2;

        int width1 = logn;
        int dist1 = 1;
        while (width1 <= TB / 2) {
          width1 *= 2;
          dist1 *= 2;
        }
        const int pos1 = lane * TB;
        const int src1 = (pos1 / width1) * dist1;
        const int drop1 = pos1 % width1;
        const int bits1 = width1 - drop1;
        const bool cond = (flag & (1 << i));

        for (int j = beg + lane; j < end; j += WS) {
          T val = tmp_t[j];
          if (cond) {
            val = (val << 1) ^ (((std::make_signed_t<T>)val) >> (sizeof(T) * 8 - 1));  // TCMS
          }

          int dist = 1;
          int width = logn;
          while (width <= TB / 2) {
            const T recv = __shfl_xor(val, dist);
            dist *= 2;
            if (lane % dist == 0) {
              val |= recv << width;
            }
            width *= 2;
          }

          T res = __shfl(val, src1);
          int src = src1;
          res >>= drop1;
          int bits = bits1;
          while (__any_sync(~0, (bits < TB) && (lane < words))) {
            src += dist1;
            const T recv = __shfl(val, src);
            if (bits < TB) {
              res |= recv << bits;
            }
            bits += width1;
          }

          if ((sizeof(T) == 8) && (WS == 32) && (logn & 1)) {
            if (lane < words) {
              cuda::atomic<int, cuda::thread_scope_block>* const ptr = (cuda::atomic<int, cuda::thread_scope_block>*)out_t;
              ptr[wpos + 2 * lane].store((int)res, cuda::memory_order_relaxed);
              if (lane < words - 1) ptr[wpos + 2 * lane + 1].store((int)(res >> (TB / 2)), cuda::memory_order_relaxed);
            }
            wpos += words * 2 - 1;
          } else {
            if (lane < words) {
              //out_t[wpos + lane] = res;
              cuda::atomic<T, cuda::thread_scope_block>* const ptr = (cuda::atomic<T, cuda::thread_scope_block>*)&out_t[wpos + lane];
              ptr->store(res, cuda::memory_order_relaxed);
            }
            wpos += words;
          }
        }
      }
    }
  }
  if (lane == 0) atomicOr_block(flags, flag);  // set flags
  __syncthreads();

  // output header info
  if (tid == 0) {
    *(int*)&out[newsize - 6] = *flags;
    *(short*)&out[newsize - 2] = csize;
  }

  csize = newsize;
  return true;
}


template <typename T>
static __device__ inline void d_iSPEED(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int tid = threadIdx.x;
  const int lane = tid % WS;
  const int warp = tid / WS;
  const int warps = TPB / WS;

  const int TB = sizeof(T) * 8;  // number of bits in T
  const int size = CS / sizeof(T);
  const int SC = 32;  // subchunks [do not change]
  const int chunksize = size / SC;

  static_assert(sizeof(T) >= 4);
  static_assert(WS == SC);
  static_assert(SC == sizeof(int) * 8);
  static_assert(std::is_unsigned<T>::value);

  // type casts
  const T* const in_t = (T*)&in[SC];
  T* const tmp_t = (T*)temp;
  T* const out_t = (T*)out;
  int* const bits = (int*)out;

  // read header info
  const int orig_csize = *(short*)&in[csize - 2];
  const int flags = *(int*)&in[csize - 6];

   // warp prefix sum over bits
  if (warp == 0) {
    const int org = (lane < SC) ? (in[lane] * chunksize) : 0;
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
  __syncthreads();

  // decode data values
  for (int i = warp; i < SC; i += warps) {
    const int logn = in[i];
    const int beg = i * chunksize;
    const int end = beg + chunksize;
    if (logn == 0) {
      for (int j = beg + lane; j < end; j += WS) {
        tmp_t[j] = 0;
      }
    } else if (logn == TB) {
      const int offs = (SC * TB + bits[i]) / TB - beg;
      for (int j = beg + lane; j < end; j += WS) {
        T val = in_t[offs + j];
        val = (val >> 1) ^ ((std::make_signed_t<T>)(val << (sizeof(T) * 8 - 1))) >> (sizeof(T) * 8 - 1);  // iTCMS
        tmp_t[j] = val;
      }
    } else {
      const int offs = SC * TB + bits[i];
      const bool flag = flags & (1 << i);
      const T mask = ((T)1 << logn) - 1;
      for (int j = beg + lane; j < end; j += WS) {
        const int loc = offs + (j - beg) * logn;
        const int pos = loc / TB;
        const int shift = loc % TB;
        T res = in_t[pos] >> shift;
        if (TB - logn < shift) {
          res |= in_t[pos + 1] << (TB - shift);
        }
        T val = res & mask;
        if (flag) {
          val = (val >> 1) ^ ((std::make_signed_t<T>)(val << (sizeof(T) * 8 - 1))) >> (sizeof(T) * 8 - 1);  // iTCMS
        }
        val = (val >> 1) ^ ((std::make_signed_t<T>)(val << (sizeof(T) * 8 - 1))) >> (sizeof(T) * 8 - 1);  // iTCMS
        tmp_t[j] = val;
      }
    }
  }
  __syncthreads();

  if (tid < SC) {
    tmp_t[tid * chunksize] = in_t[tid];  // copy first value verbatim
  }
  __syncthreads();

  for (int i = warp; i < SC; i += warps) {
    const int wbeg = i * chunksize;
    const int beg = wbeg + lane * chunksize / WS;
    const int end = wbeg + (lane + 1) * chunksize / WS;

    // compute local sums
    T sum = 0;
    for (int i = beg; i < end; i++) {
      sum += tmp_t[i];
    }

    // compute prefix sum
    T tmp = __shfl_up(sum, 1);
    if (lane >= 1) sum += tmp;
    tmp = __shfl_up(sum, 2);
    if (lane >= 2) sum += tmp;
    tmp = __shfl_up(sum, 4);
    if (lane >= 4) sum += tmp;
    tmp = __shfl_up(sum, 8);
    if (lane >= 8) sum += tmp;
    tmp = __shfl_up(sum, 16);
    if (lane >= 16) sum += tmp;

    // compute intermediate values
    for (int i = end - 1; i >= beg; i--) {
      out_t[i] = sum;
      sum -= tmp_t[i];
    }
  }

  csize = orig_csize;
}


#endif
