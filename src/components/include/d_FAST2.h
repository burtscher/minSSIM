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


#ifndef GPU_FAST2
#define GPU_FAST2


#include <cuda/atomic>


template <typename T>
static __device__ inline bool d_FAST2(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int tid = threadIdx.x;
  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;
  const int warps = TPB / WS;

  const int TB = sizeof(T) * 8;  // number of bits in T
  const int size = CS / sizeof(T);
  const int SC = 32;  // subchunks [do not change]
  const int chunksize = size / SC;

  static_assert(WS >= SC);
  static_assert(SC == sizeof(int) * 8);
  static_assert(std::is_unsigned<T>::value);

  // T casts
  T* const in_t = (T*)in;
  T* const out_t = (T*)&out[SC];

  for (int i = tid; i < size; i += TPB) { //rm
    T* const out_t = (T*)out;
    out_t[i] = 0;
  }
  __syncthreads();

  // clear out unused part of input buffer
  if (csize < CS) {
    for (int i = csize + tid; i < CS; i += TPB) {
      in[i] = 0;
    }
    __syncthreads();
  }

  int* const bits = (int*)temp;
  int* const total_bits = &bits[WS];
  int* const flags = &total_bits[1];

  // copy first value and set it to zero
  if (tid == 0) {
    out_t[0] = in_t[0];
    in_t[0] = 0;
    *flags = 0;  // clear flags
  }
  __syncthreads();

  // determine bits needed for each subchunk
  for (int i = warp; i < SC; i += warps) {
    // compute maximum value using both approaches
    const int beg = i * chunksize;
    const int end = beg + chunksize;
    T max_val1 = 0;
    T max_val2 = 0;
    // max of values for each thread
    for (int j = beg + lane; j < end; j += WS) {
      const T val1 = in_t[j];
      max_val1 = max(max_val1, val1);
      const T val2 = (val1 << 1) ^ (((std::make_signed_t<T>)val1) >> (sizeof(T) * 8 - 1));  // TCMS
      max_val2 = max(max_val2, val2);
    }

    // warp level max
    max_val1 = max(max_val1, __shfl_xor(max_val1, 1));
    max_val1 = max(max_val1, __shfl_xor(max_val1, 2));
    max_val1 = max(max_val1, __shfl_xor(max_val1, 4));
    max_val1 = max(max_val1, __shfl_xor(max_val1, 8));
    max_val1 = max(max_val1, __shfl_xor(max_val1, 16));
#if defined(WS) && (WS == 64)
    max_val1 = max(max_val1, __shfl_xor(max_val1, 32));
#endif
    max_val2 = max(max_val2, __shfl_xor(max_val2, 1));
    max_val2 = max(max_val2, __shfl_xor(max_val2, 2));
    max_val2 = max(max_val2, __shfl_xor(max_val2, 4));
    max_val2 = max(max_val2, __shfl_xor(max_val2, 8));
    max_val2 = max(max_val2, __shfl_xor(max_val2, 16));
#if defined(WS) && (WS == 64)
    max_val2 = max(max_val2, __shfl_xor(max_val2, 32));
#endif

    // figure out number of bits needed
    if (lane == 0) {
      int cnt1 = 0;
      if (max_val1 != 0) {
        cnt1 = (sizeof(T) == 8) ? (sizeof(unsigned long long) * 8 - __clzll((unsigned long long)max_val1)) : (sizeof(unsigned int) * 8 - __clz((unsigned int)max_val1));
      }
      int cnt2 = 0;
      if (max_val2 != 0) {
        cnt2 = (sizeof(T) == 8) ? (sizeof(unsigned long long) * 8 - __clzll((unsigned long long)max_val2)) : (sizeof(unsigned int) * 8 - __clz((unsigned int)max_val2));
      }

      // use approach requiring fewer bits
      const int cnt = min(cnt1, cnt2);
      if (cnt2 < cnt1) {
        atomicOr_block(flags, 1 << i);  // set flag
      }
      out[i] = cnt;  // store logn value
    }
  }
  __syncthreads();

  // warp prefix sum over bits
  if (warp == 0) {
    const int org = (lane < SC) ? (out[lane] * chunksize) : 0;
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
    if (lane == SC - 1) *total_bits = val;
  }
  __syncthreads();

  // check if encoded data fits
  const int newsize = (TB + SC + 16 + 8 * SC + *total_bits) / 8;
  if (newsize >= CS) return false;

  // encode data values
  for (int i = warp; i < SC; i += warps) {
    const int logn = out[i];
    if (logn > 0) {
      const int beg = i * chunksize;
      const int end = beg + chunksize;
      if (logn == TB) {
        const int offs = 1 - beg + bits[i] / TB;
        cuda::atomic<T>* const ptr = (cuda::atomic<T>*)&out_t[offs];
        for (int j = beg + lane; j < end; j += WS) {
          ptr[j].store(in_t[j], cuda::memory_order_relaxed);
        }
      } else {
        const int offs = TB + bits[i];
        const bool flag = *flags & (1 << i);
        for (int j = beg + lane; j < end; j += WS) {
          T val = in_t[j];
          if (flag) {
            val = (val << 1) ^ (((std::make_signed_t<T>)val) >> (sizeof(T) * 8 - 1));  // TCMS
          }
          const int loc = offs + (j - beg) * logn;
          const int pos = loc / TB;
          const int shift = loc % TB;
          atomicOr_block(&out_t[pos], val << shift);
          if (TB - logn < shift) {
            atomicOr_block(&out_t[pos + 1], val >> (TB - shift));
          }
        }
      }
    }
  }

  // output header info
  if (tid < 4) {
    const int shift = tid * 8;
    out[newsize - 6 + tid] = *flags >> shift;
    if (tid < 2) {
      out[newsize - 2 + tid] = csize >> shift;
    }
  }

  csize = newsize;
  return true;
}


template <typename T>
static __device__ inline bool d_FAST2works(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int tid = threadIdx.x;
  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;
  const int warps = TPB / WS;

  const int TB = sizeof(T) * 8;  // number of bits in T
  const int size = CS / sizeof(T);
  const int SC = 32;  // subchunks [do not change]
  const int chunksize = size / SC;

  static_assert(WS >= SC);
  static_assert(SC == sizeof(int) * 8);
  static_assert(std::is_unsigned<T>::value);

  // T casts
  T* const in_t = (T*)in;
  T* const out_t = (T*)&out[SC];

  for (int i = tid; i < size; i += TPB) {
    T* const out_t = (T*)out;
    out_t[i] = 0;
  }
  __syncthreads();

  // clear out unused part of input buffer
  if (csize < CS) {
    for (int i = csize + tid; i < CS; i += TPB) {
      in[i] = 0;
    }
    __syncthreads();
  }

  int* const bits = (int*)&temp[4];
  int* const total_bits = &bits[WS];
  int* const flags = &total_bits[1];

  // copy first value and set it to zero
  if (tid == 0) {
    out_t[0] = in_t[0];
    in_t[0] = 0;
    *flags = 0;  // clear flags
  }
  __syncthreads();

  // determine bits needed for each subchunk
  for (int i = warp; i < SC; i += warps) {
    // compute maximum value using both approaches
    const int beg = i * chunksize;
    const int end = beg + chunksize;
    T max_val1 = 0;
    T max_val2 = 0;
    // max of values for each thread
    for (int j = beg + lane; j < end; j += WS) {
      const T val1 = in_t[j];
      max_val1 = max(max_val1, val1);
      const T val2 = (val1 << 1) ^ (((std::make_signed_t<T>)val1) >> (sizeof(T) * 8 - 1));  // TCMS
      max_val2 = max(max_val2, val2);
    }

    // warp level max
    max_val1 = max(max_val1, __shfl_xor(max_val1, 1));
    max_val1 = max(max_val1, __shfl_xor(max_val1, 2));
    max_val1 = max(max_val1, __shfl_xor(max_val1, 4));
    max_val1 = max(max_val1, __shfl_xor(max_val1, 8));
    max_val1 = max(max_val1, __shfl_xor(max_val1, 16));
#if defined(WS) && (WS == 64)
    max_val1 = max(max_val1, __shfl_xor(max_val1, 32));
#endif
    max_val2 = max(max_val2, __shfl_xor(max_val2, 1));
    max_val2 = max(max_val2, __shfl_xor(max_val2, 2));
    max_val2 = max(max_val2, __shfl_xor(max_val2, 4));
    max_val2 = max(max_val2, __shfl_xor(max_val2, 8));
    max_val2 = max(max_val2, __shfl_xor(max_val2, 16));
#if defined(WS) && (WS == 64)
    max_val2 = max(max_val2, __shfl_xor(max_val2, 32));
#endif

    // figure out number of bits needed
    if (lane == 0) {
      int cnt1 = 0;
      if (max_val1 != 0) {
        cnt1 = (sizeof(T) == 8) ? (sizeof(unsigned long long) * 8 - __clzll((unsigned long long)max_val1)) : (sizeof(unsigned int) * 8 - __clz((unsigned int)max_val1));
      }
      int cnt2 = 0;
      if (max_val2 != 0) {
        cnt2 = (sizeof(T) == 8) ? (sizeof(unsigned long long) * 8 - __clzll((unsigned long long)max_val2)) : (sizeof(unsigned int) * 8 - __clz((unsigned int)max_val2));
      }

      // use approach requiring fewer bits
      const int cnt = min(cnt1, cnt2);
      if (cnt2 < cnt1) {
        atomicOr_block(flags, 1 << i);  // set flag
      }
      out[i] = cnt;  // store logn value
    }
  }
  __syncthreads();

  // warp prefix sum over bits
  if (warp == 0) {
    const int org = (lane < SC) ? (out[lane] * chunksize) : 0;
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
    if (lane == SC - 1) *total_bits = val;
  }
  __syncthreads();

  // check if encoded data fits
  const int ns = (TB + SC + 16 + 8 * SC + *total_bits) / 8;
  if (ns >= CS) return false;

  // encode data values
  for (int i = warp; i < SC; i += warps) {
    const int logn = out[i];
    if (logn > 0) {
      const int beg = i * chunksize;
      const int end = beg + chunksize;
      if (logn == TB) {
        //const int offs = (TB + bits[i]) / TB - beg;
        //if (lane == 0) printf("F start = %ld (%d)\n", &out_t[offs + beg], i);
        //if (lane == 0) memcpy(&out_t[offs + beg], &in_t[beg], chunksize * sizeof(T));

        //for (int j = beg + lane; j < end; j += WS) {
          //out_t[offs + j] = in_t[j];
        //}
        const int offs = 1 - beg + bits[i] / TB;
        T* const ptr = &out_t[offs];
        for (int j = beg + lane; j < end; j += WS) {
          //assert(j >= 0);
          //ptr[j] = in_t[j];
          (*((cuda::atomic<T>*)&ptr[j])).store(in_t[j], cuda::memory_order_relaxed);
        }
      } else {
        const int offs = TB + bits[i];
        const bool flag = *flags & (1 << i);
        for (int j = beg + lane; j < end; j += WS) {
          T val = in_t[j];
          if (flag) {
            val = (val << 1) ^ (((std::make_signed_t<T>)val) >> (sizeof(T) * 8 - 1));  // TCMS
          }
          const int loc = offs + (j - beg) * logn;
          const int pos = loc / TB;
          const int shift = loc % TB;
          atomicOr_block(&out_t[pos], val << shift);
          if (TB - logn < shift) {
            atomicOr_block(&out_t[pos + 1], val >> (TB - shift));
          }
        }
      }
    }
  }
  //__syncthreads();//??

  // output header info
  if (tid < 4) {
    const int shift = tid * 8;
    out[ns - 6 + tid] = *flags >> shift;
    if (tid < 2) {
      out[ns - 2 + tid] = csize >> shift;
    }
  }

  __syncthreads();//??




  int* const newsize = (int*)temp;

if (tid == 0) {
  static_assert(std::is_unsigned<T>::value);
  const int TB = sizeof(T) * 8;  // number of bits in T
  const int size = CS / sizeof(T);
  const int SC = 32;  // subchunks [do not change]
  const int chunksize = size / SC;
  static_assert(SC == sizeof(int) * 8);

  // T casts
  T* in_t = (T*)in;
  T* out_t = (T*)&out[SC];

  // clear out unused part of input buffer
  if (csize < CS) {
//    memset(&in[csize], 0, CS - csize);
  }

  // copy first value and set it to zero
//  out_t[0] = in_t[0];
//  in_t[0] = 0;

  // determine bits needed for each subchunk
  int flags = 0;
  int bbits = 0;
  for (int i = 0; i < SC; i++) {
    // compute maximum value using both approaches
    const int beg = i * chunksize;
    const int end = beg + chunksize;
    T max_val1 = 0;
    T max_val2 = 0;
    for (int j = beg; j < end; j++) {
      const T val1 = in_t[j];
      max_val1 = max(max_val1, val1);
      const T val2 = (val1 << 1) ^ (((std::make_signed_t<T>)val1) >> (sizeof(T) * 8 - 1));  // TCMS
      max_val2 = max(max_val2, val2);
    }

    // figure out number of bits needed
    int cnt1 = 0;
    if (max_val1 != 0) {
      cnt1 = (sizeof(T) == 8) ? (sizeof(unsigned long long) * 8 - __clzll((unsigned long long)max_val1)) : (sizeof(unsigned int) * 8 - __clz((unsigned int)max_val1));
    }
    int cnt2 = 0;
    if (max_val2 != 0) {
      cnt2 = (sizeof(T) == 8) ? (sizeof(unsigned long long) * 8 - __clzll((unsigned long long)max_val2)) : (sizeof(unsigned int) * 8 - __clz((unsigned int)max_val2));
    }

    // use approach requiring fewer bits
    const int cnt = min(cnt1, cnt2);
    if (cnt2 < cnt1) {
      flags |= 1 << i;
    }

assert(bits[i] == bbits * chunksize);
    bbits += cnt;
//    out[i] = cnt;  // store logn value
assert(out[i] == cnt);
  }
  bbits *= chunksize;
assert(bbits == *total_bits);

  // check if encoded data fits
  *newsize = (TB + SC + 16 + 8 * SC + bbits) / 8;
  assert(ns == *newsize);
  if (*newsize < CS) {

  // encode data values
  int loc = TB;
  int cpos = 1;
  T cval = 0;
  for (int i = 0; i < SC; i++) {
    const int logn = out[i];
    if (logn > 0) {
      const int beg = i * chunksize;
      const int end = beg + chunksize;
      if (logn == TB) {
        const int pos = loc / TB;
        if (pos > cpos) {
          //out_t[cpos] = cval;
        }
        //printf("S start = %ld (%d)\n", &out_t[pos], i);
        //memcpy(&out_t[pos], &in_t[beg], chunksize * sizeof(T));
        loc += chunksize * TB;
        cpos = loc / TB;
        cval = 0;
      } else {
        const bool flag = flags & (1 << i);
        for (int j = beg; j < end; j++) {
          T val = in_t[j];
          if (flag) {
            val = (val << 1) ^ (((std::make_signed_t<T>)val) >> (sizeof(T) * 8 - 1));  // TCMS
          }
          const int pos = loc / TB;
          const int shift = loc % TB;
          if (pos > cpos) {
            //out_t[cpos] = cval;
            cpos++;
            cval = 0;
          }
          cval |= val << shift;
          if (TB - logn < shift) {
            //out_t[cpos] = cval;
            cpos++;
            cval = val >> (TB - shift);
          }
          loc += logn;
        }
      }
    }
  }
  if (loc > cpos * TB) {
    //out_t[cpos] = cval;
  }
/*
  // output header info 2
  out[*newsize - 6] = flags;
  out[*newsize - 5] = flags >> 8;
  out[*newsize - 4] = flags >> 16;
  out[*newsize - 3] = flags >> 24;
  out[*newsize - 2] = csize;
  out[*newsize - 1] = csize >> 8;
  */
  }
}
__syncthreads();

  if (*newsize >= CS) return false;

  csize = *newsize;
  return true;
}


template <typename T>
static __device__ inline bool d_FAST2xxx(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int tid = threadIdx.x;
  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;
  const int warps = TPB / WS;

  const int TB = sizeof(T) * 8;  // number of bits in T
  const int size = CS / sizeof(T);
  const int SC = 32;  // subchunks [do not change]
  const int chunksize = size / SC;

  static_assert(WS >= SC);
  static_assert(SC == sizeof(int) * 8);
  static_assert(std::is_unsigned<T>::value);

  // T casts
  T* const in_t = (T*)in;
  T* const out_t = (T*)&out[SC];
/*
  for (int i = tid; i < size; i += TPB) {
    T* const out_t = (T*)out;
    out_t[i] = 0;
  }
  __syncthreads();
*/
if (tid == 0) {
  for (int i = 0; i < CS; i++) {
    out[i] = 0;
  }
}
  __syncthreads();
/*
  // clear out unused part of input buffer
  if (csize < CS) {
    for (int i = csize + tid; i < CS; i += TPB) {
      in[i] = 0;
    }
    __syncthreads();
  }
*/
if (tid == 0) {
  // clear out unused part of input buffer
  for (int i = csize; i < CS; i++) {
    in[i] = 0;
  }
}
__syncthreads();

  int* const bits = (int*)temp;
  int* const total_bits = &bits[WS+128];
  int* const flags = &total_bits[1+128];

  // copy first value and set it to zero
  if (tid == 0) {
    out_t[0] = in_t[0];
    in_t[0] = 0;
    *flags = 0;  // clear flags
  }
  __syncthreads();
/*
  // determine bits needed for each subchunk
  for (int i = warp; i < SC; i += warps) {
    // compute maximum value using both approaches
    const int beg = i * chunksize;
    const int end = beg + chunksize;
    T max_val1 = 0;
    T max_val2 = 0;
    // max of values for each thread
    for (int j = beg + lane; j < end; j += WS) {
      const T val1 = in_t[j];
      max_val1 = max(max_val1, val1);
      const T val2 = (val1 << 1) ^ (((std::make_signed_t<T>)val1) >> (sizeof(T) * 8 - 1));  // TCMS
      max_val2 = max(max_val2, val2);
    }

    // warp level max
    max_val1 = max(max_val1, __shfl_xor(max_val1, 1));
    max_val1 = max(max_val1, __shfl_xor(max_val1, 2));
    max_val1 = max(max_val1, __shfl_xor(max_val1, 4));
    max_val1 = max(max_val1, __shfl_xor(max_val1, 8));
    max_val1 = max(max_val1, __shfl_xor(max_val1, 16));
#if defined(WS) && (WS == 64)
    max_val1 = max(max_val1, __shfl_xor(max_val1, 32));
#endif
    max_val2 = max(max_val2, __shfl_xor(max_val2, 1));
    max_val2 = max(max_val2, __shfl_xor(max_val2, 2));
    max_val2 = max(max_val2, __shfl_xor(max_val2, 4));
    max_val2 = max(max_val2, __shfl_xor(max_val2, 8));
    max_val2 = max(max_val2, __shfl_xor(max_val2, 16));
#if defined(WS) && (WS == 64)
    max_val2 = max(max_val2, __shfl_xor(max_val2, 32));
#endif

    // figure out number of bits needed
    if (lane == 0) {
      int cnt1 = 0;
      if (max_val1 != 0) {
        cnt1 = (sizeof(T) == 8) ? (sizeof(unsigned long long) * 8 - __clzll((unsigned long long)max_val1)) : (sizeof(unsigned int) * 8 - __clz((unsigned int)max_val1));
      }
      int cnt2 = 0;
      if (max_val2 != 0) {
        cnt2 = (sizeof(T) == 8) ? (sizeof(unsigned long long) * 8 - __clzll((unsigned long long)max_val2)) : (sizeof(unsigned int) * 8 - __clz((unsigned int)max_val2));
      }

      // use approach requiring fewer bits
      const int cnt = min(cnt1, cnt2);
      if (cnt2 < cnt1) {
        atomicOr_block(flags, 1 << i);  // set flag
      }
      out[i] = cnt;  // store logn value
//printf("G logn[%d] = %d\n", i, cnt);
    }
  }
  __syncthreads();

  // warp prefix sum over bits
  if (warp == 0) {
    const int org = (lane < SC) ? (out[lane] * chunksize) : 0;
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
    if (lane == SC - 1) *total_bits = val;
  }
  __syncthreads();
*/
if (tid == 0) {
  // determine bits needed for each subchunk
  //int flags = 0;
  int bbb = 0;
  for (int i = 0; i < SC; i++) {
    // compute maximum value using both approaches
    const int beg = i * chunksize;
    const int end = beg + chunksize;
    T max_val1 = 0;
    T max_val2 = 0;
    for (int j = beg; j < end; j++) {
      const T val1 = in_t[j];
      max_val1 = max(max_val1, val1);
      const T val2 = (val1 << 1) ^ (((std::make_signed_t<T>)val1) >> (sizeof(T) * 8 - 1));  // TCMS
      max_val2 = max(max_val2, val2);
    }

    // figure out number of bits needed
    int cnt1 = 0;
    if (max_val1 != 0) {
      cnt1 = (sizeof(T) == 8) ? (sizeof(unsigned long long) * 8 - __clzll((unsigned long long)max_val1)) : (sizeof(unsigned int) * 8 - __clz((unsigned int)max_val1));
    }
    int cnt2 = 0;
    if (max_val2 != 0) {
      cnt2 = (sizeof(T) == 8) ? (sizeof(unsigned long long) * 8 - __clzll((unsigned long long)max_val2)) : (sizeof(unsigned int) * 8 - __clz((unsigned int)max_val2));
    }

    // use approach requiring fewer bits
    const int cnt = min(cnt1, cnt2);
    if (cnt2 < cnt1) {
      *flags |= 1 << i;
    }

    bits[i] = bbb * chunksize;
    bbb += cnt;
    out[i] = cnt;  // store logn value
//printf("C logn[%d] = %d\n", i, cnt);
  }
  *total_bits = bbb * chunksize;
}
__syncthreads();

  // check if encoded data fits
  const int newsize = (TB + SC + 16 + 8 * SC + *total_bits) / 8;
  if (newsize >= CS) return false;
//if (tid == 0) printf("G newsize = %d\n", newsize);
/*
  // encode data values
  for (int i = warp; i < SC; i += warps) {
    const int logn = out[i];
    if (logn > 0) {
      const int beg = i * chunksize;
      const int end = beg + chunksize;
      if (logn == TB) {
        const int offs = (TB + bits[i]) / TB - beg;
        for (int j = beg + lane; j < end; j += WS) {
          out_t[offs + j] = in_t[j];
        }
      } else {
        const int offs = TB + bits[i];
        const bool flag = *flags & (1 << i);
        //const T mask = ((T)1 << logn) - 1;
        for (int j = beg + lane; j < end; j += WS) {
          T val = in_t[j];
          if (flag) {
            val = (val << 1) ^ (((std::make_signed_t<T>)val) >> (sizeof(T) * 8 - 1));  // TCMS
          }
          const int loc = offs + (j - beg) * logn;
          const int pos = loc / TB;
          const int shift = loc % TB;
          //atomicAnd_block(&out_t[pos], ~(mask << shift));
          atomicOr_block(&out_t[pos], val << shift);
          if (TB - logn < shift) {
            //atomicAnd_block(&out_t[pos + 1], ~(mask >> (TB - shift)));
            atomicOr_block(&out_t[pos + 1], val >> (TB - shift));
          }
        }
      }
    }
  }
  //__syncthreads();
*/

if (tid == 0) {
  // encode data values
  int loc = TB;
  int cpos = 1;
  T cval = 0;
  for (int i = 0; i < SC; i++) {
    const int logn = out[i];
    if (logn > 0) {
      const int beg = i * chunksize;
      const int end = beg + chunksize;
      if (logn == TB) {
        const int pos = loc / TB;
        if (pos > cpos) {
          out_t[cpos] = cval;
        }
        //memcpy(&out_t[pos], &in_t[beg], chunksize * sizeof(T));
        for (int k = 0; k < chunksize; k++) {
          out_t[pos + k] = in_t[beg + k];
        }
        loc += chunksize * TB;
        cpos = loc / TB;
        cval = 0;
      } else {
        const bool flag = *flags & (1 << i);
        for (int j = beg; j < end; j++) {
          T val = in_t[j];
          if (flag) {
            val = (val << 1) ^ (((std::make_signed_t<T>)val) >> (sizeof(T) * 8 - 1));  // TCMS
          }
          const int pos = loc / TB;
          const int shift = loc % TB;
          if (pos > cpos) {
            out_t[cpos] = cval;
            cpos++;
            cval = 0;
          }
          cval |= val << shift;
          if (TB - logn < shift) {
            out_t[cpos] = cval;
            cpos++;
            cval = val >> (TB - shift);
          }
          loc += logn;
        }
      }
    }
  }
  if (loc > cpos * TB) {
    out_t[cpos] = cval;
  }
}
__syncthreads();
/*
  // output header info
  if (tid < 4) {
    const int shift = tid * 8;
    out[newsize - 6 + tid] = *flags >> shift;
    if (tid < 2) {
      out[newsize - 2 + tid] = csize >> shift;
    }
  }
*/
if (tid == 0) {
  // output header info 2
  out[newsize - 6] = *flags;
  out[newsize - 5] = *flags >> 8;
  out[newsize - 4] = *flags >> 16;
  out[newsize - 3] = *flags >> 24;
  out[newsize - 2] = csize;
  out[newsize - 1] = csize >> 8;
}
__syncthreads();

  csize = newsize;
  return true;
}


template <typename T>
static __device__ inline void d_iFAST2(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;
  const int warps = TPB / WS;

  const int TB = sizeof(T) * 8;  // number of bits in T
  const int size = CS / sizeof(T);
  const int SC = 32;  // subchunks [do not change]
  const int chunksize = size / SC;

  static_assert(WS >= SC);
  static_assert(SC == sizeof(int) * 8);
  static_assert(std::is_unsigned<T>::value);

  // T casts
  const T* const in_t = (T*)&in[SC];
  T* const out_t = (T*)out;
  int* const bits = (int*)temp;

  // read header info
  const int orig_csize = in[csize - 2] | ((int)in[csize - 1] << 8);
  const int flags = in[csize - 6] | ((int)in[csize - 5] << 8) | ((int)in[csize - 4] << 16) | ((int)in[csize - 3] << 24);

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
        out_t[j] = 0;
      }
    } else if (logn == TB) {
      const int offs = (TB + bits[i]) / TB - beg;
      for (int j = beg + lane; j < end; j += WS) {
        out_t[j] = in_t[offs + j];
      }
    } else {
      const int offs = TB + bits[i];
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
        out_t[j] = val;
      }
    }
  }

  if (threadIdx.x == 0) {
    out_t[0] = in_t[0];  // copy first value verbatim
  }
  csize = orig_csize;
}


#endif
