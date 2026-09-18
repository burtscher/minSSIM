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


#ifndef GPU_SLEEK
#define GPU_SLEEK


template <typename T>
static __device__ inline bool d_SLEEK(int& csize, byte in [CS], byte out [CS], byte temp [CS])
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

  // clear unused part of input buffer
  T* const in_t = (T*)in;
  if (csize < CS) {
    if (csize % sizeof(T) == 0) {
      for (int i = csize / sizeof(T) + tid; i < size; i += TPB) {
        in_t[i] = 0;
      }
    } else {
      for (int i = csize + tid; i < CS; i += TPB) {
        in[i] = 0;
      }
    }
    __syncthreads();
  }

  // determine bits needed for each subchunk
  int ln = -1;
  for (int i = warp; i < SC; i += warps) {
    const int beg = i * chunksize;
    const int end = beg + chunksize;

    // max of values for each thread
    T max_val = 0;
    for (int j = beg + lane; j < end; j += WS) {
      T val = in_t[j];
      if constexpr (TB == 32) {
        val = ((val << 1) | (val >> 31));
        if ((val & 0xff00'0000) != 0) {
          val -= 0x8000'0000;
          if (((val & 0xff00'0000) == 0) || (val >= 0x8000'0000)) {
            val -= 0x0100'0000;
          }
        }
        val = (val << 1) ^ (((int)val) >> 31);  // TCMS
      } else {
        val = ((val << 1) | (val >> 63));
        if ((val & 0xffe0'0000'0000'0000) != 0) {
          val -= 0x8000'0000'0000'0000;
          if (((val & 0xffe0'0000'0000'0000) == 0) || (val >= 0x8000'0000'0000'0000)) {
            val -= 0x0020'0000'0000'0000;
          }
        }
        val = (val << 1) ^ (((long long)val) >> 63);  // TCMS
      }
      in_t[j] = val;
      max_val = max(max_val, val);
    }

    // warp level max
    max_val = max(max_val, __shfl_xor(max_val, 1));
    max_val = max(max_val, __shfl_xor(max_val, 2));
    max_val = max(max_val, __shfl_xor(max_val, 4));
    max_val = max(max_val, __shfl_xor(max_val, 8));
    max_val = max(max_val, __shfl_xor(max_val, 16));

    // figure out number of bits needed
    if (lane == i) {
      int cnt = TB;
      if (max_val != 0) {
        cnt = (TB == 64) ? __clzll(max_val) : __clz(max_val);
      }
      ln = TB - cnt;  // logn value for each chunk
    }
  }
  if (ln >= 0) out[lane] = ln;
  __syncthreads();

  // warp prefix sum over bits
  int* const bits = (int*)temp;
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
    if (lane == SC - 1) bits[SC] = val;
  }
  __syncthreads();

  // check if encoded data fits
  const int tot = bits[SC];
  const int newsize = (SC * 8 + tot + 16) / 8;
  if (newsize >= CS) return false;

  // clear out buffer
  T* const out_t = (T*)&out[SC];
  for (int i = tid; i < tot / TB; i += TPB) out_t[i] = 0;
  __syncthreads();

  // encode data values
  for (int i = warp; i < SC; i += warps) {
    const int logn = out[i];
    if (logn > 0) {
      const int beg = i * chunksize;
      const int end = beg + chunksize;
      if (logn == TB) {
        const int offs = bits[i] / TB - beg;
        for (int j = beg + lane; j < end; j += WS) {
          out_t[offs + j] = in_t[j];
        }
      } else {
        const int incr = WS * logn;
        int loc = bits[i] + lane * logn;
        for (int j = beg + lane; j < end; j += WS) {
          const T val = in_t[j];
          const int pos = loc / TB;
          const int shift = loc % TB;
          atomicOr_block(&out_t[pos], val << shift);
          if (TB - shift < logn) {
            atomicOr_block(&out_t[pos + 1], val >> (TB - shift));
          }
          loc += incr;
        }
      }
    }
  }

  // output header info
  if (tid == 0) {
    *(short*)&out[newsize - 2] = csize;
  }

  csize = newsize;
  return true;
}


template <typename T>
static __device__ inline void d_iSLEEK(int& csize, byte in [CS], byte out [CS], byte temp [CS])
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

  // warp prefix sum over bits
  int* const bits = (int*)temp;
  if (warp == 0) {
    const int org = in[lane] * chunksize;
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
  const T* const in_t = (T*)&in[SC];
  T* const out_t = (T*)out;
  for (int i = warp; i < SC; i += warps) {
    const int logn = in[i];
    const int beg = i * chunksize;
    const int end = beg + chunksize;
    if (logn == 0) {
      for (int j = beg + lane; j < end; j += WS) {
        out_t[j] = 0;
      }
    } else if (logn == TB) {
      const int offs = bits[i] / TB - beg;
      for (int j = beg + lane; j < end; j += WS) {
        T val = in_t[offs + j];
        if constexpr (TB == 32) {
          val = (val >> 1) ^ (((int)(val << 31)) >> 31);  // iTCMS
          if ((val & 0xff00'0000) != 0) {
            if (val >= 0x8000'0000) {
              val += 0x0100'0000;
            }
            val += 0x8000'0000;
          }
          val = (val << 31) | (val >> 1);
        } else {
          val = (val >> 1) ^ (((long long)(val << 63)) >> 63);  // iTCMS
          if ((val & 0xffe0'0000'0000'0000) != 0) {
            if (val >= 0x8000'0000'0000'0000) {
              val += 0x0020'0000'0000'0000;
            }
            val += 0x8000'0000'0000'0000;
          }
          val = (val << 63) | (val >> 1);
        }
        out_t[j] = val;
      }
    } else {
      const int incr = WS * logn;
      int loc = bits[i] + lane * logn;
      const T mask = (logn == 64) ? (~0ULL) : ((1ULL << logn) - 1);
      for (int j = beg + lane; j < end; j += WS) {
        const int pos = loc / TB;
        const int shift = loc % TB;
        T res = in_t[pos] >> shift;
        if (TB - shift < logn) {
          res |= in_t[pos + 1] << (TB - shift);
        }
        loc += incr;
        T val = res & mask;
        if constexpr (TB == 32) {
          val = (val >> 1) ^ (((int)(val << 31)) >> 31);  // iTCMS
          if ((val & 0xff00'0000) != 0) {
            if (val >= 0x8000'0000) {
              val += 0x0100'0000;
            }
            val += 0x8000'0000;
          }
          val = (val << 31) | (val >> 1);
        } else {
          val = (val >> 1) ^ (((long long)(val << 63)) >> 63);  // iTCMS
          if ((val & 0xffe0'0000'0000'0000) != 0) {
            if (val >= 0x8000'0000'0000'0000) {
              val += 0x0020'0000'0000'0000;
            }
            val += 0x8000'0000'0000'0000;
          }
          val = (val << 63) | (val >> 1);
        }
        out_t[j] = val;
      }
    }
  }

  // read header info
  csize = *(short*)&in[csize - 2];
}


#endif
