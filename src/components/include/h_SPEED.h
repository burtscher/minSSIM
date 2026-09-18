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


#ifndef CPU_SPEED
#define CPU_SPEED


template <typename T>
static inline bool h_SPEED(int& csize, byte in [CS], byte out [CS])
{
  static_assert(sizeof(T) >= 4);
  static_assert(std::is_unsigned<T>::value);
  const int TB = sizeof(T) * 8;  // number of bits in T
  const int size = CS / sizeof(T);
  const int SC = 32;  // subchunks [do not change]
  const int chunksize = size / SC;
  static_assert(SC == sizeof(int) * 8);

  // clear out unused part of input buffer
  if (csize < CS) {
    memset(&in[csize], 0, CS - csize);
  }

  // type casts
  T* in_t = (T*)in;
  T* out_t = (T*)&out[SC];

  // determine bits needed for each subchunk
  int flags = 0;
  int bits = 0;
  for (int i = 0; i < SC; i++) {
    const int beg = i * chunksize;
    const int end = beg + chunksize;

    // copy first value and indirectly set it to zero
    const T firstval = in_t[beg];
    out_t[i] = firstval;
    T prev = firstval;

    // compute maximum value using both approaches
    T max_val1 = 0;
    T max_val2 = 0;
    for (int j = beg; j < end; j++) {
      // compute difference sequence plus TCMS
      const T val = in_t[j];
      const T data = val - prev;
      prev = val;
      const T val1 = (data << 1) ^ ((std::make_signed_t<T>)data) >> (sizeof(T) * 8 - 1);  // TCMS
      in_t[j] = val1;
      max_val1 = std::max(max_val1, val1);
      const T val2 = (val1 << 1) ^ (((std::make_signed_t<T>)val1) >> (sizeof(T) * 8 - 1));  // TCMS
      max_val2 = std::max(max_val2, val2);
    }

    // figure out number of bits needed
    int cnt1 = 0;
    if (max_val1 != 0) {
      cnt1 = (sizeof(T) == 8) ? (64 - __builtin_clzll((unsigned long long)max_val1)) : (32 - __builtin_clz((unsigned int)max_val1));
    }
    int cnt2 = 0;
    if (max_val2 != 0) {
      cnt2 = (sizeof(T) == 8) ? (64 - __builtin_clzll((unsigned long long)max_val2)) : (32 - __builtin_clz((unsigned int)max_val2));
    }

    // use approach requiring fewer bits
    const int cnt = std::min(cnt1, cnt2);
    if (cnt2 < cnt1) {
      flags |= 1 << i;
    }

    bits += cnt;
    out[i] = cnt;  // store logn value
  }
  bits *= chunksize;

  // check if encoded data fits
  const int newsize = (SC * 8 + SC * TB + SC + 16 + bits) / 8;
  if (newsize >= CS) return false;

  // encode data values
  int loc = SC * TB;
  int cpos = SC;
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
        memcpy(&out_t[pos], &in_t[beg], chunksize * sizeof(T));
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

  // output header info
  *(int*)&out[newsize - 6] = flags;
  *(short*)&out[newsize - 2] = csize;

  csize = newsize;
  return true;
}


template <typename T>
static inline void h_iSPEED(int& csize, byte in [CS], byte out [CS])
{
  static_assert(sizeof(T) >= 4);
  static_assert(std::is_unsigned<T>::value);
  const int TB = sizeof(T) * 8;  // number of bits in T
  const int size = CS / sizeof(T);
  const int SC = 32;  // subchunks [do not change]
  const int chunksize = size / SC;
  static_assert(SC == sizeof(int) * 8);

  // read header info
  const int orig_csize = *(short*)&in[csize - 2];
  const int flags = *(int*)&in[csize - 6];

  // type casts
  T* in_t = (T*)&in[SC];
  T* out_t = (T*)out;

  // decode data values
  int loc = SC * TB;
  int cpos = 0;
  for (int i = 0; i < SC; i++) {
    const int logn = in[i];
    T sum = in_t[i];  // firstval
    const int beg = i * chunksize;
    const int end = beg + chunksize;
    if (logn == 0) {
      for (int j = beg; j < end; j++) {
        out_t[j] = sum;
      }
    } else if (logn == TB) {
      const int offs = loc / TB - beg;
      for (int j = beg; j < end; j++) {
        const T val = in_t[offs + j];
        const T diff = (val >> 1) ^ ((std::make_signed_t<T>)(val << (sizeof(T) * 8 - 1))) >> (sizeof(T) * 8 - 1);  // iTCMS
        sum += diff;
        out_t[j] = sum;
      }
      loc += chunksize * TB;
    } else {
      T cval;
      const T mask = ((T)1 << logn) - 1;
      const bool flag = flags & (1 << i);
      for (int j = beg; j < end; j++) {
        const int pos = loc / TB;
        const int shift = loc % TB;
        if (pos > cpos) {
          cval = in_t[pos];
          cpos = pos;
        }
        T res = cval >> shift;
        if (TB - logn < shift) {
          cpos++;
          cval = in_t[cpos];
          res |= cval << (TB - shift);
        }
        T val = res & mask;
        if (flag) {
          val = (val >> 1) ^ ((std::make_signed_t<T>)(val << (sizeof(T) * 8 - 1))) >> (sizeof(T) * 8 - 1);  // iTCMS
        }
        const T diff = (val >> 1) ^ ((std::make_signed_t<T>)(val << (sizeof(T) * 8 - 1))) >> (sizeof(T) * 8 - 1);  // iTCMS
        sum += diff;
        out_t[j] = sum;
        loc += logn;
      }
    }
  }

  csize = orig_csize;
}


#endif
