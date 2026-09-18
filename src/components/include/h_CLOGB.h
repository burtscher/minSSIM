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


#ifndef CPU_CLOGB
#define CPU_CLOGB


template <typename T>
static inline bool h_CLOGB(int& csize, byte in [CS], byte out [CS])
{
  static_assert(std::is_unsigned<T>::value);
  const int constexpr TB = sizeof(T) * 8;  // number of bits in T
  const int constexpr SC = 32;  // subchunks [do not change]
  const int constexpr CB = (sizeof(T) >= 4) ? ((sizeof(T) == 8) ? 7 : 6) : ((sizeof(T) == 2) ? 5 : 4);  // counter bits
  static_assert((1 << CB) > TB);
  static_assert((1 << (CB - 1)) <= TB);

  // type casts
  T* const in_t = (T*)in;
  T* const out_t = (T*)out;
  const int size = csize / sizeof(T);

  // determine bits needed for each subchunk
  byte ln [SC];
  byte tr [SC];
  int end = 0;
  int saved = 0;
  for (int i = 0; i < SC; i++) {
    const int beg = end;
    end = (i + 1) * size / SC;
    T or_val = 0;
    for (int j = beg; j < end; j++) {
      or_val |= in_t[j];
    }
    int cnt = 0;
    int trunc = 0;
    if (or_val != 0) {
      trunc = (sizeof(T) == 8) ? (__builtin_ffsll((unsigned long long)or_val) - 1) : (__builtin_ffs((unsigned int)or_val) - 1);
      or_val >>= trunc;
      cnt = (sizeof(T) == 8) ? (sizeof(unsigned long long) * 8 - __builtin_clzll((unsigned long long)or_val)) : (sizeof(unsigned int) * 8 - __builtin_clz((unsigned int)or_val));
    }
    ln[i] = cnt;  // logn value for each subchunk
    tr[i] = trunc;  // truncation value for each subchunk
    saved += (end - beg) * trunc;
  }

  // check if truncating LSBs is worth it
  const int flag = (saved > CB * SC) ? 0x8000 : 0;
  const bool cond = (saved > 0) && !flag;
  int bits = 0;
  end = 0;
  for (int i = 0; i < SC; i++) {
    const int beg = end;
    end = (i + 1) * size / SC;
    int cnt = ln[i];
    if (cond) {
      cnt += tr[i];
      ln[i] = cnt;
    }
    bits += cnt * (end - beg);
  }

  // check if encoded data fits
  int newsize = (16 + CB * SC + bits + 7) / 8;
  if (flag) newsize += CB * SC / 8;
  const int extra = csize % sizeof(T);
  if (newsize + extra >= CS) return false;

  // clear out buffer
  memset(out_t, 0, newsize);

  // output header
  if constexpr (sizeof(T) > 1) {
    out_t[0] = csize | flag;
  } else {
    out[0] = csize;
    out[1] = (csize | flag) >> 8;
  }

  // encode logn values
  int loc = 16;
  for (int i = 0; i < SC; i++) {
    const T val = ln[i];
    const int pos = loc / TB;
    const int shift = loc % TB;
    out_t[pos] |= val << shift;
    if (TB - CB < shift) {
      out_t[pos + 1] = val >> (TB - shift);
    }
    loc += CB;
  }

  // encode data values
  end = 0;
  for (int i = 0; i < SC; i++) {
    const int logn = ln[i];
    const int beg = end;
    end = (i + 1) * size / SC;
    if (logn > 0) {
      const int trunc = (flag ? tr[i] : 0);
      for (int j = beg; j < end; j++) {
        const T val = in_t[j] >> trunc;
        const int pos = loc / TB;
        const int shift = loc % TB;
        out_t[pos] |= val << shift;
        if (TB - logn < shift) {
          out_t[pos + 1] = val >> (TB - shift);
        }
        loc += logn;
      }
    }
  }

  // encode trunc values
  if (flag) {
    loc = newsize * 8 - CB * SC;
    for (int i = 0; i < SC; i++) {
      const T val = tr[i];
      const int pos = loc / TB;
      const int shift = loc % TB;
      out_t[pos] |= val << shift;
      if (TB - CB < shift) {
        out_t[pos + 1] = val >> (TB - shift);
      }
      loc += CB;
    }
  }

  // copy extra bytes at end and update csize
  if constexpr (sizeof(T) > 1) {
    for (int i = 0; i < extra; i++) out[newsize + i] = in[csize - extra + i];
  }
  csize = newsize + extra;
  return true;
}


template <typename T>
static inline void h_iCLOGB(int& csize, byte in [CS], byte out [CS])
{
  static_assert(std::is_unsigned<T>::value);
  const int constexpr TB = sizeof(T) * 8;  // number of bits in T
  const int constexpr SC = 32;  // subchunks [do not change]
  const int constexpr CB = (sizeof(T) >= 4) ? ((sizeof(T) == 8) ? 7 : 6) : ((sizeof(T) == 2) ? 5 : 4);  // counter bits
  static_assert((1 << CB) > TB);
  static_assert((1 << (CB - 1)) <= TB);

  // type casts
  T* const in_t = (T*)in;
  T* const out_t = (T*)out;

  // decode csize
  const int head = *((unsigned short*)in);
  const int orig_csize = head & 0x7fff;
  const bool flag = (head >= 0x8000);

  // decode logn values
  int loc = 16;
  byte ln [SC];
  const T constexpr mask = ((1 << CB) - 1);
  for (int i = 0; i < SC; i++) {
    const int pos = loc / TB;
    const int shift = loc % TB;
    T res = in_t[pos] >> shift;
    if (TB - CB < shift) {
      res |= in_t[pos + 1] << (TB - shift);
    }
    ln[i] = res & mask;
    loc += CB;
  }

  // decode trunc values
  byte tr [SC];
  const int extra = orig_csize % sizeof(T);
  if (flag) {
    int loc = (csize - extra) * 8 - CB * SC;  // different location
    for (int i = 0; i < SC; i++) {
      const int pos = loc / TB;
      const int shift = loc % TB;
      T res = in_t[pos] >> shift;
      if (TB - CB < shift) {
        res |= in_t[pos + 1] << (TB - shift);
      }
      tr[i] = res & mask;
      loc += CB;
    }
  }

  // decode data values
  const int size = orig_csize / sizeof(T);
  int end = 0;
  for (int i = 0; i < SC; i++) {
    const int logn = ln[i];
    const int beg = end;
    end = (i + 1) * size / SC;
    if (logn > 0) {
      const int trunc = (flag ? tr[i] : 0);
      const T mask = (sizeof(T) < 8) ? ((1ULL << logn) - 1) : ((logn == 64) ? (~0ULL) : ((1ULL << logn) - 1));
      for (int j = beg; j < end; j++) {
        const int pos = loc / TB;
        const int shift = loc % TB;
        T res = in_t[pos] >> shift;
        if (TB - logn < shift) {
          res |= in_t[pos + 1] << (TB - shift);
        }
        out_t[j] = (res & mask) << trunc;
        loc += logn;
      }
    } else {
      for (int j = beg; j < end; j++) {
        out_t[j] = 0;
      }
    }
  }

  // copy extra bytes at end and update csize
  if constexpr (sizeof(T) > 1) {
    for (int i = 0; i < extra; i++) out[orig_csize - extra + i] = in[csize - extra + i];
  }
  csize = orig_csize;
}


#endif
