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


#ifndef LC_TOPO_H_
#define LC_TOPO_H_


#include <atomic>


static inline int topo_atomicMax_f32(int* const addr, const int val)
{
  std::atomic<int>* ai = reinterpret_cast<std::atomic<int>*>(addr);
  int old = ai->load(std::memory_order_relaxed);
  do {
    if (old >= val) break;
  } while (!ai->compare_exchange_weak(old, val, std::memory_order_relaxed, std::memory_order_relaxed));
  return old;
}


static inline int topo_atomicMax_f64(long long* const addr, const long long val)
{
  std::atomic<long long>* ai = reinterpret_cast<std::atomic<long long>*>(addr);
  long long old = ai->load(std::memory_order_relaxed);
  do {
    if (old >= val) break;
  } while (!ai->compare_exchange_weak(old, val, std::memory_order_relaxed, std::memory_order_relaxed));
  return old;
}


static inline unsigned int topo_encode_f32(const int eb_e, const unsigned int val)
{
  const int e = 8;  // exponent bits
  const int m = 23;  // mantissa bits
  const int thr_e = eb_e + (m + 1);
  const int offs = (thr_e << m) - (1 << m);

  const int abs = val & (((unsigned int)1 << (e + m)) - 1);
  const int val_e = abs >> m;
  unsigned int enc = 0;
  if (val_e >= thr_e) {
    enc = abs - offs;
  } else if (val_e >= eb_e) {
    int mant = val & ((1 << m) - 1);
    const int shift = thr_e - val_e;  // bias cancels out
    mant |= 1 << m;  // insert implicit 1
    mant += 1 << (shift - 1);  // round to nearest, ties round away from zero
    enc = mant >> shift;
  }
  enc = (enc << 1) | (~val >> (e + m));  // store sign bit in LSB
  if (enc != 0) enc--;  // eliminate -0
  return enc;
}


static inline unsigned long long topo_encode_f64(const int eb_e, const unsigned long long val)
{
  const int e = 11;  // exponent bits
  const int m = 52;  // mantissa bits
  const int thr_e = eb_e + (m + 1);
  const long long offs = ((long long)thr_e << m) - (1LL << m);

  const long long abs = val & ((1ULL << (e + m)) - 1ULL);
  const int val_e = abs >> m;
  unsigned long long enc = 0ULL;
  if (val_e >= thr_e) {
    enc = abs - offs;
  } else if (val_e >= eb_e) {
    long long mant = val & ((1LL << m) - 1LL);
    const int shift = thr_e - val_e;  // bias cancels out
    mant |= 1LL << m;  // insert implicit 1
    mant += 1LL << (shift - 1);  // round to nearest, ties round away from zero
    enc = mant >> shift;
  }
  enc = (enc << 1) | (~val >> (e + m));  // store sign bit in LSB
  if (enc != 0LL) enc--;  // eliminate -0
  return enc;
}


static inline unsigned int topo_decode_f32(const int eb_e, const unsigned int enc, unsigned int& range)
{
  const int e = 8;  // exponent bits
  const int m = 23;  // mantissa bits
  const int thr_e = eb_e + (m + 1);
  const int offs = (thr_e << m) - (1 << m);

  unsigned int dec = 0;  // default value is 0
  range = eb_e << m;
  if (enc != 0) {
    const int abs = (enc + 1) >> 1;
    if (abs >= (1 << m)) {
      dec = abs + offs;
      range = 1;
    } else {
      const int shift = __builtin_clz(abs) - (31 - m);
      dec = abs << shift;  // shift to normalized position
      dec &= (1 << m) - 1;  // remove implied 1
      dec |= (thr_e - shift) << m;  // insert biased exponent
      range = 1 << shift;
    }
    dec |= enc << (e + m);  // insert sign bit
  }
  return dec;
}


static inline unsigned long long topo_decode_f64(const int eb_e, const unsigned long long enc, unsigned long long& range)
{
  const int e = 11;  // exponent bits
  const int m = 52;  // mantissa bits
  const int thr_e = eb_e + (m + 1);
  const long long offs = ((long long)thr_e << m) - (1LL << m);

  unsigned long long dec = 0ULL;  // default value is 0
  range = (unsigned long long)eb_e << m;
  if (enc != 0) {
    const long long abs = (enc + 1) >> 1;
    if (abs >= (1LL << m)) {
      dec = abs + offs;
      range = 1;
    } else {
      const int shift = __builtin_clzll(abs) - (63 - m);
      dec = abs << shift;  // shift to normalized position
      dec &= (1LL << m) - 1LL;  // remove implied 1
      dec |= ((long long)thr_e - shift) << m;  // insert biased exponent
      range = 1LL << shift;
    }
    dec |= enc << (e + m);  // insert sign bit
  }
  return dec;
}


#endif /* LC_TOPO_H_ */
