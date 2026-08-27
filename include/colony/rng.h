#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>

namespace colony {

// =====================================================================
// CPython random.Random — Mersenne Twister 19937 (init_by_array + genrand_res53)
// Bit-exact port of CPython 3.14 Modules/_randommodule.c
// =====================================================================
class MtRandom {
public:
    explicit MtRandom(int64_t seed) { seed_mt(seed); }

    // random() -> x in [0, 1), 53-bit resolution (genrand_res53)
    double random();

    // getrandbits(k), 0 < k <= 64 (CPython: k<=32 fast path, else words)
    uint64_t getrandbits(int k);

    // randrange(n) = _randbelow(n), [0, n)
    uint64_t randrange(uint64_t n);

    // randint(a, b) = a + _randbelow(b - a + 1)
    int64_t randint(int64_t a, int64_t b);

    // uniform(a, b) = a + (b - a) * random()
    double uniform(double a, double b);

private:
    void seed_mt(int64_t seed);
    uint32_t genrand_uint32();
    void init_genrand(uint32_t s);
    void init_by_array(uint32_t init_key[], size_t key_length);
    uint64_t randbelow(uint64_t n);

    uint32_t mt_[624];
    int index_;
};

// =====================================================================
// numpy SeedSequence + PCG64 (default_rng) — bit-exact port
// SeedSequence: numpy 2.5.2 numpy/random/bit_generator.pyx (hashmix/mix)
// PCG64: numpy 2.5.2 numpy/random/src/pcg64/{pcg64.c,pcg64.h} (XSL RR 128/64)
// =====================================================================
namespace detail {
constexpr uint32_t SS_INIT_A = 0x43b0d7e5;
constexpr uint32_t SS_MULT_A = 0x931e8875;
constexpr uint32_t SS_INIT_B = 0x8b51f9dd;
constexpr uint32_t SS_MULT_B = 0x58f38ded;
constexpr uint32_t SS_MIX_L = 0xca01f9dd;
constexpr uint32_t SS_MIX_R = 0x4973f715;
constexpr uint32_t SS_XSHIFT = 16;

inline uint32_t ss_hashmix(uint32_t value, uint32_t& hc) {
    value ^= hc;
    hc *= SS_MULT_A;
    value *= hc;
    value ^= value >> SS_XSHIFT;
    return value;
}
inline uint32_t ss_mix(uint32_t x, uint32_t y) {
    uint32_t result = SS_MIX_L * x - SS_MIX_R * y;
    result ^= result >> SS_XSHIFT;
    return result;
}

struct U64Pair {
    uint64_t high;
    uint64_t low;
};

inline U64Pair u64pair_add(U64Pair a, U64Pair b) {
    uint64_t lo = a.low + b.low;
    uint64_t carry = lo < a.low ? 1 : 0;
    return {a.high + b.high + carry, lo};
}

// 64x64 -> 128 bit multiply (exact, portable)
inline U64Pair u64pair_mul64(uint64_t x, uint64_t y) {
    uint64_t x0 = x & 0xffffffffULL, x1 = x >> 32;
    uint64_t y0 = y & 0xffffffffULL, y1 = y >> 32;
    uint64_t w0 = x0 * y0;
    uint64_t t = x1 * y0 + (w0 >> 32);
    uint64_t w1 = t & 0xffffffffULL;
    uint64_t w2 = t >> 32;
    w1 += x0 * y1;
    uint64_t high = x1 * y1 + w2 + (w1 >> 32);
    uint64_t low = x * y;
    return {high, low};
}

// 128x128 -> 128 bit multiply (exact, same scheme as numpy pcg128_mult)
inline U64Pair u64pair_mul(U64Pair a, U64Pair b) {
    uint64_t h1 = a.high * b.low + a.low * b.high;
    U64Pair r = u64pair_mul64(a.low, b.low);
    r.high += h1;
    return r;
}

inline uint64_t rotr64(uint64_t v, unsigned int rot) {
    if (rot == 0) return v;
    return (v >> rot) | (v << (64 - rot));
}
}  // namespace detail

class SeedSequence {
public:
    explicit SeedSequence(uint64_t entropy);

    // generate_state(n_words, uint64) semantics: produce n_words uint64 words
    // (internally 2*n_words uint32 words, then little-endian pair packing)
    void generate_state(uint64_t* out, int n_words64);

    uint32_t pool_[4];  // pool_size = 4 (default)
};

class PCG64 {
public:
    explicit PCG64(uint64_t seed);

    uint64_t next64();              // XSL RR 128/64: step-then-output
    uint32_t next32();              // cached high-word
    double next_double();           // (next64 >> 11) * 2^-53

    uint64_t randrange(uint64_t n);  // [0, n)
    int64_t randint(int64_t a, int64_t b);  // [a, b]
    double uniform(double a, double b);      // [a, b)

    // numpy Generator.binomial(n, p): random_binomial(bitgen, p, n)
    int64_t binomial(int64_t n, double p);

private:
    void step();
    uint64_t output_xsl_rr();
    void set_seed(uint64_t initstate_hi, uint64_t initstate_lo,
                  uint64_t initseq_hi, uint64_t initseq_lo);

    int64_t random_binomial_inversion(int64_t n, double p);
    int64_t random_binomial_btpe(int64_t n, double p);

    detail::U64Pair state_;
    detail::U64Pair inc_;
    int has_uint32_;
    uint32_t uinteger_;
};

}  // namespace colony