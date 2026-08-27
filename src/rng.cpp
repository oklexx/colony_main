#include "colony/rng.h"

#include <cstdlib>
#include <algorithm>

namespace colony {

// =====================================================================
// MtRandom (CPython random.Random / Mersenne Twister 19937)
// =====================================================================

void MtRandom::init_genrand(uint32_t s) {
    mt_[0] = s;
    for (int i = 1; i < 624; i++) {
        mt_[i] = (1812433253u * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) + (uint32_t)i);
    }
    index_ = 624;
}

void MtRandom::init_by_array(uint32_t init_key[], size_t key_length) {
    init_genrand(19650218u);
    size_t i = 1, j = 0;
    size_t k = (624 > key_length ? 624 : key_length);
    for (; k; k--) {
        mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525u)) +
                 init_key[j] + (uint32_t)j;
        i++;
        j++;
        if (i >= 624) {
            mt_[0] = mt_[623];
            i = 1;
        }
        if (j >= key_length) j = 0;
    }
    for (k = 623; k; k--) {
        mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941u)) -
                 (uint32_t)i;
        i++;
        if (i >= 624) {
            mt_[0] = mt_[623];
            i = 1;
        }
    }
    mt_[0] = 0x80000000u;
}

void MtRandom::seed_mt(int64_t seed) {
    uint64_t u;
    if (seed < 0) {
        // PyNumber_Absolute on a Python int: |seed|
        u = (uint64_t)(-(seed + 1)) + 1u;
    } else {
        u = (uint64_t)seed;
    }
    int bits = 0;
    for (uint64_t t = u; t; t >>= 1) bits++;
    size_t keyused = bits == 0 ? 1 : (size_t)((bits - 1) / 32 + 1);
    uint32_t key[2] = {0, 0};
    if (keyused >= 1) key[0] = (uint32_t)(u & 0xffffffffULL);
    if (keyused >= 2) key[1] = (uint32_t)(u >> 32);
    init_by_array(key, keyused);
}

uint32_t MtRandom::genrand_uint32() {
    uint32_t y;
    static const uint32_t mag01[2] = {0x0u, 0x9908b0dfu};
    if (index_ >= 624) {
        int kk;
        for (kk = 0; kk < 624 - 397; kk++) {
            y = (mt_[kk] & 0x80000000u) | (mt_[kk + 1] & 0x7fffffffu);
            mt_[kk] = mt_[kk + 397] ^ (y >> 1) ^ mag01[y & 0x1u];
        }
        for (; kk < 623; kk++) {
            y = (mt_[kk] & 0x80000000u) | (mt_[kk + 1] & 0x7fffffffu);
            mt_[kk] = mt_[kk + (397 - 624)] ^ (y >> 1) ^ mag01[y & 0x1u];
        }
        y = (mt_[623] & 0x80000000u) | (mt_[0] & 0x7fffffffu);
        mt_[623] = mt_[396] ^ (y >> 1) ^ mag01[y & 0x1u];
        index_ = 0;
    }
    y = mt_[index_++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}

double MtRandom::random() {
    uint32_t a = genrand_uint32() >> 5;
    uint32_t b = genrand_uint32() >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

uint64_t MtRandom::getrandbits(int k) {
    if (k <= 0) return 0;
    if (k <= 32) {
        return genrand_uint32() >> (32 - k);
    }
    int words = (k - 1) / 32 + 1;
    uint32_t w[4] = {0, 0, 0, 0};
    int rem = k;
    for (int i = 0; i < words && i < 4; i++) {
        uint32_t r = genrand_uint32();
        if (rem < 32) r >>= (32 - rem);
        w[i] = r;
        rem -= 32;
    }
    uint64_t result = (uint64_t)w[0];
    for (int i = 1; i < words && i < 4; i++) {
        result |= ((uint64_t)w[i]) << (32 * i);
    }
    return result;
}

uint64_t MtRandom::randbelow(uint64_t n) {
    if (n <= 1) return 0;
    int k = 0;
    for (uint64_t t = n; t; t >>= 1) k++;
    uint64_t r = getrandbits(k);
    while (r >= n) r = getrandbits(k);
    return r;
}

uint64_t MtRandom::randrange(uint64_t n) {
    return randbelow(n);
}

int64_t MtRandom::randint(int64_t a, int64_t b) {
    if (a > b) std::swap(a, b);
    return a + (int64_t)randbelow((uint64_t)(b - a) + 1u);
}

double MtRandom::uniform(double a, double b) {
    return a + (b - a) * random();
}

// =====================================================================
// SeedSequence (numpy 2.5.2 bit_generator.pyx) + PCG64 (XSL RR 128/64)
// =====================================================================

SeedSequence::SeedSequence(uint64_t entropy) {
    uint32_t entropy_arr[2];
    int ent_len;
    if (entropy == 0) {
        entropy_arr[0] = 0;
        ent_len = 1;
    } else {
        entropy_arr[0] = (uint32_t)(entropy & 0xffffffffULL);
        if (entropy >> 32) {
            entropy_arr[1] = (uint32_t)(entropy >> 32);
            ent_len = 2;
        } else {
            ent_len = 1;
        }
    }

    uint32_t hc = detail::SS_INIT_A;
    uint32_t mixer[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; i++) {
        if (i < ent_len) {
            mixer[i] = detail::ss_hashmix(entropy_arr[i], hc);
        } else {
            mixer[i] = detail::ss_hashmix(0, hc);
        }
    }
    for (int i_src = 0; i_src < 4; i_src++) {
        for (int i_dst = 0; i_dst < 4; i_dst++) {
            if (i_src != i_dst) {
                mixer[i_dst] = detail::ss_mix(
                    mixer[i_dst], detail::ss_hashmix(mixer[i_src], hc));
            }
        }
    }
    for (int i = 0; i < 4; i++) pool_[i] = mixer[i];
}

void SeedSequence::generate_state(uint64_t* out, int n_words64) {
    int n = n_words64 * 2;
    uint32_t state[16];
    uint32_t hc = detail::SS_INIT_B;
    for (int i = 0; i < n; i++) {
        uint32_t data = pool_[i % 4];
        data ^= hc;
        hc *= detail::SS_MULT_B;
        data *= hc;
        data ^= data >> detail::SS_XSHIFT;
        state[i] = data;
    }
    for (int i = 0; i < n_words64; i++) {
        out[i] = (uint64_t)state[2 * i] | ((uint64_t)state[2 * i + 1] << 32);
    }
}

PCG64::PCG64(uint64_t seed) {
    SeedSequence ss(seed);
    uint64_t v[4];
    ss.generate_state(v, 4);
    set_seed(v[0], v[1], v[2], v[3]);
    has_uint32_ = 0;
    uinteger_ = 0;
}

void PCG64::set_seed(uint64_t sh, uint64_t sl, uint64_t ih, uint64_t il) {
    state_ = {0, 0};
    inc_.high = (ih << 1) | (il >> 63);
    inc_.low = (il << 1) | 1u;
    step();
    state_ = detail::u64pair_add(state_, detail::U64Pair{sh, sl});
    step();
}

void PCG64::step() {
    static const detail::U64Pair MULT{
        0x2360ed051fc65da4ULL, 0x4385df649fccf645ULL};
    state_ = detail::u64pair_add(
        detail::u64pair_mul(state_, MULT), inc_);
}

uint64_t PCG64::output_xsl_rr() {
    return detail::rotr64(state_.high ^ state_.low,
                          (unsigned int)(state_.high >> 58));
}

uint64_t PCG64::next64() {
    step();
    return output_xsl_rr();
}

uint32_t PCG64::next32() {
    if (has_uint32_) {
        has_uint32_ = 0;
        return uinteger_;
    }
    uint64_t next = next64();
    has_uint32_ = 1;
    uinteger_ = (uint32_t)(next >> 32);
    return (uint32_t)(next & 0xffffffff);
}

double PCG64::next_double() {
    return (double)(next64() >> 11) * (1.0 / 9007199254740992.0);
}

// ---------------------------------------------------------------------
// numpy random_binomial / random_binomial_inversion / random_binomial_btpe
// (numpy 2.5.2 numpy/random/src/distributions/distributions.c)
// ---------------------------------------------------------------------
int64_t PCG64::random_binomial_inversion(int64_t n, double p) {
    double q = 1.0 - p;
    double qn = std::exp((double)n * std::log1p(-p));
    double np = (double)n * p;
    int64_t bound = (int64_t)std::fmin(
        (double)n, np + 10.0 * std::sqrt(np * q + 1.0));
    int64_t X = 0;
    double px = qn;
    double U = next_double();
    while (U > px) {
        X++;
        if (X > bound) {
            X = 0;
            px = qn;
            U = next_double();
        } else {
            U -= px;
            px = ((double)(n - X + 1) * p * px) / ((double)X * q);
        }
    }
    return X;
}

int64_t PCG64::random_binomial_btpe(int64_t n, double p) {
    double r = std::fmin(p, 1.0 - p);
    double q = 1.0 - r;
    double fm = (double)n * r + r;
    int64_t m = (int64_t)std::floor(fm);
    double p1 = std::floor(2.195 * std::sqrt((double)n * r * q) - 4.6 * q) + 0.5;
    double xm = m + 0.5;
    double xl = xm - p1;
    double xr = xm + p1;
    double c = 0.134 + 20.5 / (15.3 + (double)m);
    double a = (fm - xl) / (fm - xl * r);
    double laml = a * (1.0 + a / 2.0);
    a = (xr - fm) / (xr * q);
    double lamr = a * (1.0 + a / 2.0);
    double p2 = p1 * (1.0 + 2.0 * c);
    double p3 = p2 + c / laml;
    double p4 = p3 + c / lamr;

    double nrq = (double)n * r * q;
    int64_t y;
    for (;;) {
        double u = next_double() * p4;
        double v = next_double();
        double x, s, F, t, A, rho, x1, x2, f1, f2, z, z2, w, w2;
        int64_t k, i;

        if (u > p1) goto Step20;
        y = (int64_t)std::floor(xm - p1 * v + u);
        goto Step60;

    Step20:
        if (u > p2) goto Step30;
        x = xl + (u - p1) / c;
        v = v * c + 1.0 - std::fabs((double)m - x + 0.5) / p1;
        if (v > 1.0) continue;
        y = (int64_t)std::floor(x);
        goto Step50;

    Step30:
        if (u > p3) goto Step40;
        y = (int64_t)std::floor(xl + std::log(v) / laml);
        if ((y < 0) || (v == 0.0)) continue;
        v = v * (u - p2) * laml;
        goto Step50;

    Step40:
        y = (int64_t)std::floor(xr - std::log(v) / lamr);
        if ((y > n) || (v == 0.0)) continue;
        v = v * (u - p3) * lamr;

    Step50:
        k = std::llabs(y - m);
        if ((k > 20) && (k < ((nrq) / 2.0 - 1))) goto Step52;

        s = r / q;
        a = s * (n + 1);
        F = 1.0;
        if (m < y) {
            for (i = m + 1; i <= y; i++) {
                F *= (a / (double)i - s);
            }
        } else if (m > y) {
            for (i = y + 1; i <= m; i++) {
                F /= (a / (double)i - s);
            }
        }
        if (v > F) continue;
        goto Step60;

    Step52:
        rho = (k / nrq) *
              ((k * (k / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5);
        t = -k * k / (2 * nrq);
        A = std::log(v);
        if (A < (t - rho)) goto Step60;
        if (A > (t + rho)) continue;

        x1 = (double)y + 1;
        f1 = (double)m + 1;
        z = (double)n + 1 - (double)m;
        w = (double)n - (double)y + 1;
        x2 = x1 * x1;
        f2 = f1 * f1;
        z2 = z * z;
        w2 = w * w;
        if (A > (xm * std::log(f1 / x1) +
                 ((double)n - (double)m + 0.5) * std::log(z / w) +
                 ((double)y - (double)m) * std::log(w * r / (x1 * q)) +
                 (13860. - (462. - (132. - (99. - 140. / f2) / f2) / f2) / f2) /
                     f1 / 166320. +
                 (13860. - (462. - (132. - (99. - 140. / z2) / z2) / z2) / z2) /
                     z / 166320. -
                 (13860. - (462. - (132. - (99. - 140. / x2) / x2) / x2) / x2) /
                     x1 / 166320. -
                 (13860. - (462. - (132. - (99. - 140. / w2) / w2) / w2) / w2) /
                     w / 166320.)) {
            continue;
        }

    Step60:
        if (p > 0.5) y = n - y;
        return y;
    }
}

int64_t PCG64::binomial(int64_t n, double p) {
    if (n == 0LL || p == 0.0) return 0;
    double q;
    if (p <= 0.5) {
        if (p * (double)n <= 30.0) {
            return random_binomial_inversion(n, p);
        } else {
            return random_binomial_btpe(n, p);
        }
    } else {
        q = 1.0 - p;
        if (q * (double)n <= 30.0) {
            return n - random_binomial_inversion(n, q);
        } else {
            return n - random_binomial_btpe(n, q);
        }
    }
}

uint64_t PCG64::randrange(uint64_t n) {
    if (n <= 1) return 0;
    uint64_t threshold = (uint64_t)(-(int64_t)n) % n;
    for (;;) {
        uint64_t r = next64();
        if (r >= threshold) return r % n;
    }
}

int64_t PCG64::randint(int64_t a, int64_t b) {
    if (a > b) std::swap(a, b);
    return a + (int64_t)randrange((uint64_t)(b - a) + 1u);
}

double PCG64::uniform(double a, double b) {
    return a + (b - a) * next_double();
}

}  // namespace colony