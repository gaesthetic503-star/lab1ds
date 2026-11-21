
import math
import numpy as np


class MT19937:
       N, M = 624, 397
    MATRIX_A = 0x9908B0DF
    UPPER, LOWER = 0x80000000, 0x7FFFFFFF

    def __init__(self, seed=5489):
        self.mt = [0] * self.N
        self.mt[0] = seed & 0xFFFFFFFF
        for i in range(1, self.N):
            prev = self.mt[i - 1]
            self.mt[i] = (1812433253 * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF
        self.idx = self.N

    def _twist(self):
        mt = self.mt
        for i in range(self.N):
            y = (mt[i] & self.UPPER) | (mt[(i + 1) % self.N] & self.LOWER)
            mt[i] = mt[(i + self.M) % self.N] ^ (y >> 1) ^ (self.MATRIX_A if y & 1 else 0)
        self.idx = 0

    def uint32(self):
        if self.idx >= self.N:
            self._twist()
        y = self.mt[self.idx]
        self.idx += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF

    def random(self, n=None):
        """U[0,1) с 53 битами: (a*2^26 + b) / 2^53, a - 27 бит, b - 26 бит."""
        if n is None:
            return self._one()
        return np.array([self._one() for _ in range(n)])

    def _one(self):
        a = self.uint32() >> 5
        b = self.uint32() >> 6
        return (a * 67108864.0 + b) / 9007199254740992.0


def discrete_inverse(values, probs, u):
    
    values = np.asarray(values)
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0
    idx = np.searchsorted(cdf, u, side="left")
    return values[idx]


def build_alias(probs):
        n = len(probs)
    scaled = np.asarray(probs, dtype=float) * n
    prob = np.zeros(n)
    alias = np.zeros(n, dtype=int)
    small = [i for i in range(n) if scaled[i] < 1.0]
    large = [i for i in range(n) if scaled[i] >= 1.0]
    while small and large:
        s, l = small.pop(), large.pop()
        prob[s], alias[s] = scaled[s], l
        scaled[l] = scaled[l] + scaled[s] - 1.0
        (small if scaled[l] < 1.0 else large).append(l)
    for i in small + large:
        prob[i] = 1.0
    return prob, alias


def alias_sample(values, probs, rng, size):
        values = np.asarray(values)
    prob, alias = build_alias(probs)
    n = len(values)
    out = np.empty(size, dtype=values.dtype)
    for j in range(size):
        k = int(rng.random() * n)
        out[j] = values[k] if rng.random() < prob[k] else values[alias[k]]
    return out


def normal_polar(rng, size):
        out = np.empty(size)
    j = 0
    while j < size:
        v1 = 2.0 * rng.random() - 1.0
        v2 = 2.0 * rng.random() - 1.0
        s = v1 * v1 + v2 * v2
        if 0.0 < s < 1.0:
            f = math.sqrt(-2.0 * math.log(s) / s)
            out[j] = v1 * f
            j += 1
            if j < size:
                out[j] = v2 * f
                j += 1
    return out


def normal_mixture(rng, size, weights, means, stds):
    
    weights = np.asarray(weights, dtype=float)
    u = rng.random(size)
    comp = discrete_inverse(np.arange(len(weights)), weights, u)
    z = normal_polar(rng, size)
    return np.asarray(means)[comp] + np.asarray(stds)[comp] * z


def truncated_normal(rng, size, mu, sd, lo, hi):
    
    got, have = [], 0
    while have < size:
        z = mu + sd * normal_polar(rng, size)
        z = z[(z >= lo) & (z <= hi)]
        got.append(z)
        have += len(z)
    return np.concatenate(got)[:size]


def fit_truncated_normal(x, lo, hi):
        from scipy import optimize, stats

    def nll(p):
        mu, sd = p[0], math.exp(p[1])
        return -stats.truncnorm.logpdf(x, (lo - mu) / sd, (hi - mu) / sd, mu, sd).sum()

    r = optimize.minimize(nll, [np.mean(x), math.log(np.std(x))], method="Nelder-Mead")
    return r.x[0], math.exp(r.x[1])


def truncated_mixture(rng, size, weights, params):
        comp = discrete_inverse(np.arange(len(weights)), np.asarray(weights, float), rng.random(size))
    out = np.empty(size)
    for k, (mu, sd, lo, hi) in enumerate(params):
        idx = np.where(comp == k)[0]
        if len(idx):
            out[idx] = truncated_normal(rng, len(idx), mu, sd, lo, hi)
    return out
