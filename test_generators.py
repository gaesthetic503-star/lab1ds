import numpy as np
from scipy import stats
from generators import (MT19937, discrete_inverse, build_alias, alias_sample,
                        normal_polar, normal_mixture)

PROBS = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
VALS = np.array([10, 20, 30, 40, 50])


def test_mt_reference_value():
    # эталон из статьи авторов: seed 5489, первое число
    assert MT19937(5489).uint32() == 3499211612


def test_mt_matches_numpy():
    mine = MT19937(42).random(1000)
    ref = np.random.RandomState(42).random_sample(1000)
    assert np.array_equal(mine, ref)


def test_reproducible_by_seed():
    assert np.array_equal(MT19937(7).random(100), MT19937(7).random(100))
    assert not np.array_equal(MT19937(7).random(100), MT19937(8).random(100))


def test_uniform_range_and_moments():
    u = MT19937(1).random(50000)
    assert u.min() >= 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 0.01
    assert abs(u.var() - 1 / 12) < 0.005


def test_uniform_chi2_and_autocorr():
    u = MT19937(2).random(50000)
    counts = np.histogram(u, bins=20, range=(0, 1))[0]
    assert stats.chisquare(counts).pvalue > 0.001
    assert abs(np.corrcoef(u[:-1], u[1:])[0, 1]) < 0.02


def test_discrete_inverse_edges():
    assert discrete_inverse(VALS, PROBS, np.array([0.0]))[0] == 10
    assert discrete_inverse(VALS, PROBS, np.array([0.999999]))[0] == 50


def test_discrete_inverse_support_and_freq():
    x = discrete_inverse(VALS, PROBS, MT19937(3).random(100000))
    assert set(np.unique(x)) <= set(VALS)
    obs = np.array([(x == v).sum() for v in VALS])
    assert stats.chisquare(obs, PROBS * len(x)).pvalue > 0.001


def test_alias_tables_sum_to_probs():
    prob, alias = build_alias(PROBS)
    n = len(PROBS)
    rec = np.zeros(n)
    for k in range(n):
        rec[k] += prob[k] / n
        rec[alias[k]] += (1 - prob[k]) / n
    assert np.allclose(rec, PROBS)


def test_alias_freq():
    x = alias_sample(VALS, PROBS, MT19937(4), 50000)
    obs = np.array([(x == v).sum() for v in VALS])
    assert stats.chisquare(obs, PROBS * len(x)).pvalue > 0.001


def test_normal_polar():
    z = normal_polar(MT19937(5), 20000)
    assert len(z) == 20000
    assert abs(z.mean()) < 0.03 and abs(z.std() - 1) < 0.03
    assert stats.kstest(z, "norm").pvalue > 0.001


def test_mixture_ks():
    w, m, s = [0.5, 0.5], [53.0, 77.0], [12.8, 7.6]
    x = normal_mixture(MT19937(6), 20000, w, m, s)
    cdf = lambda t: sum(wi * stats.norm.cdf(t, mi, si) for wi, mi, si in zip(w, m, s))
    assert stats.kstest(x, cdf).pvalue > 0.001


def test_truncated_normal_bounds_and_mean():
    from generators import truncated_normal
    x = truncated_normal(MT19937(8), 20000, 50.0, 10.0, 40.0, 80.0)
    assert x.min() >= 40.0 and x.max() <= 80.0
    assert abs(x.mean() - stats.truncnorm.mean(-1.0, 3.0, 50.0, 10.0)) < 0.3


def test_truncated_mixture_bounds():
    from generators import truncated_mixture
    par = [(77.8, 8.5, 48.0, 91.4), (47.6, 17.3, 32.6, 91.4)]
    x = truncated_mixture(MT19937(9), 20000, [0.5, 0.5], par)
    assert x.min() >= 32.6 and x.max() <= 91.4
