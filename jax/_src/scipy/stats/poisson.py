# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from jax._src import lax
from jax._src import numpy as jnp
from jax._src.scipy.special import entr
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import promote_args_inexact
from jax._src.scipy.special import xlogy, gammaln, gammaincc
from jax._src.typing import Array, ArrayLike
import warnings



def logpmf(k: ArrayLike, mu: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Poisson log probability mass function.

  JAX implementation of :obj:`scipy.stats.poisson` ``logpmf``.

  The Poisson probability mass function is given by

  .. math::

     f(k) = e^{-\mu}\frac{\mu^k}{k!}

  and is defined for :math:`k \ge 0` and :math:`\mu \ge 0`.

  Args:
    k: arraylike, value at which to evaluate the PMF
    mu: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of logpmf values.

  See Also:
    - :func:`jax.scipy.stats.poisson.cdf`
    - :func:`jax.scipy.stats.poisson.pmf`
  """
  k, mu, loc = promote_args_inexact("poisson.logpmf", k, mu, loc)
  zero = _lax_const(k, 0)
  x = lax.sub(k, loc)
  log_probs = xlogy(x, mu) - gammaln(x + 1) - mu
  return jnp.where(jnp.logical_or(lax.lt(x, zero),
                                  lax.ne(jnp.round(k), k)), -np.inf, log_probs)


def pmf(k: ArrayLike, mu: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Poisson probability mass function.

  JAX implementation of :obj:`scipy.stats.poisson` ``pmf``.

  The Poisson probability mass function is given by

  .. math::

     f(k) = e^{-\mu}\frac{\mu^k}{k!}

  and is defined for :math:`k \ge 0` and :math:`\mu \ge 0`.

  Args:
    k: arraylike, value at which to evaluate the PMF
    mu: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of pmf values.

  See Also:
    - :func:`jax.scipy.stats.poisson.cdf`
    - :func:`jax.scipy.stats.poisson.logpmf`
  """
  return jnp.exp(logpmf(k, mu, loc))


def cdf(k: ArrayLike, mu: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Poisson cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.poisson` ``cdf``.

  The cumulative distribution function is defined as:

  .. math::

     f_{cdf}(k, p) = \sum_{i=0}^k f_{pmf}(k, p)

  where :math:`f_{pmf}(k, p)` is the probability mass function
  :func:`jax.scipy.stats.poisson.pmf`.

  Args:
    k: arraylike, value at which to evaluate the CDF
    mu: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of cdf values.

  See Also:
    - :func:`jax.scipy.stats.poisson.pmf`
    - :func:`jax.scipy.stats.poisson.logpmf`
  """
  k, mu, loc = promote_args_inexact("poisson.logpmf", k, mu, loc)
  zero = _lax_const(k, 0)
  x = lax.sub(k, loc)
  p = gammaincc(jnp.floor(1 + x), mu)
  return jnp.where(lax.lt(x, zero), zero, p)

def entropy(mu, loc=0) -> Array:
  r"""Poisson distribution entropy.

  JAX implementation of :obj:`scipy.stats.poisson` ``entropy``.

  The entropy of a Poisson distribution is given by

  .. math::

     H(X) = \mu(1 - \log(\mu)) + e^{-\mu}\sum_{k=0}^{\infty}\frac{\mu^k\log(k!)}{k!}

  Args:
    mu: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of entropy values.

  See Also:
    - :func:`jax.scipy.stats.poisson.pmf`
    - :func:`jax.scipy.stats.poisson.logpmf`
    - :func:`jax.scipy.stats.poisson.cdf`
  """
  
  mu, loc = promote_args_inexact("poisson.entropy", mu, loc)
  lb, ub = 0, jnp.inf
  median = mu + 1/3 - 0.02/mu # SciPy uses the probability point function to compute the 0.5 quantile. This value is instead an approximation from wikipedia https://en.m.wikipedia.org/wiki/Poisson_distribution
  return _expect(lambda x: entr(pmf(x, mu, loc)),
                           lb, ub, median, 1)


def _sum_over_range(fun, start, stop, inc, chunksize, tolerance, maxcount):
    """Sum fun(x) over [start, stop) in chunks until convergence or maxcount."""
    xs = jnp.arange(start, stop, inc)
    num_chunks = (xs.size + chunksize - 1) // chunksize

    def cond_fun(state):
        i, tot, stop_flag = state
        not_done = jnp.logical_and(i < num_chunks, jnp.logical_not(stop_flag))
        return not_done

    def body_fun(state):
        i, tot, stop_flag = state
        chunk = xs[i * chunksize : (i + 1) * chunksize]
        delta = jnp.sum(fun(chunk))
        stop_flag_new = jnp.abs(delta) < tolerance * chunk.size
        return (i + 1, tot + delta, stop_flag_new)

    i_final, tot, _ = lax.while_loop(cond_fun, body_fun, (0, 0.0, False))

    # compute total processed points and whether maxcount was exceeded
    total_points = jnp.minimum(i_final * chunksize, xs.size)
    max_exceeded = total_points > maxcount

    return tot, max_exceeded


def _expect(fun, lb, ub, x0, inc, maxcount=1000, tolerance=1e-10, chunksize=32):
    """JAX version of expectation helper with convergence and maxcount control."""
    # Handle short support case directly
    if (ub - lb) <= chunksize:
        supp = jnp.arange(lb, ub + 1, inc)
        return jnp.sum(fun(supp))

    # Bound x0
    x0 = jnp.clip(x0, lb, ub)

    # Forward pass: [x0, ub]
    tot_f, max_f = _sum_over_range(fun, x0, ub + 1, inc, chunksize, tolerance, maxcount)
    if bool(max_f):
        warnings.warn("expect(): forward sum did not converge", RuntimeWarning, stacklevel=3)
        return tot_f

    # Backward pass: [lb, x0)
    tot_b, max_b = _sum_over_range(fun, x0 - 1, lb - 1, -inc, chunksize, tolerance, maxcount)
    if bool(max_b):
        warnings.warn("expect(): backward sum did not converge", RuntimeWarning, stacklevel=3)

    return tot_f + tot_b



###
# IGAM: https://github.com/jeremybarnes/cephes/blob/60f27df395b8322c2da22c83751a2366b82d50d1/cprob/igam.c
# PDTr: https://github.com/jeremybarnes/cephes/blob/60f27df395b8322c2da22c83751a2366b82d50d1/cprob/pdtr.c#L154
# Poisson Implementation in Scipy: https://github.com/scipy/scipy/blob/v1.16.2/scipy/stats/_discrete_distns.py#L961-L1024
# Generic Entropy Function: https://github.com/scipy/scipy/blob/d46b9b31d3fb71b1e5fcd9bf9fa1e3a0b235c951/scipy/stats/_distn_infrastructure.py#L1261-L1296
# Entropy Function For Discrete Random Variable: https://github.com/scipy/scipy/blob/d46b9b31d3fb71b1e5fcd9bf9fa1e3a0b235c951/scipy/stats/_distn_infrastructure.py#L3846-L3852
# Expect Function For Summation Of Infinite Support: https://github.com/scipy/scipy/blob/d46b9b31d3fb71b1e5fcd9bf9fa1e3a0b235c951/scipy/stats/_distn_infrastructure.py#L3954-L3994
