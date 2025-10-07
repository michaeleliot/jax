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
from jax._src.util import safe_map as map
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import promote_args_inexact
from jax._src.scipy.special import xlogy, gammaln, gammaincc
from jax._src.typing import Array, ArrayLike


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


def entropy(mu: ArrayLike, loc: ArrayLike = 0) -> Array:
    mu, loc = promote_args_inexact("poisson.entropy", mu, loc)

    mu, loc = jnp.broadcast_arrays(mu, loc)
    mu_flat = jnp.ravel(mu)
    loc_flat = jnp.ravel(loc)
    median = mu_flat + 1/3 - 0.02 / mu_flat
    lb = jnp.asarray(0, dtype=np.float32)
    ub = jnp.asarray(np.inf, dtype=np.float32)

    def body_fn(args):
        mu_i, loc_i, median = args
        return _expect(mu_i, loc_i, lb, ub, median)
    combined = jnp.stack([mu_flat, loc_flat, median], axis=-1)
    result = map(body_fn, combined)

    flat = jnp.stack(result)
    result = jnp.reshape(flat, mu.shape)
    return result

def _expect(mu, loc, lb, ub, x0, inc = 1.0, maxcount=1000.0, chunksize=32.0):
    """JAX version of expectation helper with convergence and maxcount control."""
    x0 = jnp.clip(x0, lb, ub)

    tot_f = _sum_over_chunks(mu, loc, x0, ub + 1, inc, chunksize, maxcount)

    tot_b = _sum_over_chunks(mu, loc, lb, x0, inc, chunksize, maxcount)

    return tot_f + tot_b

def _sum_over_chunks(mu, loc, start, stop, inc, chunksize, max_chunks):
    """
    Split the range [start, stop) into chunks and sum over each chunk
    by calling _sum_over_range.
    """
    chunk_starts = start + chunksize * jnp.arange(max_chunks)
    chunk_stops = jnp.minimum(chunk_starts + chunksize, stop)

    # Function to sum a single chunk
    def sum_chunk(args):
        lb, ub = args
        return _sum_over_range(mu, loc, lb, ub, inc)

    # Map over all chunks
    combined = jnp.stack([chunk_starts, chunk_stops], axis=-1)
    result = map(sum_chunk, combined)
    flat = jnp.stack(result)
    return jnp.sum(flat)

def _sum_over_range(mu, loc, start, stop, inc):
    fun = lambda x: entr(pmf(x, mu, loc))
    xs = jnp.arange(start, stop, inc)
    vals = fun(xs)
    return jnp.sum(vals)



# def safe_arange(lb, ub, inc, chunksize, maxcount):
#     """Return a 2D array of range values split into chunks."""
#     ub = jnp.minimum(ub, maxcount)
#     xs = jnp.arange(lb, ub, inc)
#     n = xs.shape[0]
#     # Compute how many full chunks we can have
#     num_chunks = jnp.ceil(n / chunksize).astype(int)
    
#     # Pad xs so it fits evenly into chunks
#     pad_len = num_chunks * chunksize - n
#     xs = jnp.pad(xs, (0, pad_len), mode="edge")

#     # Reshape into (num_chunks, chunksize)
#     xs_chunked = xs.reshape(num_chunks, chunksize)
#     return xs_chunked


# def _sum_over_range(mu, loc, start, stop, inc, chunksize, tolerance, maxcount):
#     """Sum over range in chunks; each chunk is a subarray of x values."""
#     fun = lambda x: entr(pmf(x, mu, loc))

#     xs_chunked = safe_arange(start, stop, inc, chunksize, maxcount)

#     def chunk_sum(x_chunk):
#         vals = fun(x_chunk)
#         return jnp.sum(vals)

#     chunk_sums = map(chunk_sum, xs_chunked)

#     flat = jnp.stack(chunk_sums)
#     return jnp.sum(flat)

# def _sum_over_range(mu, loc, inc, chunksize, tolerance, maxcount):
#   """Sum fun(x) over [start, stop) in chunks until convergence or maxcount."""
#   fun = lambda x: entr(pmf(x, mu, loc))

#   print(mu, loc)

#   def cond_fun(state):
#     i, tot, stop_flag = state
#     not_done = jnp.logical_and(i < maxcount, jnp.logical_not(stop_flag))
#     return not_done

#   def body_fun(state):
#       i, tot, stop_flag = state
#       i = i.astype(np.int32)
#       print(i, chunksize, i * chunksize, (i + 1) * chunksize)
#       chunk = jnp.arange(i * chunksize, (i + 1) * chunksize, inc)
#       delta = jnp.sum(fun(chunk))
#       stop_flag_new = jnp.abs(delta) < tolerance * chunk.size
#       return (i + 1, tot + delta, stop_flag_new)

#   _, tot, _ = lax.while_loop(cond_fun, body_fun, (0, 0.0, False))
  
#   return tot

# def safe_arange(lb, ub, inc, maxcount):
#     # Replace inf (or large) upper bounds with maxcount
#     ub = jnp.minimum(ub, maxcount)
#     return jnp.arange(lb, ub + inc, inc)



# def entropy(mu: ArrayLike, loc: ArrayLike = 0) -> Array:
#   r"""Poisson distribution entropy.

#   JAX implementation of :obj:`scipy.stats.poisson` ``entropy``.

#   The entropy of a Poisson distribution is given by

#   .. math::

#      H(X) = \mu(1 - \log(\mu)) + e^{-\mu}\sum_{k=0}^{\infty}\frac{\mu^k\log(k!)}{k!}

#   Args:
#     mu: arraylike, distribution shape parameter
#     loc: arraylike, distribution offset parameter

#   Returns:
#     array of entropy values.
#   """
#   mu, loc = promote_args_inexact("poisson.entropy", mu, loc)
#   original_shape = mu.shape
  
#   # Ravel inputs to 1D arrays to be mapped over.
#   print(mu.shape, loc.shape)
#   mu_flat = mu.ravel()
#   loc_flat = jnp.broadcast_to(loc, mu.shape).ravel()

#   # Define a function that computes entropy for a single scalar value.
#   def _scalar_entropy(mu_scalar, loc_scalar):
#     lb = 0
#     ub = np.inf
#     inc = 1
#     # Approximate the median for the scalar mu.
#     median = mu_scalar + 1/3 - 0.02 / mu_scalar
    
#     # This lambda captures the scalar mu and loc.
#     fun = lambda x: entr(pmf(x, mu_scalar, loc_scalar))
    
#     return _expect(fun, lb, ub, median, inc)

#   # Use jax.lax.map to apply the scalar calculation to each element.
#   # This is a JIT-compatible way to perform the mapping.
#   result_flat = map(_scalar_entropy, mu_flat, loc_flat)
  
#   # Reshape the flat result back to the original input shape.
#   return result_flat.reshape(original_shape)


# def _expect(fun, lb, ub, x0, inc, maxcount=1000, tolerance=1e-10, chunksize=32):
#     """JAX version of expectation helper with convergence and maxcount control."""
#     # Bound x0
#     x0 = jnp.clip(x0, lb, ub)

#     # Forward pass: [x0, ub]
#     tot_f, max_f = _sum_over_range(fun, x0, ub + 1, inc, chunksize, tolerance, maxcount)
#     if bool(max_f):
#         warnings.warn("expect(): forward sum did not converge", RuntimeWarning, stacklevel=3)
#         return tot_f

#     # Backward pass: [lb, x0)
#     tot_b, max_b = _sum_over_range(fun, x0 - 1, lb - 1, -inc, chunksize, tolerance, maxcount)
#     if bool(max_b):
#         warnings.warn("expect(): backward sum did not converge", RuntimeWarning, stacklevel=3)

#     return tot_f + tot_b


# def _sum_over_range(fun, start, stop, inc, chunksize, tolerance, maxcount):
#     xs = jnp.arange(start, stop, inc)
#     vals = fun(xs)
#     return jnp.sum(vals)

# def safe_arange_batch(lb, ub, inc, maxcount):
#     # lb, ub, inc are arrays of the same shape (e.g. [N])
#     def body_fn(vals):
#         l, u, i = vals
#         return safe_arange(l, u, i, maxcount)
#     return map(body_fn, (lb, ub, inc))

# def safe_arange(lb, ub, inc, maxcount):
#     # Replace inf (or large) upper bounds with maxcount
#     ub = jnp.minimum(ub, maxcount)
#     return jnp.arange(lb, ub + inc, inc)


# def _sum_over_range(fun, start, stop, inc, chunksize, tolerance, maxcount):
#     """Sum fun(x) over [start, stop) in chunks until convergence or maxcount."""
#     print(start, stop, inc)
#     xs = jnp.arange(start[0], stop[0], inc[0])
#     num_chunks = (xs.size + chunksize - 1) // chunksize

    # def cond_fun(state):
    #     i, tot, stop_flag = state
    #     not_done = jnp.logical_and(i < num_chunks, jnp.logical_not(stop_flag))
    #     return not_done

    # def body_fun(state):
    #     i, tot, stop_flag = state
    #     chunk = xs[i * chunksize : (i + 1) * chunksize]
    #     delta = jnp.sum(fun(chunk))
    #     stop_flag_new = jnp.abs(delta) < tolerance * chunk.size
    #     return (i + 1, tot + delta, stop_flag_new)

    # i_final, tot, _ = lax.while_loop(cond_fun, body_fun, (0, 0.0, False))

#     # compute total processed points and whether maxcount was exceeded
#     total_points = jnp.minimum(i_final * chunksize, xs.size)
#     max_exceeded = total_points > maxcount

#     return tot, max_exceeded






###
# IGAM: https://github.com/jeremybarnes/cephes/blob/60f27df395b8322c2da22c83751a2366b82d50d1/cprob/igam.c
# PDTr: https://github.com/jeremybarnes/cephes/blob/60f27df395b8322c2da22c83751a2366b82d50d1/cprob/pdtr.c#L154
# Poisson Implementation in Scipy: https://github.com/scipy/scipy/blob/v1.16.2/scipy/stats/_discrete_distns.py#L961-L1024
# Generic Entropy Function: https://github.com/scipy/scipy/blob/d46b9b31d3fb71b1e5fcd9bf9fa1e3a0b235c951/scipy/stats/_distn_infrastructure.py#L1261-L1296
# Entropy Function For Discrete Random Variable: https://github.com/scipy/scipy/blob/d46b9b31d3fb71b1e5fcd9bf9fa1e3a0b235c951/scipy/stats/_distn_infrastructure.py#L3846-L3852
# Expect Function For Summation Of Infinite Support: https://github.com/scipy/scipy/blob/d46b9b31d3fb71b1e5fcd9bf9fa1e3a0b235c951/scipy/stats/_distn_infrastructure.py#L3954-L3994
# Scipy has a generic entropy function https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html