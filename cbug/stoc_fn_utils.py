# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for building stoc_fns especially for discrete SCMs."""

from typing import List, Mapping

from cbug import scm
import numpy as np


def create_interaction_mean_stoc_fn(
    scale: float,
    input_domains: List[List[str]],
) -> scm.StocFn:
  """Creates a deterministic stoc_fn for the interaction terms.

  Args:
    scale: The magnitude of the maximum function value.
    input_domains: The variable names of the inputs to each interaction term.

  Returns:
    A stoc_fn that computes the interaction terms.
  """
  def f(**kwargs):
    first_parent = [val for val in kwargs.values()][0]
    if isinstance(first_parent, np.ndarray):
      n_samples = first_parent.shape
    else:
      # Not an array but a single sample.
      n_samples = (1, 1)
    # Compute the non-linear term.
    samples = np.zeros(n_samples)
    for domain in input_domains:
      # For each domain, take a product.
      interaction_term = scale * np.ones(n_samples)
      for var in domain:
        interaction_term *= kwargs[var]
      samples += interaction_term
    return samples
  return f


def create_discrete_to_linear_stoc_fn(
    means: Mapping[str, np.ndarray],
) -> scm.StocFn:
  """Returns a deterministic stoc_fn equal to a linear function of the parents.

  Args:
    means: A dictionary with parent names key and np.arrays of shape (dim_i,)
      values, where means[par][i] specifies the amount to add to the function's
      return values when the parent par has value i.

  Returns:
    Function that generates means for all tuples of values of the parents with
    shape (n_samples,).
  """
  def stoc_fn(**kwargs):
    first_parent = [val for val in kwargs.values()][0]
    if isinstance(first_parent, (int, float, np.integer, np.floating)):
      # Not an array but a single sample.
      n_samples = 1
    else:
      n_samples = len(first_parent)
    samples = np.zeros(n_samples)

    for parent, mean in means.items():
      samples += mean[kwargs[parent]].flatten()
    return samples

  return stoc_fn


def add_gaussian_noise_to_stoc_fn(
    stoc_fn: scm.StocFn, cov: float
) -> scm.StocFn:
  def new_stoc_fn(**kwargs):
    samples = stoc_fn(**kwargs)
    return samples + np.random.normal(0, cov, size=samples.shape)
  return new_stoc_fn


def add_stoc_fns(stoc_fn_list: List[scm.StocFn]) -> scm.StocFn:
  def new_stoc_fn(**kwargs):
    samples = stoc_fn_list[0](**kwargs)
    for stoc_fn in stoc_fn_list[1:]:
      samples += stoc_fn(**kwargs)
    return samples
  return new_stoc_fn


def create_categorical_stoc_fn(
    probs: np.ndarray,
) -> scm.StocFn:
  """Returns a stochastic function for a categorical random variable.

  Args:
    probs: the probability of each outcome
  """

  # Set the function parameters with a closure.
  def stoc_fn(n_samples):
    return np.random.choice(len(probs), size=n_samples, p=probs)

  return stoc_fn


def create_categorical_conditional_stoc_fn(
    probs: np.ndarray,
    parent_names: List[str],
) ->  scm.StocFn:
  """Returns a stoc_fn for a categorical random variable with parents.

  Args:
    probs: A tensor of dimension (parents[0].dim, ..., parents[end].dim, dim).
    parent_names: A list of the names of the parents in the same order.

  Returns: A function from a dictionary of parents to an np.ndarray where value
    of the variable is chosen from the categorical distribution specified in the
    probs tensor using the corresponding value of the parents.  E.g. if the
    parents are {'x0': [2], 'x1': [5], 'x2': [3]}, then the value is sampled as
    a categorical distribution with p=probs[2, 5, 3, :].  If vectors are passed
    as the parents, a vector of samples with the corresponding shape is
    returned.
  """

  def stoc_fn(**kwargs) -> np.ndarray:
    first_parent = [val for val in kwargs.values()][0]
    if isinstance(first_parent, (int, float, np.integer, np.floating)):
      # Not an array but a single sample.
      n_samples = 1
    else:
      n_samples = len(first_parent)
    support_size = probs.shape[-1]
    try:
      parents_matrix = np.vstack(
          [kwargs[var].flatten() for var in parent_names]
      ).transpose()
    except Exception as exc:
      raise ValueError('Parents do not have compatible size.') from exc

    # We need to make parents_matrix integer valued where each column
    # obeys the corresponding support_size constraint.
    upper_bound = np.tile(np.array(probs.shape[:-1]) - 1, (n_samples, 1))
    parents_matrix = np.minimum(
        np.round(np.abs(parents_matrix)), upper_bound
    ).astype(int)

    return np.array(
        [
            np.random.choice(support_size, p=probs[tuple(p_vals)])
            for p_vals in parents_matrix
        ]
    )

  return stoc_fn
