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

"""Utilities for cbug experiments."""

import itertools
from typing import List, Mapping, Tuple, Union

import numpy as np


def form_data_matrix(
    values: Mapping[str, np.ndarray],
    support_sizes: Mapping[str, int],
    variable_names: list[str],
) -> np.ndarray:
  """Returns covariates with one-hot encoding.

  Specifically, a dictionary of covariate_values, where covariate_values[var] is
  a np-array of values corresponding to the covariate.

  This function returns a matrix where the first row corresponds to
    (covariate_values[variable_names[0]][0], ...,
        covariate_values[variable_names[-1]][0])
  and the last row corresponds to
    (covariate_values[variable_names[0]][-1], ...,
        covariate_values[variable_names[-1]][-1]),
  and each row is transformed to one-hot encoding. The ith row is mapped to
    e_{x_0}, ..., e_{x_k}
  where x_i = covariate_values[variable_names[i]][j], and e_i is zero except
  for a 1 in the i'th place.

  Args:
    values: A np.ndarray of ints of size (n_samples, n_vars). Each row
      specifies a datapoint with values (x0, x1, ..., x_k).
    support_sizes: Dictionary with variable name keys and support size values.
      The support size specified the number of values the corresponding variable
      can take, assumed to be (0, ..., n-1).
    variable_names: The names, and order, of the variables to include.

  Returns:
    A np.ndarray of shape (n_samples, sum(support_size_list)), where each
    row in data has been one-hot-encoded separately.
  """
  one_hot_vector = []

  for var in variable_names:
    one_hot_vector.append(
        np.eye(support_sizes[var])[values[var]]
    )

  return np.hstack(one_hot_vector)


def get_name_to_idx(
    support_sizes: Mapping[str, int],
) -> Mapping[str, List[int]]:
  """Given a support size map, returns the indices in a one-hot encoding.

  If we transform a series of data using one-hot encoding, such as in the
  form_data_matrix function, each variable gets mapped to a certain range of
  rows. This function returns a dictionary where each value specifies the
  indices corresponding to the key variable.

  Args:
    support_sizes: Dictionary specifying the support size of each variable.

  Returns:
    A dictionary with a list of all the indices corresponding to each variable.
  """
  name_to_idx = {}
  start_idx = 0
  for var, support_size in support_sizes.items():
    name_to_idx[var] = np.arange(
        int(start_idx), int(start_idx + support_size))
    start_idx += support_size

  return name_to_idx


def interval_intersection(
    int1: Tuple[float, float],
    int2: Tuple[float, float],
) -> Union[Tuple[float, float], None]:
  """Finds the intersection of two real-valued intervals.

  Args:
   int1: Tuple of floats: the first interval.
   int2: Tuple of floats: the second interval.

  Returns:
    A tuple of floats indicating the interection of the intervals
    or None if it is empty. Intervals are assumed to be closed.
  """
  if int1[1] < int2[0] or int1[0] > int2[1]:
    return None
  else:
    return (max(int1[0], int2[0]), min(int1[1], int2[1]))


def get_full_action_set(
    support_sizes: Mapping[str, int],
    outcome_variable: str = 'Y',
) -> List[Mapping[str, int]]:
  """Returns the product actions set."""
  # Don't include the outcome variable in the action set.
  sizes = [
      list(range(support_sizes[var]))
      for var in support_sizes
      if var != outcome_variable
  ]
  cartesian_product = itertools.product(*sizes)
  actions = []
  # Translate tuples of ints into a dictionary.
  for action in cartesian_product:
    actions.append({var: i for var, i in zip(support_sizes.keys(), action)})
  return actions

