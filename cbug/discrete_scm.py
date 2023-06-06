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

"""Implements a Discrete Structural Causal Model (SCM).

Builds on scm.SCM class.
This SCM assumes all variables, except for the outcome variable, are discrete.
As such, it includes a few extra attributes on top of scm.SCM
  - outcome_variable: The real-valued variable that we are optimizing.
  - support_size: Mapping[str, int] specifies the number of distinct values all
    the non-outcome variables can take.
  - best_action: stores the intervention that induces the highest expected value
    of the outcome variable.
"""

from typing import Callable, Mapping, Optional, Union

from cbug import scm


class DiscreteSCM(scm.SCM):
  """Implements a discrete SCM using stochastic functions.

  All variables, except for the outcome_variable, are assumed to take on
  discrete values.

  Attributes:
    var_names: List[str] All variable names sorted to be in topological order.
    parents: Mapping[str, List[str]] parents[var] is a list of all names of all
      the parents of the variable var. If var has no parents, parents[var] is an
      empty list.
    stoc_fns: A dictionary of Callable object, where evaluating
      stoc_fns[var](**parent_dict) will provide the values of the variable var.
      Recall that this function need not be deterministic.
    outcome_variable: the real-valued variable that we are optimizing.
    support_size: A dictionary specifying the number of distinct values all
      the non-outcome variables can take.
    best_action: A mapping from variable names to a value for each variable that
      corresponds to the intervention that results in the highest expected value
      of the outcome variable.
    best_action_value: The expected value of the best action.
    outcome_expected_value_fn: A function that returns the expected value of the
      stoc_fn of the outcome_variable.
  """

  def __init__(self,
               stoc_fns: Mapping[str, Union[scm.StocFnRecipe, scm.StocFn]],
               support_sizes: Mapping[str, int],
               outcome_variable: str,
               best_action: Optional[Mapping[str, int]] = None,
               best_action_value: Optional[float] = None,
               outcome_expected_value_fn:
               Optional[Callable[[Mapping[str, int]], float]] = None,
               ):

    super().__init__(stoc_fns)
    if support_sizes is None:
      self._support_sizes = {}
    else:
      self._support_sizes = support_sizes
    self._outcome_variable = outcome_variable
    self.best_action = best_action
    self.best_action_value = best_action_value
    self.outcome_expected_value_fn = outcome_expected_value_fn

  @property
  def support_sizes(self) -> Mapping[str, int]:
    """Number of distinct values for each variable."""
    return self._support_sizes

  @property
  def outcome_variable(self) -> str:
    return self._outcome_variable
