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

"""Code to run all algorithms in the paper on a SCM."""

from typing import Any, List, Mapping, Optional, Union

from cbug import base
from cbug import discrete_scm_utils as scm_utils
import numpy as np


def single_experiment(
    scm_params: Mapping[str, Union[int, float, List[int]]],
    scm_starting_seed: int = 0,
    alg_starting_seed: int = 0,
    num_scms: int = 1,
    num_seeds: int = 1,
    epsilon: float = .5,
    delta: float = 0.1,
    include_se: bool = False,
    num_parents_bound: Optional[int] = None,
    known_num_parents: bool = False,
) -> Mapping[str, Any]:
  """Returns the average sample complexity of several algorithms.

  Algorithms included:
  MODL
  parents-first: learn the parents then run MODL
  oracle: run MODL with the true parents given
  bandit: run a bandit ignoring the causal structure.

  Args:
    scm_params: Dictionary of parameters to generate the scm.
    scm_starting_seed: Seed to generate the scm using utils.sample_discrete_scm.
    alg_starting_seed: Seed to run the algorithm.
    num_scms: The number of different scms.
    num_seeds: The number of different runs of the algorithm.
    epsilon: Error tolerance.
    delta: Error probability.
    include_se: Whether to include the Successive Elimination baseline (needs
      very few, <8 nodes to be able to run).
    num_parents_bound: An upper bound on the number of parents. None indicates
      no bound.
    known_num_parents: Whether the true number of parents is given to the
      algorithm.

  The scm_params dictionary should include:
    num_variables: Number of variables.
    num_parents: Number of parents of Y.
    degree: Erdos-Renyi degree.
    mean_bound: Scale of linear coefficients.
    cov: Covariance of noise for Y.
    interaction_magnitude: float > 0 and typically < 1. The amount of model
      mispecification.
    interactions: A list of sizes of interaction. Each element is this list
      will be turned into an interaction term that is the multiple of the
      element number of parents * interaction_magnitude. Only relevant when
      interaction_magnitude != 0. The stoc_fn of y is set to be this interaction
      function.
    alpha: Y values are  sampled from a beta(alpha, beta) distribution.
    beta: Y values are  sampled from a beta(alpha, beta) distribution.
    support_size_min: Lower bound on support size for each variable.
    support_size_max: Upper bound on support size for each variable.

  Returns:
    Dictionary of: number of samples, final value, for each algorithm
  """
  assert scm_params["num_parents"] <= scm_params["num_variables"]

  results = {
      "num_variables": scm_params["num_variables"],
      "num_parents": scm_params["num_parents"],
      "degree": scm_params["degree"],
      "epsilon": epsilon,
      "delta": delta,
      "mean_bound": scm_params["mean_bound"],
      "cov": scm_params["cov"],
  }

  results["oracle_samples"] = []
  results["MODL_samples"] = []
  results["parents_first_samples"] = []
  results["se_samples"] = []

  results["oracle_value"] = []
  results["MODL_value"] = []
  results["parents_first_value"] = []
  results["se_value"] = []

  results["oracle_gap"] = []
  results["MODL_gap"] = []
  results["parents_first_gap"] = []
  results["se_gap"] = []

  results["true_value"] = []

  results["instance"] = []
  results["seed"] = []

  outcome_bound = scm_params["mean_bound"] * scm_params["num_variables"]

  for instance in range(num_scms):
    scm_seed = instance + scm_starting_seed
    np.random.seed(scm_seed)
    model = scm_utils.sample_discrete_additive_scm(**scm_params)
    if known_num_parents:
      modl_parents_bound = len(model.parents["Y"])
    else:
      modl_parents_bound = num_parents_bound

    for alg_seed in range(num_seeds):
      results["instance"].append(scm_seed)
      results["true_value"].append(model.best_action_value)
      results["seed"].append(alg_seed + alg_starting_seed)
      np.random.seed(alg_seed + alg_starting_seed)

      # Run the MODL algorithm.
      modl_results = base.run_modl(
          delta=delta,
          epsilon=epsilon,
          model=model,
          cov=scm_params["cov"],
          outcome_bound=outcome_bound,
          num_parents_bound=modl_parents_bound,
      )
      results["MODL_samples"].append(sum(modl_results.n_samples_t))
      results["MODL_value"].append(
          scm_utils.get_expected_value_of_outcome(
              model, modl_results.best_action_t[-1]
          )
      )
      results["MODL_gap"].append(
          model.best_action_value - results["MODL_value"][-1]
      )

      # Run the algorithm with knowledge of the parents.
      oracle_results = base.run_modl(
          delta=delta,
          epsilon=epsilon,
          model=model,
          cov=scm_params["cov"],
          outcome_bound=outcome_bound,
          opt_scope=model.parents["Y"],
      )
      results["oracle_samples"].append(sum(oracle_results.n_samples_t))
      results["oracle_value"].append(
          scm_utils.get_expected_value_of_outcome(
              model, oracle_results.best_action_t[-1]
          )
      )
      results["oracle_gap"].append(
          model.best_action_value - results["oracle_value"][-1]
      )

      # Run the parents-first algorithm.
      (parents_hat, parents_first_samples) = base.find_parents(
          target_var="Y",
          delta=delta / 2,
          epsilon=epsilon / 2,
          model=model,
          cov=scm_params["cov"],
          num_parents_bound=num_parents_bound,
      )
      if not parents_hat:  # Find_parents failed.
        parents_hat = model.var_names.copy()
        parents_hat.remove("Y")

      parents_first_results = base.run_modl(
          delta=delta / 2,
          epsilon=epsilon,
          model=model,
          cov=scm_params["cov"],
          outcome_bound=outcome_bound,
          opt_scope=parents_hat,
          num_parents_bound=num_parents_bound,
      )
      parents_first_samples += sum(parents_first_results.n_samples_t)
      results["parents_first_samples"].append(parents_first_samples)
      results["parents_first_value"].append(
          scm_utils.get_expected_value_of_outcome(
              model, parents_first_results.best_action_t[-1]
          )
      )
      results["parents_first_gap"].append(
          model.best_action_value - results["parents_first_value"][-1]
      )

      if include_se:
        se_results = base.run_se(
            delta=delta / 2,
            epsilon=epsilon,
            model=model,
            cov=scm_params["cov"],
            outcome_bound=outcome_bound,
        )
        results["se_samples"].append(se_results.n_samples_t[-1])
        results["se_value"].append(
            scm_utils.get_expected_value_of_outcome(
                model, se_results.best_action_t[-1]
            )
        )
        results["se_gap"].append(
            model.best_action_value - results["se_value"][-1]
        )
      else:
        results["se_samples"].append(np.nan)
        results["se_value"].append(np.nan)
        results["se_gap"].append(np.nan)

  results["MODL_mean_samples"] = np.mean(results["MODL_samples"])
  results["oracle_mean_samples"] = np.mean(results["oracle_samples"])
  results["parents_first_mean_samples"] = np.mean(
      results["parents_first_samples"]
  )
  results["se_mean_samples"] = np.mean(results["se_samples"])

  results["oracle_mean_gap"] = np.mean(results["oracle_gap"])
  results["MODL_mean_gap"] = np.mean(results["MODL_gap"])
  results["parents_first_mean_gap"] = np.mean(results["parents_first_gap"])
  results["se_mean_gap"] = np.mean(results["se_gap"])

  return results


def sweep(
    scm_params: Mapping[str, Union[int, float, List[int]]],
    parameter_name: str,
    parameter_values: np.array,
    num_parents_bound: Optional[bool] = None,
    known_num_parents: bool = False,
    include_se: bool = False,
    num_scms: int = 20,
    num_seeds: int = 5,
) -> List[Mapping[str, Any]]:
  """Runs algorithms across a sweep of parameter_values.

  Args:
    scm_params: A dictionary of scm parameters.
    parameter_name: The name of the parameter to sweep over. Must be a parameter
      in scm_params.
    parameter_values: The values to sweep over.
    num_parents_bound: An optional upper bound on the number of parents provided
      to the algorithms.
    known_num_parents: Whether the algorithms are given the number of parents.
    include_se: Whether to run the Successive Elimination algorithm; it becomes
      intractable for more than ~7 variables.
    num_scms: The number of scms to sample.
    num_seeds: the number of reruns of the algorithms for each scm.

  Returns:
    A results dictionary for every value in parameter_values.
  """
  results = []
  for value in parameter_values:
    scm_params[parameter_name] = value
    results.append(single_experiment(
        scm_params,
        num_scms=num_scms,
        num_seeds=num_seeds,
        epsilon=.5,
        delta=.1,
        include_se=include_se,
        num_parents_bound=num_parents_bound,
        known_num_parents=known_num_parents,
    ))
  return results

