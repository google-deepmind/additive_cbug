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

"""Unit tests for the  algorithms."""

from absl.testing import absltest
from cbug import base
from cbug import scm
from cbug import utils
import numpy as np


class CBUGTest(absltest.TestCase):
  """Unit test for the CBUG algorithm."""

  def test_reduce_actions(self):
    s_marginal = {'X': np.array([0, 1, 2]), 'Z': np.array([1, 2, 3])}
    support_sizes = {'X': 3, 'Z': 4}
    name_to_idx = utils.get_name_to_idx(support_sizes)
    mean_est = np.array([1.9, 2.7, 3, 2, 1.2, 1, 1.8])
    # The gaps are: 1.1, .3, 0, 0, .8, 1, .2.
    gamma = .1
    returned = base.reduce_actions(s_marginal, name_to_idx, mean_est, gamma)
    ans = {
        'X': np.array([2]),
        'Z': np.array([0]),
    }
    self.assertDictEqual(returned, ans)

    gamma = .4
    returned = base.reduce_actions(s_marginal, name_to_idx, mean_est, gamma)
    ans = {
        'X': np.array([1, 2]),
        'Z': np.array([0, 3]),
    }
    self.assertDictEqual(returned, ans)

    gamma = 1
    returned = base.reduce_actions(s_marginal, name_to_idx, mean_est, gamma)
    ans = {
        'X': np.array([1, 2]),
        'Z': np.array([0, 1, 3]),
    }
    self.assertDictEqual(returned, ans)

  def test_xy_optimal_design(self):
    s_marginal = {'X': np.array([0, 1, 2]), 'Z': np.array([0, 1, 2, 3])}
    support_sizes = {'X': 3, 'Z': 4}
    gamma = 1
    delta = .5
    sigma2 = 1.2

    covariates = base.xy_optimal_design(
        s_marginal,
        support_sizes,
        gamma,
        delta,
        sigma2,
    )
    np.testing.assert_allclose(len(covariates['X']), 24)
    _, counts = np.unique(covariates['X'], return_counts=True)
    np.testing.assert_allclose(counts, 8)
    _, counts = np.unique(covariates['Z'], return_counts=True)
    np.testing.assert_allclose(counts, 6)

    s_marginal = {'X': np.array([0, 1]), 'Z': np.array([0, 1, 2])}
    covariates = base.xy_optimal_design(
        s_marginal,
        support_sizes,
        gamma,
        delta,
        sigma2,
    )
    np.testing.assert_allclose(len(covariates['X']), 17)
    _, counts = np.unique(covariates['X'], return_counts=True)
    np.testing.assert_allclose(counts, 8, atol=1)
    _, counts = np.unique(covariates['Z'], return_counts=True)
    np.testing.assert_allclose(counts, 6, atol=1)

  def test_run_modl(self):
    support_sizes = {'X1': 3, 'X2': 4}
    cov_variables = ['X1', 'X2']
    var_names = ['X1', 'X2', 'Y']
    parents = {'X1': [],
               'X2': ['X1'],
               'Y': ['X1', 'X2'],
               }
    strc_fn_probs = {}
    strc_fn_probs['X1'] = np.array([0, 1, 0])
    strc_fn_probs['X2'] = np.array([[0, 1, 0], [1, 0, 0]])
    additive_means = {'X1': np.array([0, 1, 0]),
                      'X2': np.array([0, .2, .4]),
                      }
    support_sizes = {'X1': 3, 'X2': 3}
    model = scm.DiscreteAdditiveSCM(
        'test',
        var_names,
        parents,
        strc_fn_probs,
        additive_means,
        cov_variables,
        'Y',
        support_sizes,
        cov=0,
    )
    results = base.run_modl(delta=.5,
                            epsilon=.1,
                            model=model,
                            sigma2=1,
                            outcome_bound=2,
                            num_parents_bound=None,
                            )

    np.testing.assert_equal(results.n_samples_t, [72, 288, 767])
    np.testing.assert_equal(results.gamma_t, [1, .5, .25])
    np.testing.assert_equal(results.best_action_t[-1], {'X1': [1], 'X2': [2]})
    np.testing.assert_equal(results.s_t[0], {'X1': [0, 1, 2], 'X2': [0, 1, 2]})
    np.testing.assert_equal(results.s_t[-1], {'X1': [1], 'X2': [1, 2]})
