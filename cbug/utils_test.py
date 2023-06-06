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

"""A unit test for the utils implementation."""

import inspect

from absl.testing import absltest
from cbug import scm
from cbug import utils
import numpy as np


class UtilsTest(absltest.TestCase):
  """Unit test for the CBUG utilities."""

  def test_name_to_idx(self):
    support_sizes = {
        'X1': 2,
        'X2': 3,
        'X3': 4,
    }
    name_to_idx = utils.get_name_to_idx(support_sizes)
    answer = {
        'X1': [0, 1],
        'X2': [2, 3, 4],
        'X3': [5, 6, 7],
    }
    self.assertDictEqual(name_to_idx, answer)

  def test_generate_beta_cpd(self):
    support_sizes = {'X': 2, 'Y': 3}
    parents = {'X': [], 'Y': ['X']}
    probs = utils.generate_beta_cpd(
        support_sizes,
        parents,
        'Y',
    )
    assert len(probs) == 2
    for p in probs:
      assert sum(p) == 1
      assert len(p) == 3

  def test_sample_discrete_scm_graph(self):
    np.random.seed(2)  # Y will have children with this seed.
    model = utils.sample_discrete_scm(num_var=2, degree=2, num_parents=2)
    correct_parents = {
        'X0': [],
        'X1': ['X0'],
        'X2': ['X0', 'X1', 'Y'],
        'Y': ['X1', 'X0'],
    }
    self.assertDictEqual(model.parents, correct_parents)
    self.assertListEqual(['X0', 'X1', 'Y', 'X2'], model.var_names)

  def test_sample_discrete_scm_signatures(self):
    np.random.seed(2)  # Y will have children with this seed
    model = utils.sample_discrete_scm(num_var=2, degree=2, num_parents=2)

    # Check Signature for X0.
    args = inspect.signature(model.stoc_fns['X0']).parameters.keys()
    self.assertListEqual(args, [scm.N_SAMPLES_NAME])

    # Check Signature for X1.
    args = inspect.signature(model.stoc_fns['X1']).parameters.keys()
    self.assertListEqual(args, ['kwargs'])
    assert model.stoc_fns['X1'](X0=np.array([0, 1])).shape == (2,)
    self.assertRaises(KeyError, model.stoc_fns['X1'](X1=0))  # Wrong parent.

    # Check Signature for X2.
    args = inspect.signature(model.stoc_fns['X2']).parameters.keys()
    self.assertListEqual(args, ['kwargs'])
    parent_values = {'X0': np.array([0, 1]),
                     'X1': np.array([0, 1]),
                     'Y': np.array([0, 1]),
                     }
    assert model.stoc_fns['X2'](**parent_values).shape == (2,)
    self.assertRaises(KeyError, model.stoc_fns['X1'](Z=0))  # wrong parent

    # Check Signature for Y.
    args = inspect.signature(model.stoc_fns['Y']).parameters.keys()
    self.assertListEqual(args, ['kwargs'])
    parent_values = {'X0': np.array([0, 1]),
                     'X1': np.array([0, 1]),
                     }
    assert model.stoc_fns['Y'](**parent_values).shape == (2,)
    self.assertRaises(KeyError, model.stoc_fns['Y'](X2=0))  # Wrong parent.

  def test_sample_discrete_scm_sample_shapes(self):
    np.random.seed(2)  # Y will have children with this seed.
    model = utils.sample_discrete_scm(num_var=2, degree=2, num_parents=2)

    # Check shape of the stoc_fns.
    n_samples = 3
    samples = model.sample(n_samples)
    assert samples['X0'].shape == (n_samples,)
    assert samples['X1'].shape == (n_samples,)
    assert samples['X2'].shape == (n_samples,)
    assert samples['Y'].shape == (n_samples,)

  def test_interval_intersection(self):
    returned = utils.interval_intersection((0, 1), (1, 2))
    np.testing.assert_allclose(returned, (1, 1))

    returned = utils.interval_intersection((1, 3), (-2, 1))
    np.testing.assert_allclose(returned, (1, 1))

    returned = utils.interval_intersection((0, 1), (0, 2))
    np.testing.assert_allclose(returned, (0, 1))

    returned = utils.interval_intersection((0, 1), (2, 3))
    assert returned is None

    returned = utils.interval_intersection((0, 1), (2, 3))
    assert returned is None

  def test_one_hot_encoding(self):
    covariate_values = {'x': np.array([1, 2, 3]), 'y': np.array([0, 1, 2])}
    support_sizes = {'x': 4, 'y': 3}
    variable_names = ['x', 'y']
    results = utils.form_data_matrix(
        covariate_values, support_sizes, variable_names
    )
    answers = np.array([[0, 1, 0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0, 0, 1]])
    np.testing.assert_array_equal(answers, results)

    results = utils.form_data_matrix(
        covariate_values, support_sizes, ['y', 'x']
    )
    answers = np.array([[1, 0, 0, 0, 1, 0, 0],
                        [0, 1, 0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0, 0, 1]])
    np.testing.assert_array_equal(answers, results)
