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

"""Tests for scm."""

from absl.testing import absltest
from cbug import scm
import numpy as np


class SCMTest(absltest.TestCase):

  def test_scm(self):
    def x3_fn(x1, x2):
      return x1 + x2

    stoc_fns = {
        'x0': lambda n_samples: np.random.binomial(1, 0.5, size=n_samples),
        'x1': lambda n_samples: np.random.binomial(1, 0.25, size=n_samples),
        'x2': lambda x0, x1: x1 + x0,
        'x3': scm.StocFnRecipe(x3_fn, ['x1', 'x2']),
    }
    model = scm.SCM(stoc_fns)

    parents = {
        'x0': [],
        'x1': [],
        'x2': ['x0', 'x1'],
        'x3': ['x1', 'x2'],
    }

    np.testing.assert_equal(model.parents, parents)
    np.testing.assert_equal(model.var_names, ['x0', 'x1', 'x2', 'x3'])

  def test_pure_sample(self):
    def x3_fn(x1, x2):
      return x1 * x2
    stoc_fns = {
        'x0': lambda n_samples: np.random.binomial(1, 0.5, size=n_samples),
        'x1': lambda n_samples: np.random.binomial(1, 0.25, size=n_samples),
        'x2': lambda x0, x1: x1 + x0,
        'x3': scm.StocFnRecipe(x3_fn, ['x1', 'x2']),
    }
    model = scm.SCM(stoc_fns)

    np.random.seed(3)
    samples = model.sample(4)
    answer = {
        'x0': [1, 1, 0, 1],
        'x1': [1, 1, 0, 0],
        'x2': [2, 2, 0, 1],
        'x3': [2, 2, 0, 0],
    }

    np.testing.assert_equal(samples, answer)

  def test_pure_sample2(self):
    def x3_fn(**kwargs):
      return sum(list(kwargs.values()))

    stoc_fns = {
        'x0': lambda n_samples: np.random.binomial(1, 0.5, size=n_samples),
        'x1': lambda n_samples: np.random.binomial(1, 0.25, size=n_samples),
        'x2': lambda x0, x1: x1 + x0,
        'x3': scm.StocFnRecipe(x3_fn, ['x0', 'x1', 'x2']),
    }
    model = scm.SCM(stoc_fns)

    np.random.seed(3)
    samples = model.sample(4)
    answer = {
        'x0': [1, 1, 0, 1],
        'x1': [1, 1, 0, 0],
        'x2': [2, 2, 0, 1],
        'x3': [4, 4, 0, 2],
    }

    np.testing.assert_equal(samples, answer)

  def test_sample_with_intervention(self):
    def x3_fn(x1, x2):
      return x1 * x2

    stoc_fns = {
        'x0': lambda n_samples: np.random.binomial(1, 0.5, size=n_samples),
        'x1': lambda n_samples: np.random.binomial(1, 0.25, size=n_samples),
        'x2': lambda x0, x1: x1 + x0,
        'x3': scm.StocFnRecipe(x3_fn, ['x1', 'x2']),
    }
    model = scm.SCM(stoc_fns)

    np.random.seed(4)
    samples = model.sample(4, intervention={'x0': 1})
    answer = {
        'x0': [1, 1, 1, 1],
        'x1': [1, 0, 1, 0],
        'x2': [2, 1, 2, 1],
        'x3': [2, 0, 2, 0],
    }
    np.testing.assert_equal(samples, answer)

    np.random.seed(3)
    samples = model.sample(4, intervention={'x1': 2})
    answer = {
        'x0': [1, 1, 0, 1],
        'x1': [2, 2, 2, 2],
        'x2': [3, 3, 2, 3],
        'x3': [6, 6, 4, 6],
    }
    np.testing.assert_equal(samples, answer)

    np.random.seed(4)
    samples = model.sample(4, intervention={'x0': 0, 'x2': 2})
    answer = {
        'x0': [0, 0, 0, 0],
        'x1': [1, 0, 1, 0],
        'x2': [2, 2, 2, 2],
        'x3': [2, 0, 2, 0],
    }
    np.testing.assert_equal(samples, answer)


if __name__ == '__main__':
  absltest.main()
