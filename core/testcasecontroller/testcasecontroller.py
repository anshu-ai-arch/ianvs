# Copyright 2022 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test Case Controller"""

import copy
import yaml

from core.common import utils
from core.common.log import LOGGER
from core.common.constant import TestObjectType
from core.testcasecontroller.algorithm import Algorithm
from core.testcasecontroller.testcase import TestCase


class TestCaseController:
    """
    Test Case Controller:
    Control the runtime behavior of test cases like instance generation and vanish.
    """

    def __init__(self):
        self.test_cases = []

    def build_testcases(self, test_env, test_object):
        """
        Build multiple test cases by Using a test environment and multiple test algorithms.
        """

        test_object_type = test_object.get("type")
        test_object_config = test_object.get(test_object_type)
        if test_object_type == TestObjectType.ALGORITHMS.value:
            algorithms = self._parse_algorithms_config(test_object_config)
            for algorithm in algorithms:
                self.test_cases.append(TestCase(test_env, algorithm))

    def run_testcases(self, workspace):
        """
        Run all test cases.
        """
        succeed_results = {}
        succeed_testcases = []
        failed = []
        for testcase in self.test_cases:
            try:
                res, time = (testcase.run(workspace), utils.get_local_time())
            except Exception as err:  # pylint: disable=broad-except
                # contain the failure so completed results still reach the ranking layer
                LOGGER.error("testcase(id=%s) runs failed, error: %s", testcase.id, err)
                failed.append(f"{testcase.id}: {err}")
                continue

            succeed_results[testcase.id] = (res, time)
            succeed_testcases.append(testcase)

        if failed and not succeed_testcases:
            raise RuntimeError(f"all testcases failed: {'; '.join(failed)}")

        return succeed_testcases, succeed_results

    @classmethod
    def _parse_algorithms_config(cls, config):
        if config is None:
            raise ValueError("algorithms configuration cannot be None.")
        if not isinstance(config, list):
            raise ValueError(
                f"expected list of algorithm configurations, got {type(config).__name__}."
            )
        if len(config) == 0:
            return []

        new_algorithms = []
        for index, algorithm_config in enumerate(config):
            if not isinstance(algorithm_config, dict):
                raise ValueError(
                    f"Invalid algorithm configuration at index {index}: expected dict, got {type(algorithm_config).__name__}."
                )

            name = algorithm_config.get("name")
            if not name or not isinstance(name, str):
                raise ValueError(
                    f"Missing or invalid mandatory key 'name' in algorithm configuration at index {index}."
                )

            config_file = algorithm_config.get("url")
            if not config_file or not isinstance(config_file, str):
                raise ValueError(
                    f"Missing or invalid mandatory key 'url' for algorithm '{name}' at index {index}."
                )

            if not utils.is_local_file(config_file):
                raise RuntimeError(f"not found algorithm config file({config_file}) in local")

            try:
                raw_config = utils.yaml2dict(config_file)
            except yaml.YAMLError as err:
                raise RuntimeError(
                    f"algorithm '{name}' (index {index}): invalid YAML in {config_file}: {err}"
                ) from err
            except OSError as err:
                raise RuntimeError(
                    f"algorithm '{name}' (index {index}): cannot read {config_file}: {err}"
                ) from err

            try:
                algorithm = Algorithm(name, raw_config)
            except Exception as err:  # pylint: disable=broad-except
                raise RuntimeError(
                    f"algorithm '{name}' (index {index}): failed to initialize from {config_file}, error: {err}"
                ) from err

            for modules in algorithm.modules_list:
                new_algorithm = copy.deepcopy(algorithm)
                new_algorithm.modules = modules
                new_algorithms.append(new_algorithm)

        return new_algorithms
