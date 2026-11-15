# Copyright 2026 The KubeEdge Authors.
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

import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

# Ensure mock subpackages if sedna/onnx are not installed in the test environment
def mock_module(name):
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        mod_name = ".".join(parts[:i])
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            m.__path__ = []
            sys.modules[mod_name] = m
    return sys.modules[name]

for mod in [
    "sedna.common.class_factory",
    "sedna.datasources",
    "sedna.core.base",
    "sedna.core.joint_inference",
    "sedna.core.incremental_learning",
    "sedna.core.lifelong_learning",
    "sedna.algorithms.aggregation",
    "onnx",
]:
    mock_module(mod)

sys.modules["sedna.common.class_factory"].ClassFactory = MagicMock()
sys.modules["sedna.common.class_factory"].ClassType = MagicMock()
sys.modules["sedna.core.base"].JobBase = type("JobBase", (), {})
sys.modules["sedna.core.joint_inference"].JointInference = type("JointInference", (), {})
sys.modules["sedna.core.incremental_learning"].IncrementalLearning = type("IncrementalLearning", (), {})
sys.modules["sedna.core.lifelong_learning"].LifelongLearning = type("LifelongLearning", (), {})
sys.modules["sedna.algorithms.aggregation"].AggClient = type("AggClient", (), {})

for name in ["BaseDataSource", "CSVDataParse", "TxtDataParse", "JSONDataParse"]:
    setattr(sys.modules["sedna.datasources"], name, MagicMock)
    setattr(sys.modules["sedna.core.base"], name, MagicMock)

import yaml

from core.testcasecontroller.testcasecontroller import TestCaseController


class TestParseAlgorithmsConfig(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.algo_dir = Path(self.temp_dir.name)

        # Create a valid algorithm yaml
        self.valid_algo_yaml = self.algo_dir / "valid_algorithm.yaml"
        algo_content = {
            "algorithm": {
                "paradigm_type": "singletasklearning",
                "modules": [
                    {
                        "type": "basemodel",
                        "name": "BaseModel",
                        "url": "/tmp/dummy_model.py",
                    }
                ],
            }
        }
        with open(self.valid_algo_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(algo_content, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_algorithms_config_success(self):
        """Valid algorithm configuration is parsed and returns list of expanded algorithms."""
        config = [
            {"name": "test_algo", "url": str(self.valid_algo_yaml)}
        ]
        with patch("core.testcasecontroller.testcasecontroller.Algorithm") as mock_algo_cls:
            mock_algo_instance = MagicMock()
            mock_algo_instance.modules_list = [["module_set_1"], ["module_set_2"]]
            mock_algo_cls.return_value = mock_algo_instance

            algorithms = TestCaseController._parse_algorithms_config(config)

            self.assertEqual(len(algorithms), 2)
            mock_algo_cls.assert_called_once_with("test_algo", unittest.mock.ANY)

    def test_permutations_are_deepcopied_independently(self):
        """Mutating modules on one expanded algorithm does not mutate others."""
        config = [
            {"name": "test_algo", "url": str(self.valid_algo_yaml)}
        ]
        with patch("core.testcasecontroller.testcasecontroller.Algorithm") as mock_algo_cls:
            class DummyAlgorithm:
                def __init__(self, name, cfg):
                    self.name = name
                    self.modules_list = [[{"param": 1}], [{"param": 2}]]
                    self.modules = None

            mock_algo_cls.side_effect = DummyAlgorithm

            algorithms = TestCaseController._parse_algorithms_config(config)
            self.assertEqual(len(algorithms), 2)

            # Mutate modules of first permutation
            algorithms[0].modules[0]["param"] = 999
            self.assertEqual(algorithms[1].modules[0]["param"], 2)

    def test_none_config_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            TestCaseController._parse_algorithms_config(None)
        self.assertIn("cannot be None", str(ctx.exception))

    def test_non_list_config_raises_value_error(self):
        for invalid in [{"name": "a"}, "invalid_str", 123]:
            with self.assertRaises(ValueError) as ctx:
                TestCaseController._parse_algorithms_config(invalid)
            self.assertIn("expected list of algorithm configurations", str(ctx.exception))

    def test_empty_list_config_returns_empty_list(self):
        result = TestCaseController._parse_algorithms_config([])
        self.assertEqual(result, [])

    def test_non_dict_element_raises_value_error(self):
        config = ["not_a_dict"]
        with self.assertRaises(ValueError) as ctx:
            TestCaseController._parse_algorithms_config(config)
        self.assertIn("expected dict", str(ctx.exception))

    def test_missing_or_invalid_name_raises_value_error(self):
        invalid_names = [None, "", 123, []]
        for name_val in invalid_names:
            config = [{"url": str(self.valid_algo_yaml)}]
            if name_val is not None:
                config[0]["name"] = name_val
            with self.assertRaises(ValueError) as ctx:
                TestCaseController._parse_algorithms_config(config)
            self.assertIn("mandatory key 'name'", str(ctx.exception))

    def test_missing_or_invalid_url_raises_value_error(self):
        invalid_urls = [None, "", 123, []]
        for url_val in invalid_urls:
            config = [{"name": "test_algo"}]
            if url_val is not None:
                config[0]["url"] = url_val
            with self.assertRaises(ValueError) as ctx:
                TestCaseController._parse_algorithms_config(config)
            self.assertIn("mandatory key 'url'", str(ctx.exception))
            self.assertIn("test_algo", str(ctx.exception))

    def test_missing_local_file_raises_runtime_error(self):
        non_existent_file = str(self.algo_dir / "does_not_exist.yaml")
        config = [{"name": "test_algo", "url": non_existent_file}]
        with self.assertRaises(RuntimeError) as ctx:
            TestCaseController._parse_algorithms_config(config)
        self.assertIn(f"not found algorithm config file({non_existent_file}) in local", str(ctx.exception))

    def test_corrupt_yaml_syntax_raises_runtime_error_with_cause(self):
        """Corrupt YAML on disk raises RuntimeError and chains yaml.YAMLError as __cause__."""
        corrupt_yaml = self.algo_dir / "corrupt.yaml"
        corrupt_yaml.write_text("invalid_yaml: [unclosed_list", encoding="utf-8")

        config = [{"name": "broken_algo", "url": str(corrupt_yaml)}]
        with self.assertRaises(RuntimeError) as ctx:
            TestCaseController._parse_algorithms_config(config)

        self.assertIn("invalid YAML", str(ctx.exception))
        self.assertIn("broken_algo", str(ctx.exception))
        self.assertIn("index 0", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, yaml.YAMLError)

    def test_os_error_raises_runtime_error_with_cause(self):
        """File read error raises RuntimeError and chains OSError as __cause__."""
        config = [{"name": "os_error_algo", "url": str(self.valid_algo_yaml)}]
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with self.assertRaises(RuntimeError) as ctx:
                TestCaseController._parse_algorithms_config(config)

        self.assertIn("cannot read", str(ctx.exception))
        self.assertIn("os_error_algo", str(ctx.exception))
        self.assertIn("index 0", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, OSError)

    def test_algorithm_init_failure_reports_name_and_index(self):
        """Algorithm initialization failure reports name, index, and config file."""
        config = [{"name": "init_fail_algo", "url": str(self.valid_algo_yaml)}]
        with patch("core.testcasecontroller.testcasecontroller.Algorithm", side_effect=ValueError("Invalid paradigm")):
            with self.assertRaises(RuntimeError) as ctx:
                TestCaseController._parse_algorithms_config(config)

        self.assertIn("init_fail_algo", str(ctx.exception))
        self.assertIn("index 0", str(ctx.exception))
        self.assertIn("failed to initialize", str(ctx.exception))

    def test_end_to_end_example_algorithms_parse(self):
        """Ensure existing algorithm YAML files in examples/ can be parsed without error."""
        repo_root = Path(__file__).resolve().parents[2]
        old_cwd = os.getcwd()
        try:
            os.chdir(repo_root)
            scanned = 0
            for algo_yaml in repo_root.glob("examples/**/testalgorithms/**/*.yaml"):
                if not algo_yaml.is_file():
                    continue
                rel_path = algo_yaml.relative_to(repo_root).as_posix()
                config = [{"name": "example_algo", "url": rel_path}]
                with patch("core.testcasecontroller.testcasecontroller.Algorithm") as mock_algo:
                    mock_instance = MagicMock()
                    mock_instance.modules_list = [[]]
                    mock_algo.return_value = mock_instance
                    TestCaseController._parse_algorithms_config(config)
                    scanned += 1
            self.assertGreater(scanned, 5, f"Expected at least 5 algorithm configs, found {scanned}")
        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
