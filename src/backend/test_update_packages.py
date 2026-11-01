import importlib.metadata
import pathlib
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import update_packages as scan


class PackageUpdates(unittest.TestCase):
    def check(self, current, latest, **options):
        distribution = Mock(version=current)
        distribution.read_text.return_value = options.get("source")
        metadata = Mock()
        metadata.distribution.return_value = distribution
        get = Mock(return_value={"info": {"version": latest}, "urls": [{"yanked": options.get("yanked", False)}],
                                 "releases": options.get("releases", {})})
        result = scan.check(options.get("package", "fixture"), options.get("offline", False),
                            options.get("pinned", []), metadata, get,
                            constraints=options.get("constraints", ""),
                            requirements=options.get("requirements", []))
        return result, get

    def test_versions_are_compared_with_pep440(self):
        for current, latest, status in [
            ("1.9", "1.10", "update_available"),
            ("2.0+cu128", "2.0", "up_to_date"),
            ("2.1", "2.0", "up_to_date"),
            ("2.0rc1", "2.0", "update_available"),
        ]:
            self.assertEqual(self.check(current, latest)[0]["status"], status)

    def test_offline_never_requests_remote_and_pins_are_not_upgrades(self):
        result, get = self.check("1", "2", offline=True)
        self.assertEqual(result["status"], "unknown")
        get.assert_not_called()
        self.assertEqual(self.check("1", "2", pinned=["fixture"])[0]["status"], "pinned")

    def test_custom_sources_prereleases_and_yanks_are_not_update_promises(self):
        result, get = self.check("1", "2", source='{"url":"file:///custom"}')
        self.assertEqual(result["status"], "unknown")
        get.assert_not_called()
        self.assertEqual(self.check("1", "2rc1")[0]["status"], "unknown")
        self.assertEqual(self.check("1", "2", yanked=True)[0]["status"], "unknown")

    def test_errors_remain_visible_in_report(self):
        self.assertEqual(self.check("invalid", "2")[0]["status"], "error")
        metadata = Mock()
        metadata.distribution.side_effect = importlib.metadata.PackageNotFoundError
        self.assertIsNone(scan.check("missing", False, [], metadata, Mock()))

    def test_local_cuda_compatibility_pin_is_not_offered_as_update(self):
        result, _ = self.check("2.11.0+cu128", "2.15.0", constraints="# GPU-compatible wheel\nfixture==2.11.0+cu128\n")
        self.assertEqual(result["status"], "pinned")
        self.assertIn("constraints.txt", result["detail"])
        self.assertEqual(self.check("2.11", "2.15", constraints="fixture<3")[0]["status"], "update_available")

    def test_laya_update_is_complete_within_the_installers_supported_range(self):
        result, _ = self.check("0.3.29", "0.4.17", package="laya",
                               requirements=["laya>=0.3.29,<0.4"],
                               releases={"0.3.29": [{}], "0.4.0": [{}], "0.4.17": [{}]})
        self.assertEqual(result["status"], "pinned")
        self.assertIn("installer requirements", result["detail"])
        self.assertIn("<0.4", result["detail"])

    def test_newer_supported_release_is_offered_even_if_latest_major_is_excluded(self):
        result, _ = self.check("0.3.29", "0.4.17", package="laya",
                               requirements=["laya>=0.3.29,<0.4"],
                               releases={"0.3.29": [{}], "0.3.30": [{}], "0.4.17": [{}]})
        self.assertEqual(result["status"], "update_available")
        self.assertEqual(result["available"], "0.3.30")
        self.assertIn("0.4.17 is excluded", result["detail"])

    def test_builtin_and_local_constraints_both_limit_the_candidate(self):
        result, _ = self.check("0.3.29", "0.4.17", package="laya",
                               requirements=["laya>=0.3.29,<0.4"],
                               constraints="laya==0.3.29",
                               releases={"0.3.29": [{}], "0.3.30": [{}], "0.4.17": [{}]})
        self.assertEqual(result["status"], "pinned")
        self.assertIn("constraints.txt", result["detail"])

    def test_excluded_yanked_prerelease_and_unsupported_python_are_not_updates(self):
        result, _ = self.check("0.3.29", "0.4.17", package="laya",
                               requirements=["laya>=0.3.29,<0.4"], releases={
                                   "0.3.30": [{"yanked": True}],
                                   "0.3.31rc1": [{}],
                                   "0.3.32": [{"requires_python": ">=99"}],
                                   "invalid": [{}],
                                   "0.4.17": [{}],
                               })
        self.assertEqual(result["status"], "pinned")

    def test_requirement_extras_normalized_names_and_markers(self):
        result, _ = self.check("6.1", "7.1", package="sentence_transformers",
                               requirements=["sentence-transformers[image]>=6.1,<7"],
                               releases={"6.2": [{}], "7.1": [{}]})
        self.assertEqual((result["status"], result["available"]), ("update_available", "6.2"))
        result, _ = self.check("1", "2", requirements=['fixture<2; sys_platform == "not-a-platform"'])
        self.assertEqual(result["status"], "update_available")


if __name__ == "__main__":
    unittest.main()
