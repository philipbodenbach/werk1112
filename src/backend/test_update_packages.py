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
        get = Mock(return_value={"info": {"version": latest}, "urls": [{"yanked": options.get("yanked", False)}]})
        result = scan.check("fixture", options.get("offline", False), options.get("pinned", []), metadata, get,
                            constraints=options.get("constraints", ""))
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


if __name__ == "__main__":
    unittest.main()
