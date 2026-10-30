import json
import pathlib
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import update_external_check as check


def distribution(name, version, requires=(), direct=None):
    item = Mock(version=version, requires=requires, metadata={"Name": name})
    item.read_text.return_value = json.dumps(direct) if direct else None
    return item


def update(name, version, requires=()):
    return {"metadata": {"name": name, "version": version, "requires_dist": requires}}


class ExternalUpdateTests(unittest.TestCase):
    def test_torch_update_cannot_leave_installed_torchvision_incompatible(self):
        installed = [distribution("torch", "2.6.0+cu124"),
                     distribution("torchvision", "0.21.0+cu124", ["torch==2.6.0"])]
        with self.assertRaisesRegex(ValueError, "torchvision.*requires torch==2.6.0"):
            check.validate({"install": [update("torch", "2.15.0")]}, installed, ["torch>=2.6,<3"])
        result = check.validate({"install": [update("torch", "2.15.0"),
                                update("torchvision", "0.30.0", ["torch==2.15.0"])]},
                                installed, ["torch>=2.6,<3"])
        self.assertIn("torch==2.15.0\n", result)

    def test_unrelated_app_pins_are_checked_before_installation(self):
        installed = [distribution("open-webui", "0.10.2", ["transformers==5.5.4"]),
                     distribution("transformers", "5.5.4")]
        with self.assertRaisesRegex(ValueError, "open-webui"):
            check.validate({"install": [update("transformers", "5.22.0")]}, installed, [])

    def test_markers_and_transitive_requested_extras(self):
        report = {"install": [update("diffusers", "1", [
            'torch[image]>=2; extra == "torch"',
            'windows-only; sys_platform == "not-a-real-platform"',
            'optional; extra == "unused"']),
            update("torch", "2", ['Pillow>=12; extra == "image"'])]}
        with self.assertRaisesRegex(ValueError, "Pillow>=12"):
            check.validate(report, [], ["diffusers[torch]"])
        report["install"].append(update("pillow", "12"))
        self.assertIn("pillow==12", check.validate(report, [], ["diffusers[torch]"]))

    def test_downgrades_editable_and_custom_sources_are_retained(self):
        for old, message in [
            (distribution("fixture", "2"), "downgrade"),
            (distribution("fixture", "1", direct={"dir_info": {"editable": True}}), "Editable"),
            (distribution("fixture", "1", direct={"url": "file:///custom"}), "external source"),
        ]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                check.validate({"install": [update("fixture", "1")]}, [old], [])

    def test_new_package_requirements_and_normalized_names_are_checked(self):
        installed = [distribution("huggingface_hub", "1")]
        report = {"install": [update("new-package", "1", ["huggingface-hub>=2"])]}
        with self.assertRaisesRegex(ValueError, "huggingface-hub>=2"):
            check.validate(report, installed, [])
        report["install"].append(update("huggingface-hub", "2"))
        self.assertIn("huggingface-hub==2", check.validate(report, installed, []))


if __name__ == "__main__":
    unittest.main()
