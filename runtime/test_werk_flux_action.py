import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("werk_flux_action", Path(__file__).with_name("werk_flux_action.py"))
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)


class ActionValidationTests(unittest.TestCase):
    def request(self):
        return {"task": "hold position", "state": [0.0] * 8,
                "images": {name: "AA==" for name in ("wrist", "left", "right")}}

    def test_droid_contract_and_rejected_state(self):
        BRIDGE.validate_request(self.request())
        for state in ([0] * 6, [float("nan")] * 8, [True] * 8, [0] * 7 + [2]):
            with self.assertRaises(ValueError):
                BRIDGE.validate_request(dict(self.request(), state=state))

    def test_camera_names_and_unknown_fields(self):
        for change in ({"images": {"scene": "AA=="}}, {"execute": True}, {"task": " "}):
            with self.assertRaises(ValueError):
                BRIDGE.validate_request(dict(self.request(), **change))


if __name__ == "__main__":
    unittest.main()
