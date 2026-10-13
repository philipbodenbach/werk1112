"""Convert a recorded upstream DROID observation.npz to Werk's typed request.

Usage: python droid-request.py observation.npz 'recorded task caption' > action.json
Requires numpy and Pillow in the official flux-action environment.
"""
import base64
import io
import json
import sys

import numpy as np
from PIL import Image

with np.load(sys.argv[1], allow_pickle=False) as observation:
    images = {}
    for name in ("wrist", "left", "right"):
        array = observation["images." + name]
        if array.dtype != np.uint8 or array.shape != (360, 640, 3):
            raise ValueError("expected synchronized 360x640 RGB uint8 camera frames")
        data = io.BytesIO()
        Image.fromarray(array).save(data, format="PNG")
        images[name] = base64.b64encode(data.getvalue()).decode("ascii")
    state = observation["state"]
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("expected eight finite DROID state values")
    print(json.dumps({"model": "flux3-droid", "task": sys.argv[2],
                      "state": state.tolist(), "images": images}, allow_nan=False))
