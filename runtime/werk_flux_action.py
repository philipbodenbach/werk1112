"""Resident DROID inference bridge for the official flux-action package.

Wire stdout is JSON lines; dependencies and progress go to stderr. Predictions
are data only. This process has no robot/actuator connection.
"""
import base64
import contextlib
import io
import json
import sys


def validate_request(request):
    if not isinstance(request, dict) or set(request) - {"task", "state", "images"}:
        raise ValueError("invalid_request: expected task, state and images")
    task = request.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("invalid_request: task must not be blank")
    state = request.get("state")
    import math
    if not isinstance(state, list) or len(state) != 8 or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
        for x in state
    ):
        raise ValueError("invalid_request: DROID state requires eight finite values")
    if not 0 <= state[7] <= 1:
        raise ValueError("invalid_request: gripper closed fraction must be 0..1")
    images = request.get("images")
    if not isinstance(images, dict) or set(images) != {"wrist", "left", "right"}:
        raise ValueError("invalid_request: DROID requires wrist, left and right cameras")
    if any(not isinstance(v, str) or not v or len(v) > 8 * 1024 * 1024 for v in images.values()):
        raise ValueError("invalid_request: each camera must be a base64 image <=8 MiB")


def observation(request, torch):
    validate_request(request)
    import numpy as np
    from PIL import Image
    result = {"state": torch.tensor([request["state"]], dtype=torch.float32, device="cuda"),
              "task": [request["task"]]}
    for camera, value in request["images"].items():
        image = Image.open(io.BytesIO(base64.b64decode(value, validate=True)))
        if image.size != (640, 360) or image.mode != "RGB":
            raise ValueError("invalid_request: cameras must be synchronized 640x360 RGB images")
        array = np.array(image, copy=True)
        result["images." + camera] = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device="cuda", dtype=torch.float32) / 255
    return result


def predict(policy, obs, torch):
    with torch.no_grad():
        actions = policy.predict_action_chunk(obs)
    if tuple(actions.shape) != (1, 32, 8) or not bool(torch.isfinite(actions).all()):
        raise ValueError("unsupported_architecture: checkpoint did not produce DROID (1,32,8) actions")
    return {"actions": actions.float().cpu()[0].tolist(), "embodiment": "droid",
            "joint_units": "radians", "gripper_units": "closed_fraction",
            "action_space": "absolute_joint_targets", "executed": False}


def run(checkpoint):
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        if not torch.cuda.is_available():
            raise ValueError("incompatible_hardware: FLUX Action requires an NVIDIA CUDA GPU")
        from flux_action.policy import FluxActionPolicy
        policy = FluxActionPolicy.from_pretrained(checkpoint, device="cuda")
        policy.prepare_inference(compile=False)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request == {"preload": True}:
                    obs = {"state": torch.zeros((1, 8), device="cuda"), "task": ["Hold position."]}
                    for camera in ("wrist", "left", "right"):
                        obs["images." + camera] = torch.zeros((1, 3, 360, 640), device="cuda")
                else:
                    obs = observation(request, torch)
                response = predict(policy, obs, torch)
            print(json.dumps({"result": response}), flush=True)
        except Exception as error:
            # Details stay on stderr; no observations or arbitrary package errors
            # are returned to the public API.
            print(type(error).__name__, str(error), file=sys.stderr)
            print(json.dumps({"error": "action_inference_failed"}), flush=True)


if __name__ == "__main__":
    try:
        run(sys.argv[1])
    except Exception as error:
        print(type(error).__name__, str(error), file=sys.stderr)
        print(json.dumps({"error": "action_runtime_unavailable: check flux-action environment, checkpoint and GPU"}), flush=True)
        raise SystemExit(1)
