"""Metadata-only preflight, executed by the selected MLX launcher interpreter.

Never call load(), load_model(), or a model constructor: even lazy loading can
allocate an entire model. Only installed runtime code is imported. The probe
uses mlx-lm's own resolver and at most a 1 x 1024 synthetic quantization input.
"""

import ast
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import textwrap


def launcher_module(path):
    """Recognize Python console entry points without executing launcher code."""
    with open(path, encoding="utf-8") as launcher:
        source = launcher.read(65537)
    if len(source) > 65536:
        raise ValueError("cannot verify MLX launcher: script exceeds 64 KiB")
    tree = ast.parse(source)
    entry = None
    for node in tree.body:
        if isinstance(node, ast.Import) and all(
            name.name in ("sys", "re") and name.asname is None for name in node.names
        ):
            continue
        if isinstance(node, ast.ImportFrom) and node.level == 0 and len(node.names) == 1:
            name = node.names[0]
            if name.name == "main" and name.asname is None and entry is None:
                entry = node.module
                continue
        if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
            # pip's launchers optionally normalize sys.argv[0] before main().
            body = node.body
            # uv entry points normalize Windows suffixes with an if/elif.
            # Match the complete inert AST, never arbitrary launcher code.
            uv_normalization = ast.parse("""
if sys.argv[0].endswith("-script.pyw"):
    sys.argv[0] = sys.argv[0][:-11]
elif sys.argv[0].endswith(".exe"):
    sys.argv[0] = sys.argv[0][:-4]
""").body[0]
            if body and ast.dump(body[0]) == ast.dump(uv_normalization):
                body = body[1:]
            if body and isinstance(body[0], ast.Assign):
                assignment = body[0]
                if len(assignment.targets) != 1 or ast.unparse(assignment.targets[0]) != "sys.argv[0]":
                    raise ValueError("cannot verify custom MLX launcher assignments")
                expected = ast.parse(r"re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])", mode="eval").body
                homebrew = ast.parse("sys.argv[0].removesuffix('.exe')", mode="eval").body
                if ast.dump(assignment.value) not in (ast.dump(expected), ast.dump(homebrew)):
                    raise ValueError("cannot verify custom MLX launcher argv handling")
                body = body[1:]
            if not node.orelse and len(body) == 1 and ast.unparse(body[0]) == "sys.exit(main())":
                continue
        raise ValueError("cannot verify custom MLX launcher; select its Python and module explicitly")
    if not entry:
        raise ValueError("cannot identify the MLX launcher's Python module")
    return entry


def version(name, module):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown"))


def function_tree(function):
    try:
        return ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError, SyntaxError) as error:
        raise ValueError("cannot verify installed runtime implementation: Python source is unavailable") from error


def is_subscript(node, value, key):
    return (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id == value and isinstance(node.slice, ast.Constant) and node.slice.value == key)


def is_get(node, value, key):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == value
            and node.func.attr == "get" and node.args
            and isinstance(node.args[0], ast.Constant) and node.args[0].value == key)


def standard_loader(module):
    utils = importlib.import_module("mlx_lm.utils")
    load = getattr(module, "load", None)
    loader = getattr(utils, "load_model", None)
    if (load is not getattr(utils, "load", None) or not callable(load)
            or not callable(loader) or getattr(load, "__globals__", {}).get("load_model") is not loader):
        raise ValueError("cannot verify architecture resolution: configured module must re-export the installed mlx_lm.utils.load; custom load wrappers are unverified")
    calls = [node for node in ast.walk(function_tree(load)) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "load_model"]
    message = "cannot verify architecture resolution: installed load overrides or hides its model-class resolver"
    if not calls:
        raise ValueError(message)
    signature = inspect.signature(loader)
    for call in calls:
        if any(isinstance(arg, ast.Starred) for arg in call.args) or any(
                keyword.arg is None for keyword in call.keywords):
            raise ValueError(message)
        try:
            # AST nodes are inert argument values: binding neither evaluates
            # their expressions nor calls the loader. Positional lazy/strict
            # bind normally; a positional or keyword resolver is still explicit.
            bound = signature.bind(*call.args, **{
                keyword.arg: keyword.value for keyword in call.keywords
            })
        except TypeError as error:
            raise ValueError(message) from error
        if "get_model_classes" in bound.arguments or any(
                signature.parameters[name].kind in (
                    inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD,
                ) for name in bound.arguments):
            raise ValueError(message)
    return loader


def verify_quantization_loader(loader, mixed, modes):
    tree = function_tree(loader)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
             and node.func.value.id == "nn" and node.func.attr == "quantize"]
    for call in calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        group = keywords.get("group_size")
        if not isinstance(group, ast.Subscript) or not isinstance(group.value, ast.Name):
            continue
        quant = group.value.id
        if not is_subscript(group, quant, "group_size") or not is_subscript(keywords.get("bits"), quant, "bits"):
            continue
        if any(mode != "affine" for mode in modes) and not is_get(keywords.get("mode"), quant, "mode"):
            continue
        if mixed:
            predicate = keywords.get("class_predicate")
            if not isinstance(predicate, ast.Name):
                continue
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                         and node.name == predicate.id]
            forwards_dictionary = False
            for function in functions:
                for node in ast.walk(function):
                    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
                        continue
                    test = node.test
                    if (len(test.ops) != 1 or not isinstance(test.ops[0], ast.In)
                            or not isinstance(test.left, ast.Name)
                            or not is_subscript(test.comparators[0], "config", "quantization")):
                        continue
                    for statement in node.body:
                        value = statement.value if isinstance(statement, ast.Return) else None
                        if (isinstance(value, ast.Subscript) and is_subscript(value.value, "config", "quantization")
                                and isinstance(value.slice, ast.Name) and value.slice.id == test.left.id):
                            forwards_dictionary = True
            if not forwards_dictionary:
                continue
        return
    raise ValueError("incompatible runtime version: cannot verify loader forwarding of quantization modes/per-layer layouts")


def verify_nn_dictionary_dispatch(quantize):
    for node in ast.walk(function_tree(quantize)):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Call):
            continue
        test = node.test
        if (not isinstance(test.func, ast.Name) or test.func.id != "isinstance" or len(test.args) != 2
                or not isinstance(test.args[0], ast.Name) or not isinstance(test.args[1], ast.Name)
                or test.args[1].id != "dict"):
            continue
        for statement in node.body:
            for call in ast.walk(statement):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "to_quantized" and any(
                            keyword.arg is None and isinstance(keyword.value, ast.Name)
                            and keyword.value.id == test.args[0].id for keyword in call.keywords)):
                    return
    raise ValueError("incompatible runtime version: mlx.nn.quantize cannot verify per-layer parameter dictionaries")


def legacy_mxfp4_quantization(loader, nested):
    # The pinned loader has no separate metadata-only normalizer: this branch
    # runs after weight loading and model construction. Recognize its installed
    # AST, then copy only the fixed metadata locally; never execute that branch.
    # Source: mlx-lm/utils.py, 6d21ce4b065a2e163fa6de76a9936c61aeb5784a,
    # lines 376-379 (nested metadata) and 408-419 (MXFP4 dispatch).
    message = "cannot verify legacy quantization_config layout 'mxfp4': installed loader normalization is unsupported or unverified"
    if loader is None:
        raise ValueError(message)
    body = function_tree(loader).body[0].body
    if nested:
        promotion = ast.parse('''
if "quantization_config" not in config:
    text_config = config.get("text_config", {})
    if "quantization_config" in text_config:
        config["quantization_config"] = text_config["quantization_config"]
''').body[0]
        if not any(ast.dump(node) == ast.dump(promotion) for node in body):
            raise ValueError(message)
    normalization = ast.parse('''
quantization = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
config["quantization"] = quantization
config["quantization_config"] = quantization
_quantize(quantization)
''').body
    method_assignment = ast.parse('quant_method = quantization_config["quant_method"]').body[0]
    method_test = ast.parse('quant_method == "mxfp4"', mode="eval").body
    # Walk only executable top-level if/elif chains, not unrelated nested
    # helpers that happen to contain similar metadata or normalization code.
    for statement in body:
        node = statement
        while isinstance(node, ast.If):
            if (isinstance(node.test, ast.NamedExpr)
                    and isinstance(node.test.target, ast.Name)
                    and node.test.target.id == "quantization_config"
                    and is_get(node.test.value, "config", "quantization_config")
                    and node.body and ast.dump(node.body[0]) == ast.dump(method_assignment)):
                for branch in node.body[1:]:
                    while isinstance(branch, ast.If):
                        if (ast.dump(branch.test) == ast.dump(method_test)
                                and [ast.dump(item) for item in branch.body]
                                == [ast.dump(item) for item in normalization]):
                            return {"group_size": 32, "bits": 4, "mode": "mxfp4"}
                        branch = branch.orelse[0] if len(branch.orelse) == 1 else None
            node = node.orelse[0] if len(node.orelse) == 1 else None
    raise ValueError(message)


def quantization_layouts(config, loader=None):
    quant = config.get("quantization")
    legacy = config.get("quantization_config", config.get("text_config", {}).get("quantization_config"))
    if quant is None:
        if legacy:
            method = legacy.get("quant_method", "unknown") if isinstance(legacy, dict) else "invalid"
            if method != "mxfp4":
                raise ValueError(f"cannot verify legacy quantization_config layout '{method}' without a supported metadata contract")
            quant = legacy_mxfp4_quantization(loader, nested="quantization_config" not in config)
        else:
            return []
    if not isinstance(quant, dict):
        raise ValueError("damaged model metadata: quantization must be an object")
    if not all(key in quant for key in ("group_size", "bits")):
        raise ValueError("damaged model metadata: quantization requires group_size and bits")
    layouts = []
    entries = [("default", {key: quant[key] for key in ("group_size", "bits", "mode") if key in quant})]
    entries += [(key, value) for key, value in quant.items() if key not in ("group_size", "bits", "mode")]
    for name, entry in entries:
        if entry is False:
            continue
        if not isinstance(entry, dict) or any(key not in ("group_size", "bits", "mode") for key in entry):
            raise ValueError(f"cannot verify quantization layout for '{name}'")
        group, bits, mode = entry.get("group_size"), entry.get("bits"), entry.get("mode", "affine")
        if type(group) is not int or type(bits) is not int or group <= 0 or bits <= 0 or not isinstance(mode, str):
            raise ValueError(f"damaged model metadata: invalid quantization layout for '{name}'")
        if group > 1024:
            raise ValueError(f"cannot verify quantization group_size {group}: bounded probe limit is 1024")
        if (group, bits, mode) not in layouts:
            layouts.append((group, bits, mode))
    return layouts


def probe(payload, target, launcher=False):
    config = payload.get("config")
    if "config" in payload:
        if not isinstance(config, dict) or not isinstance(config.get("model_type"), str) or not config["model_type"]:
            raise ValueError("damaged model metadata: config.json requires a nonempty model_type")
        if config.get("model_file") and not payload.get("werk_gemma4_compat"):
            raise ValueError("cannot verify model_file architecture without executing model repository code")
    module_name = launcher_module(target) if launcher else target
    if launcher and not getattr(sys.flags, "safe_path", False):
        # python -c prepends cwd; running the actual console script prepends its
        # resolved directory. Match that environment before importing its entry.
        sys.path[0] = str(Path(target).resolve().parent)
    module = importlib.import_module(module_name)
    mlx_lm = importlib.import_module("mlx_lm")
    mx = importlib.import_module("mlx.core")
    runtime = f"mlx-lm {version('mlx-lm', mlx_lm)}, mlx {version('mlx', mx)}, module {module_name}, Python {sys.executable}"
    if "config" not in payload:
        return runtime
    try:
        # A custom generator may re-export the standard installed load function.
        # A wrapper can override its resolver, so its default cannot certify it.
        loader = standard_loader(module)
        parameter = inspect.signature(loader).parameters.get("get_model_classes")
        resolver = parameter.default if parameter else None
        if not callable(resolver):
            raise ValueError("incompatible runtime implementation: no metadata-only architecture resolver")
        resolver_config = dict(config)
        resolver_config.pop("model_file", None)
        try:
            model_class, args_class = resolver(resolver_config)
        except (ImportError, ValueError) as error:
            raise ValueError(f"architecture '{config['model_type']}' is unsupported by the installed runtime: {error}") from error
        if not callable(model_class) or not callable(getattr(args_class, "from_dict", None)):
            raise ValueError("incompatible runtime implementation: unresolved model or ModelArgs class")
        try:
            args_class.from_dict(resolver_config)
        except (TypeError, KeyError, ValueError) as error:
            raise ValueError(f"damaged or incompatible model metadata for '{config['model_type']}': {error}") from error
        if payload.get("werk_gemma4_compat"):
            nodes = list(ast.walk(function_tree(loader)))
            if not (any(is_get(node, "config", "model_file") for node in nodes)
                    and any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "spec_from_file_location" for node in nodes)
                    and any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "exec_module" for node in nodes)):
                raise ValueError("incompatible runtime version: Werk's Gemma4 compatibility model_file is not supported")
            trust = inspect.signature(loader).parameters.get("trust_remote_code")
            if trust is not None and trust.default is False:
                raise ValueError("incompatible runtime version: the configured loader disables Werk's Gemma4 compatibility model_file")
        layouts = quantization_layouts(config, loader)
        if layouts:
            mixed = any(key not in ("group_size", "bits", "mode") for key in (config.get("quantization") or {}))
            verify_quantization_loader(loader, mixed, [mode for _, _, mode in layouts])
            nn = importlib.import_module("mlx.nn")
            if mixed:
                verify_nn_dictionary_dispatch(nn.quantize)
            for group, bits, mode in layouts:
                try:
                    # A tiny synthetic input exercises the installed core API;
                    # no model weights or model-sized dimensions are consulted.
                    kwargs = {"group_size": group, "bits": bits}
                    if mode != "affine" or "mode" in inspect.signature(nn.quantize).parameters:
                        kwargs["mode"] = mode
                    inspect.signature(nn.quantize).bind(None, **kwargs)
                    result = mx.quantize(mx.zeros((1, max(32, group))), **kwargs)
                    mx.eval(result)
                except Exception as error:
                    raise ValueError(f"incompatible runtime quantization {mode}/{bits}-bit/group-{group}: {error}") from error
        if not getattr(getattr(mx, "metal", None), "is_available", lambda: False)():
            raise ValueError("MLX Metal device is unavailable in the selected Python environment")
    except Exception as error:
        raise ValueError(f"{error} ({runtime})") from error
    return runtime


def main():
    try:
        detail = probe(json.load(sys.stdin), sys.argv[2], launcher=sys.argv[1] == "launcher")
        print(json.dumps({"ok": True, "detail": detail}))
    except Exception as error:
        print(json.dumps({"ok": False, "detail": f"{error} (selected Python {sys.executable})"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
