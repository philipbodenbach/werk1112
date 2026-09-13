"""Weight-free oMLX preflight, executed by the selected console interpreter.

Imports only installed runtime code. Never call load(), load_model(), tokenizer
constructors, or model constructors. The only tensor work is a bounded synthetic
quantizer check. Safetensors headers are read without accessing tensor payloads.
"""

import ast
import contextlib
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import struct
import sys
import textwrap


# These helpers deliberately share the established MLX probe's source contracts.
# This file is embedded in the Rust binary and cannot import a repository sibling.
def launcher_module(path):
    """Recognize Python console entry points without executing launcher code."""
    with open(path, encoding="utf-8") as launcher:
        source = launcher.read(65537)
    if len(source) > 65536:
        raise ValueError("cannot verify oMLX launcher: script exceeds 64 KiB")
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
            if body and isinstance(body[0], ast.Assign):
                assignment = body[0]
                if len(assignment.targets) != 1 or ast.unparse(assignment.targets[0]) != "sys.argv[0]":
                    raise ValueError("cannot verify custom oMLX launcher assignments")
                expected = ast.parse(r"re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])", mode="eval").body
                homebrew = ast.parse("sys.argv[0].removesuffix('.exe')", mode="eval").body
                if ast.dump(assignment.value) not in (ast.dump(expected), ast.dump(homebrew)):
                    raise ValueError("cannot verify custom oMLX launcher argv handling")
                body = body[1:]
            if not node.orelse and len(body) == 1 and ast.unparse(body[0]) == "sys.exit(main())":
                continue
        raise ValueError("cannot verify custom oMLX launcher; set WERK_OMLX_BIN to the installed omlx.cli Python console script")
    if not entry:
        raise ValueError("cannot identify the oMLX launcher's Python module")
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
    legacy = config.get("quantization_config", (config.get("text_config") or {}).get("quantization_config"))
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


# Executable AST contracts, not version or comment-based capability assertions.
# oMLX v0.6.4: https://github.com/jundot/omlx/tree/v0.6.4/omlx
# mlx-lm: ab1806e8f5d6aa035973af194a1b9198ab4754dc
# Only side-effect-sensitive functions are pinned; unrelated module edits and
# comments/docstrings do not invalidate a verified implementation.
CONTRACTS = {
    "dispatcher": {"7b34bb4e4fdbe25e1338c02e5121186a48e230dbf897ae0099388b521b7507ad"},
    "config_patch": {"370f920ee02984517e0b7c96f943b453b10bf9c6e861a189c40ac894c55bdb95"},
    "config_wrapper": {"895d17f86b7babed6d20200b8d26354ae91bff5de894ace3e1b431be70686b39"},
    "text_load_wrapper": {"b63a68b27c37654415b1d165597f02a0355ae45261f07904f308c25545763628"},
    "text_loader": {"97aff65d843806bcaf16169f78db27593a034539ac1dddb8ffdcbab8a91b2e05"},
    "config_reader": {"09869a04b2043adf24b8763486a5197c344d57e499a59e6125e4c515f5e19dcb"},
    "resolver": {"2ccd643c6717c8c5d871f20302d10cf28d2bff3b7595f7b77bd27ae6140719f8"},
    "base_args": {"b0e3ce91635b8189389155c071223824a40a2b49d1d65d101ebbb455f960a2c7"},
    "mtp_args": {"360b7131b70899797b478714cdf4b50b9eb574e35fa0006ade70ae8db62ad010"},
    "deepseek_post_init": {"fcf7c8be44ad7bc6a1486d0350ad00debe84b1eb811dec3601bb8e91b2aa5cfb"},
    "deepseek_patch": {"914a494711228e64ff92002739fc61757218334fe2aa3b3803f3f07f1cf84aa5"},
    "deepseek_loader": {"e0947e4523ae6d49350f47c54284da771d4f6337832e6fbeed6ebc019ae20bb3"},
    "deepseek_quant": {"e8dd0d16a59b10e80790939d7b86ed178a28277b0d971ca4699f8516b80eb22a"},
    "deepseek_tokenizer_patch": {"a9cef3a580b1bc3d75764cb2bf15e11fcae9cfe001680464eafa6a607ccb60ed"},
    "deepseek_tokenizer": {"0d5b1b4229d30a1e279258be649e6671f4f1e61cded088301bad2a78685edfab"},
    "deepseek_parser": {"fe3ad579842a87f816d6def9992f900112ecadc9d7d50005eab3db32171b475f"},
    "native_tokenizer": {"81805beafb00c4a23b60252397f3dbfdd16fb3312b409bcc02d6d5ccaa215d3b"},
    "native_tokenizer_loader": {"5a54ff0969ea3e2766be98887f244b7f0cfa7f889dfed70df620700317c2eab2"},
    "native_tool_inference": {"ce5386192baf92f6f8590e868ea81e748102ba5e773cfcc05134e16a4e54741c"},
    # Entire module: parser, schema-based argument conversion and delimiters.
    "qwen_coder_parser_module": {"f89e1b330159dc991c04595362c62eeec26f92de4dcea9a429da28031af41088"},
}
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_SHARDS = 4096


def canonical_ast(node):
    if isinstance(node, ast.AST):
        fields = []
        for name, value in ast.iter_fields(node):
            # Python 3.12 added an empty type_params field to older syntax.
            if name in ("type_params", "type_comment") and not value:
                continue
            if (name == "body" and isinstance(value, list) and value
                    and isinstance(value[0], ast.Expr)
                    and isinstance(value[0].value, ast.Constant)
                    and isinstance(value[0].value.value, str)):
                value = value[1:]
            fields.append((name, canonical_ast(value)))
        return [type(node).__name__, fields]
    if isinstance(node, list):
        return [canonical_ast(value) for value in node]
    if node is not None and not isinstance(node, (str, int, float, bool)):
        return [type(node).__name__, repr(node)]
    return node


def ast_digest(tree):
    encoded = json.dumps(canonical_ast(tree), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def verify_contract(function, contract):
    if ast_digest(function_tree(function)) not in CONTRACTS[contract]:
        raise ValueError(f"unverified oMLX loader contract: {contract}")


def read_json(path, required=True, installed_text_port=False):
    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_JSON_BYTES + 1)
    except FileNotFoundError:
        if not required:
            return {}
        raise ValueError(f"missing model metadata: {path.name}") from None
    if len(data) > MAX_JSON_BYTES:
        raise ValueError(f"model metadata exceeds bounded probe limit: {path.name}")
    try:
        result = json.loads(data)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError(f"damaged model metadata: {path.name}: {error}") from error
    if not isinstance(result, dict):
        raise ValueError(f"damaged model metadata: {path.name} must be an object")
    validate_metadata(result, installed_text_port=installed_text_port)
    return result


def validate_metadata(config, installed_text_port=False):
    pending = [(config, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if depth > 32 or count > 250000:
            raise ValueError("model metadata exceeds bounded probe structure limit")
        if isinstance(item, dict):
            installed_override = (installed_text_port and item is config
                                  and item.get("model_type") == "qwen4_exp"
                                  and item.get("model_file") == "qwen4_exp.py")
            if item.get("model_file") is not None and not installed_override:
                raise ValueError("model_file requires executing model repository code; oMLX preflight cannot verify it")
            for key, value in item.items():
                # ModelArgs may expand these counts into Python lists.
                if (key.endswith(("num_hidden_layers", "n_mtp_layers", "num_nextn_predict_layers"))
                        or key in ("num_layers", "n_layers", "mtp_num_hidden_layers")):
                    if type(value) is not int or not 0 <= value <= 4096:
                        raise ValueError(f"model metadata {key} exceeds bounded probe limit")
                pending.append((value, depth + 1))
        elif isinstance(item, list):
            if len(item) > 65536:
                raise ValueError("model metadata list exceeds bounded probe limit")
            pending.extend((value, depth + 1) for value in item)
    model_type = config.get("model_type")
    if model_type is not None and (
            not isinstance(model_type, str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", model_type) is None):
        raise ValueError("damaged model metadata: invalid model_type")
    text_config = config.get("text_config")
    if text_config is not None and not isinstance(text_config, dict):
        raise ValueError("damaged model metadata: text_config must be an object")
    if (model_type is not None and "text_config" in config and text_config is None
            and "quantization_config" not in config):
        raise ValueError("installed oMLX loader cannot promote quantization_config from null text_config")


def check_headers(model_dir):
    count = 0
    for shard in model_dir.glob("model*.safetensors"):
        count += 1
        if count > MAX_SHARDS:
            raise ValueError("too many safetensors shards for bounded oMLX preflight")
        with shard.open("rb") as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise ValueError(f"damaged safetensors header: {shard.name}")
            header_size = struct.unpack("<Q", length_bytes)[0]
            if header_size > MAX_JSON_BYTES or header_size < 2:
                raise ValueError(f"safetensors header exceeds bounded probe limit: {shard.name}")
            # fstat does not read the tensor payload or follow a second path.
            file_size = os.fstat(stream.fileno()).st_size
            if header_size > file_size - 8:
                raise ValueError(f"truncated safetensors header: {shard.name}")
            header_data = stream.read(header_size)
        try:
            header = json.loads(header_data)
        except (ValueError, UnicodeError, RecursionError) as error:
            raise ValueError(f"damaged safetensors header: {shard.name}") from error
        if not isinstance(header, dict):
            raise ValueError(f"damaged safetensors header: {shard.name}")
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(tensor, dict) or not isinstance(tensor.get("dtype"), str):
                raise ValueError(f"damaged tensor metadata: {shard.name}")
            if tensor["dtype"] == "F8_E8M0":
                raise ValueError(
                    "raw F8_E8M0 requires oMLX's in-place safetensors header conversion; "
                    "use an MLX-converted checkpoint with standard dtypes"
                )
    if not count:
        raise ValueError("no model*.safetensors files in the selected oMLX model directory")


def installed_module(name, model_dir=None):
    # find_spec of a dotted name imports its parent. Check and import each
    # parent individually so a model-local package cannot execute first.
    parts = name.split(".")
    for length in range(1, len(parts) + 1):
        prefix = ".".join(parts[:length])
        module = sys.modules.get(prefix)
        if module is not None:
            origin = getattr(module, "__file__", None)
            locations = getattr(module, "__path__", ())
        else:
            spec = importlib.util.find_spec(prefix)
            if spec is None:
                raise ImportError(f"installed oMLX runtime module is unavailable: {prefix}")
            origin = spec.origin
            locations = spec.submodule_search_locations or ()
        # MLX 0.32 ships its top-level 'mlx' as a namespace package, with
        # no __init__.py. Its search locations still have to be installed paths.
        if (not origin and not locations) or origin in ("built-in", "frozen"):
            raise ValueError(f"unverified installed oMLX module origin: {prefix}")
        paths = ([origin] if origin else []) + list(locations)
        if model_dir is not None and any(
                Path(path).resolve().is_relative_to(model_dir)
                for path in paths):
            raise ValueError(f"refusing model repository Python module: {prefix}")
        if module is None:
            module = importlib.import_module(prefix)
    return module


def check_function_origin(function, model_dir):
    try:
        filename = inspect.getsourcefile(function)
    except TypeError as error:
        raise ValueError("unverified installed runtime function origin") from error
    if not filename or Path(filename).resolve().is_relative_to(model_dir):
        raise ValueError("refusing unverified/model repository Python function")


def prepare_runtime(root, config):
    loading = installed_module("omlx.utils.model_loading", root)
    utils = installed_module("mlx_lm.utils", root)
    verify_contract(loading.maybe_apply_pre_load_patches, "dispatcher")
    verify_contract(loading._patch_mlx_lm_load_config, "config_patch")
    verify_contract(loading.lm_load_compat, "text_load_wrapper")
    verify_contract(loading.load_text_model, "text_loader")
    verify_contract(utils.load_config, "config_reader")
    verify_contract(utils._get_classes, "resolver")
    deepseek = config["model_type"].startswith("deepseek_v4")
    if deepseek:
        patch_module = installed_module("omlx.patches.deepseek_v4", root)
        tokenizer_patch = installed_module("omlx.patches.deepseek_v4.tokenizer_patch", root)
        verify_contract(patch_module.apply_deepseek_v4_patch, "deepseek_patch")
        verify_contract(tokenizer_patch.apply_load_patch, "deepseek_tokenizer_patch")
    # Only this verified installed dispatcher is called; never any loader.
    loading.maybe_apply_pre_load_patches(str(root), model_settings=None, for_vlm=False)
    verify_contract(utils.load_config, "config_wrapper")
    runtime_config = utils.load_config(root)
    validate_metadata(runtime_config)
    mlx_lm = installed_module("mlx_lm", root)
    loader = standard_loader(mlx_lm)
    if deepseek:
        verify_contract(loader, "deepseek_loader")
    parameter = inspect.signature(loader).parameters.get("get_model_classes")
    resolver = parameter.default if parameter else None
    if resolver is not utils._get_classes:
        raise ValueError("unverified oMLX loader contract: model-class resolver override")
    model_class, args_class = resolver(runtime_config)
    check_function_origin(model_class, root)
    check_function_origin(args_class, root)
    from_dict = getattr(args_class, "from_dict", None)
    digest = ast_digest(function_tree(from_dict))
    if digest not in CONTRACTS["base_args"] | CONTRACTS["mtp_args"]:
        raise ValueError("unverified oMLX loader contract: metadata-only ModelArgs.from_dict")
    if digest in CONTRACTS["mtp_args"]:
        original = inspect.getclosurevars(from_dict).nonlocals.get("original_from_dict")
        verify_contract(original, "base_args")
    if deepseek:
        verify_contract(args_class.__post_init__, "deepseek_post_init")
    args_class.from_dict(runtime_config)
    return runtime_config, utils, loader


def omlx_quantization_layouts(config, loader, root):
    legacy = config.get("quantization_config", (config.get("text_config") or {}).get("quantization_config"))
    if (config.get("quantization") is None and isinstance(legacy, dict)
            and legacy.get("quant_method") == "fp8"
            and config["model_type"].startswith("deepseek_v4")):
        # The full map requires a constructed model. Read the verified installed
        # function's fixed layouts instead; never execute it or fabricate a tree.
        verify_contract(loader, "deepseek_loader")
        architecture = installed_module("mlx_lm.models.deepseek_v4", root)
        verify_contract(architecture.make_quantization_config, "deepseek_quant")
        return [(64, 8, "affine"), (32, 4, "mxfp4"), (32, 8, "mxfp8")], True
    layouts = quantization_layouts(config, loader)
    quant = config.get("quantization") or {}
    mixed = isinstance(quant, dict) and any(key not in ("group_size", "bits", "mode") for key in quant)
    return layouts, mixed


def check_quantization(config, loader, root, mx):
    if config.get("quantize_activations"):
        raise ValueError("unverified oMLX activation quantization: validating eligible modes and linear biases requires model construction")
    layouts, mixed = omlx_quantization_layouts(config, loader, root)
    if not layouts:
        return
    verify_quantization_loader(loader, mixed, [mode for _, _, mode in layouts])
    nn = installed_module("mlx.nn", root)
    if mixed:
        verify_nn_dictionary_dispatch(nn.quantize)
    for group, bits, mode in layouts:
        kwargs = {"group_size": group, "bits": bits}
        if mode != "affine" or "mode" in inspect.signature(nn.quantize).parameters:
            kwargs["mode"] = mode
        try:
            inspect.signature(nn.quantize).bind(None, **kwargs)
            mx.eval(mx.quantize(mx.zeros((1, max(32, group))), **kwargs))
        except Exception as error:
            raise ValueError(f"incompatible oMLX quantization {mode}/{bits}-bit/group-{group}: {error}") from error


def supports_tools(config, tokenizer_config, root, utils):
    if config["model_type"] == "qwen4_exp":
        if tokenizer_config.get("chat_template_type") is not None:
            return False
        tokenizer = installed_module("mlx_lm.tokenizer_utils", root)
        verify_contract(utils.load_tokenizer, "native_tokenizer_loader")
        native_load = tokenizer.load
        if ast_digest(function_tree(native_load)) in CONTRACTS["deepseek_tokenizer"]:
            # A previously imported native DeepSeek patch delegates other
            # architectures to its original tokenizer loader.
            native_load = inspect.getclosurevars(native_load).nonlocals.get("orig_load")
            check_function_origin(native_load, root)
        verify_contract(native_load, "native_tokenizer")
        verify_contract(tokenizer._infer_tool_parser, "native_tool_inference")
        if getattr(utils, "_load_tokenizer", None) is not tokenizer.load:
            return False
        # Match the local Transformers template precedence without constructing
        # a tokenizer. Multiple named templates need a separate verified path.
        if (root / "chat_templates").exists():
            return False
        template = tokenizer_config.get("chat_template")
        template_file = root / "chat_template.jinja"
        if template_file.exists():
            with template_file.open(encoding="utf-8") as source:
                template = source.read(MAX_JSON_BYTES + 1)
            if len(template) > MAX_JSON_BYTES:
                return False
        inferred = tokenizer._infer_tool_parser(template)
        selected = tokenizer_config.get("tool_parser_type", inferred)
        if inferred != "qwen3_coder" or selected != inferred:
            return False
        parser = installed_module("mlx_lm.tool_parsers.qwen3_coder", root)
        verify_contract(parser, "qwen_coder_parser_module")
        for node in function_tree(parser).body:
            if isinstance(node, ast.FunctionDef):
                function = getattr(parser, node.name, None)
                expected = ast.Module(body=[node], type_ignores=[])
                if (getattr(function, "__globals__", None) is not parser.__dict__
                        or ast_digest(function_tree(function)) != ast_digest(expected)):
                    return False
        return (parser.tool_call_start == "<tool_call>"
                and parser.tool_call_end == "</tool_call>")
    if config["model_type"].startswith("deepseek_v4"):
        if (tokenizer_config.get("tool_parser_type") not in (None, "deepseek_v4")
                or tokenizer_config.get("chat_template_type") not in (None, "deepseek_v4")):
            return False
        tokenizer = installed_module("mlx_lm.tokenizer_utils", root)
        verify_contract(tokenizer.load, "deepseek_tokenizer")
        if getattr(utils, "_load_tokenizer", None) is not tokenizer.load:
            return False
        parser = installed_module("mlx_lm.tool_parsers.deepseek_v4", root)
        verify_contract(parser.parse_tool_call, "deepseek_parser")
        return bool(getattr(parser, "tool_call_start", None) and getattr(parser, "tool_call_end", None))
    # Other architectures remain text-only until tokenizer wiring is verified.
    return False


def probe_cache_paths():
    """Files and import directories whose changes invalidate this preflight.

    Only paths are returned. Werk compares filesystem metadata without reading
    weights or importing Python again for each request. Search directories and
    .pth files cover new shadowing modules and installation changes as well as
    edits to the modules actually imported by this successful probe.
    """
    paths = {Path(sys.executable).absolute()}
    for module in tuple(sys.modules.values()):
        namespace = getattr(module, "__dict__", {})
        filename = namespace.get("__file__")
        if isinstance(filename, str) and Path(filename).exists():
            path = Path(filename).absolute()
            paths.add(path)
            paths.add(path.parent)
        for directory in namespace.get("__path__", ()) or ():
            if isinstance(directory, str) and Path(directory).is_dir():
                paths.add(Path(directory).absolute())
    for directory in sys.path:
        if not isinstance(directory, str):
            continue
        root = Path(directory).absolute()
        if root.exists():
            paths.add(root)
        if root.is_dir():
            paths.update(root.glob("*.pth"))
        elif root.parent.is_dir():
            # Python also searches zip archives, including paths which do not
            # yet exist. Their parent detects a newly installed archive.
            paths.add(root.parent)
    for name in ("omlx", "mlx", "mlx-lm"):
        distribution = importlib.metadata.distribution(name)
        found_metadata = False
        for entry in distribution.files or ():
            if entry.name == "METADATA" and str(entry.parent).endswith(".dist-info"):
                paths.add(Path(distribution.locate_file(entry)).absolute())
                found_metadata = True
        if not found_metadata:
            # Homebrew omits RECORD, making Distribution.files return None.
            # Its standard PathDistribution still exposes the metadata folder.
            metadata_root = getattr(distribution, "_path", None)
            if metadata_root is None:
                raise ValueError("runtime package metadata path is unavailable")
            metadata_root = Path(metadata_root)
            metadata_files = [metadata_root / filename for filename in ("METADATA", "PKG-INFO")]
            metadata_files = [path.absolute() for path in metadata_files if path.is_file()]
            if not metadata_files:
                raise ValueError("runtime package metadata file is unavailable")
            paths.update(metadata_files)
    return sorted(map(str, paths))


def add_probe_cache_paths(result):
    """Optional optimization metadata must fit the bounded JSON transport."""
    try:
        paths = probe_cache_paths()
        if len(paths) <= 8192 and len(json.dumps(paths).encode("utf-8")) <= 2 * 1024 * 1024:
            result["cache_paths"] = paths
    except Exception:
        pass


def probe(payload):
    if not isinstance(payload, dict):
        raise ValueError("oMLX preflight input must be an object")
    launcher = payload.get("launcher")
    if not isinstance(launcher, str) or launcher_module(launcher) != "omlx.cli":
        raise ValueError("unverified oMLX launcher: expected installed omlx.cli console entry point")
    root = None
    config = tokenizer_config = None
    if payload.get("model_dir") is not None:
        root = Path(payload["model_dir"]).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("oMLX model path must be a local directory")
        config = read_json(root / "config.json", installed_text_port=True)
        if not config.get("model_type"):
            raise ValueError("damaged model metadata: config.json requires model_type")
        tokenizer_config = read_json(root / "tokenizer_config.json", required=False)
        auto_map = tokenizer_config.get("auto_map")
        if auto_map:
            raise ValueError("custom tokenizer auto_map requires model repository code; unverified by oMLX preflight")
        read_json(root / "generation_config.json", required=False)
        check_headers(root)
    # Match the console-script import location, excluding the current model/repo
    # directory that python -c would otherwise prepend.
    if not getattr(sys.flags, "safe_path", False):
        sys.path[0] = str(Path(launcher).resolve().parent)
    # Rust rejects model-local Python environment paths before Python starts,
    # including startup hooks. Preserve the accepted console import environment.
    omlx = installed_module("omlx", root)
    mlx_lm = installed_module("mlx_lm", root)
    mx = installed_module("mlx.core", root)
    runtime = {
        "omlx_version": version("omlx", omlx),
        "mlx_version": version("mlx", mx),
        "mlx_lm_version": version("mlx-lm", mlx_lm),
        # Keep the venv executable spelling; resolving its symlink loses venv identity.
        "python": os.path.abspath(sys.executable),
    }
    detail = (f"oMLX {runtime['omlx_version']}, mlx-lm {runtime['mlx_lm_version']}, "
              f"mlx {runtime['mlx_version']}, Python {runtime['python']}")
    if not getattr(getattr(mx, "metal", None), "is_available", lambda: False)():
        raise ValueError(f"MLX Metal device is unavailable ({detail})")
    result = {"ok": True, "detail": detail, "runtime": runtime, "supports_tool_calling": False}
    if root is not None:
        text_offload_requested = (payload.get("expert_cache_bytes") is not None or payload.get("ngram_cache_bytes") is not None)
        # Qwen's default/explicit N-gram Auto also works with native experts.
        # Other architectures and unverified runtime versions keep their route.
        text_offload_requested |= config.get("model_type") == "qwen4_exp" and payload.get("ngram_cache_bytes") is None
        text_offload_explicit = bool(payload.get("expert_cache_bytes") or payload.get("ngram_cache_bytes"))
        if (config.get("model_type") in ("qwen4_exp", "glm5_next") and text_offload_requested
                and (runtime["omlx_version"] == "0.6.4" or text_offload_explicit)):
            from _werk_omlx_text_offload import inspect_model
            result["runtime"]["expert_offload"] = inspect_model(root, payload.get("expert_cache_bytes"), payload.get("ngram_cache_bytes"))
            result["model_type"] = config["model_type"]
            try:
                utils = installed_module("mlx_lm.utils", root)
                result["supports_tool_calling"] = supports_tools(config, tokenizer_config, root, utils)
                if not result["supports_tool_calling"]:
                    result["tool_calling_detail"] = "native text offload tokenizer/tool parser wiring is unverified for this model"
            except Exception as error:
                result["tool_calling_detail"] = f"native text offload tool parser is unverified: {error}"
            add_probe_cache_paths(result)
            return result
        if payload.get("ngram_cache_bytes"):
            raise ValueError("this architecture has no verified N-gram offload adapter")
        try:
            config, utils, loader = prepare_runtime(root, config)
            check_quantization(config, loader, root, mx)
        except Exception as error:
            raise ValueError(f"{error} ({detail})") from error
        try:
            result["supports_tool_calling"] = supports_tools(config, tokenizer_config, root, utils)
            if not result["supports_tool_calling"]:
                result["tool_calling_detail"] = "installed oMLX tokenizer/tool parser wiring is unverified for this model"
        except Exception as error:
            result["tool_calling_detail"] = f"oMLX tool parser is unverified: {error}"
        result["model_type"] = config["model_type"]
        expert_budget = payload.get("expert_cache_bytes")
        # Auto applies only to the verified adapter. Preserve native loading
        # for other architectures, runtime versions and quantization layouts.
        auto_eligible = (runtime["omlx_version"] == "0.6.4"
                         and config.get("model_type") == "deepseek_v4"
                         and not config.get("model_file")
                         and isinstance(config.get("quantization"), dict)
                         and config["quantization"].get("mode", "affine") == "affine")
        if expert_budget is not None and (expert_budget > 0 or auto_eligible):
            if runtime["omlx_version"] != "0.6.4":
                raise ValueError("experimental expert streaming currently requires oMLX 0.6.4")
            from _werk_omlx_experts import inspect_model
            result["runtime"]["expert_offload"] = inspect_model(root, payload["expert_cache_bytes"])
    # A missing dependency inventory disables the Rust-side optimization; it
    # must never make an otherwise valid runtime incompatible.
    add_probe_cache_paths(result)
    return result


def main():
    try:
        data = sys.stdin.buffer.read(65537) if hasattr(sys.stdin, "buffer") else sys.stdin.read(65537)
        if len(data) > 65536:
            raise ValueError("oMLX preflight input exceeds 64 KiB")
        # Installed imports/patches may log to stdout; preserve one JSON result.
        with contextlib.redirect_stdout(sys.stderr):
            result = probe(json.loads(data))
    except Exception as error:
        result = {"ok": False, "detail": str(error), "supports_tool_calling": False}
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
