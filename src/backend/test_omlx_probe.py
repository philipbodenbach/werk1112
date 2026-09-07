"""Deterministic installed-oMLX simulations; no MLX install or model weights.

Executable fixture excerpts are from jundot/omlx v0.6.4 (Apache-2.0) and its
mlx-lm pin ab1806e8f5d6aa035973af194a1b9198ab4754dc. Comments/docstrings were
removed by AST parsing, but executable bodies, signatures, and decorators are
unchanged. The actual dispatch, configuration wrapper, DeepSeek patch, loader,
resolver and ModelArgs paths below run against small instrumented modules.
Model/tokenizer constructors and all weight loading deliberately fail.
https://github.com/jundot/omlx/tree/v0.6.4/omlx
https://github.com/ml-explore/mlx-lm/blob/ab1806e8f5d6aa035973af194a1b9198ab4754dc/mlx_lm/utils.py
"""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("werk_omlx_probe", Path(__file__).with_name("omlx_probe.py"))
probe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_module)

# Readable installed-source fixtures: these are never copied into model folders.
STOCK = {
    "load_model": r'''
def load_model(model_path: Path, lazy: bool=False, strict: bool=True, model_config: Optional[Dict[str, Any]]=None, get_model_classes: Callable[[dict], Tuple[Type[nn.Module], Type]]=_get_classes, trust_remote_code: bool=False) -> Tuple[nn.Module, dict]:
    config = load_config(model_path)
    if model_config is not None:
        config.update(model_config)
    weight_files = glob.glob(str(model_path / 'model*.safetensors'))
    if not weight_files and strict:
        raise FileNotFoundError(f'No safetensors found in {model_path}')
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))
    if (model_file := config.get('model_file')) is not None:
        if not trust_remote_code:
            raise ValueError(f'The model at {model_path} requires importing and running a custom module ({model_file!r}) to build its architecture. This is disabled by default. Pass trust_remote_code=True if you trust this model.')
        spec = importlib.util.spec_from_file_location('custom_model', model_path / model_file)
        arch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arch)
        model_class, model_args_class = (arch.Model, arch.ModelArgs)
    else:
        model_class, model_args_class = get_model_classes(config=config)
    if 'quantization_config' not in config:
        text_config = config.get('text_config', {})
        if 'quantization_config' in text_config:
            config['quantization_config'] = text_config['quantization_config']
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)
    if hasattr(model, 'sanitize'):
        weights = model.sanitize(weights)

    def _quantize(quantization):

        def class_predicate(p, m):
            if p in config['quantization']:
                return config['quantization'][p]
            if not hasattr(m, 'to_quantized'):
                return False
            return f'{p}.scales' in weights
        nn.quantize(model, group_size=quantization['group_size'], bits=quantization['bits'], mode=quantization.get('mode', 'affine'), class_predicate=class_predicate)
    if (quantization := config.get('quantization', None)) is not None:
        _quantize(quantization)
    elif (quantization_config := config.get('quantization_config', False)):
        quant_method = quantization_config['quant_method']
        if quant_method == 'bitnet':
            from .models.bitlinear_layers import bitnet_quantize
            model = bitnet_quantize(model, quantization_config)
        elif quant_method == 'mxfp4':
            quantization = {'group_size': 32, 'bits': 4, 'mode': 'mxfp4'}
            config['quantization'] = quantization
            config['quantization_config'] = quantization
            _quantize(quantization)
        elif quant_method == 'compressed-tensors':
            quantization = {'group_size': 32, 'bits': 4, 'mode': 'affine'}
            config['quantization'] = quantization
            config['quantization_config'] = quantization
            _quantize(quantization)
        elif quant_method in ('awq', 'gptq'):
            weights, quantization = _transform_awq_weights(weights, quantization_config)
            config['quantization'] = quantization
            config['quantization_config'] = quantization
            _quantize(quantization)
    if config.get('quantize_activations', False):

        def _maybe_qq(m):
            if isinstance(m, nn.QuantizedLinear):
                if m.mode not in ('nvfp4', 'mxfp8'):
                    raise ValueError('Mode ({m.mode}) does not support activation quantization')
                if m.get('bias', False):
                    raise ValueError('Linear layer with bias does not support activation quantization')
                out_dims, in_dims = m.weight.shape
                in_dims *= 32 // m.bits
                return nn.QQLinear(in_dims, out_dims, m.group_size, m.bits, m.mode)
            else:
                return m
        leaves = tree_map(_maybe_qq, model.leaf_modules(), is_leaf=nn.Module.is_module)
        model.update_modules(leaves)
    model.eval()
    model.load_weights(list(weights.items()), strict=strict)
    if not lazy:
        mx.eval(model.parameters())
    return (model, config)
''',
    "lm_load_compat": r'''
def lm_load_compat(path_or_repo: str, *, trust_remote_code: bool=False, **kwargs):
    preflight_text_remote_code(path_or_repo, tokenizer_config=kwargs.get('tokenizer_config'), trust_remote_code=trust_remote_code)
    from mlx_lm import load
    if _LM_LOAD_ACCEPTS_TRC:
        kwargs['trust_remote_code'] = trust_remote_code
    return load(path_or_repo, **kwargs)
''',
    "load_text_model": r'''
def load_text_model(model_name: str, tokenizer_config: dict[str, Any] | None=None, model_settings: Any | None=None):
    maybe_apply_pre_load_patches(model_name, model_settings=model_settings)
    trust_remote_code = bool(getattr(model_settings, 'trust_remote_code', False)) if model_settings is not None else False
    return lm_load_compat(model_name, tokenizer_config=tokenizer_config, trust_remote_code=trust_remote_code)
''',
    "_patch_mlx_lm_load_config": r'''
def _patch_mlx_lm_load_config() -> None:
    global _MLX_LM_LOAD_CONFIG_PATCHED
    if _MLX_LM_LOAD_CONFIG_PATCHED:
        return
    try:
        import mlx_lm.utils as _lu
    except ImportError:
        return
    _original = _lu.load_config

    def _patched(model_path, *args, **kwargs):
        cfg = _original(model_path, *args, **kwargs)
        normalize_hy_v3_rope_config(cfg)
        expand_per_layer_quant_keys(cfg)
        expand_glm_moe_dsa_fused_quant_keys(cfg)
        normalize_laguna_compressed_quant(cfg)
        normalize_bailing_hybrid_fp8_quant(cfg)
        return cfg
    _lu.load_config = _patched
    _MLX_LM_LOAD_CONFIG_PATCHED = True
''',
    "maybe_apply_pre_load_patches": r'''
def maybe_apply_pre_load_patches(model_name: str, model_settings: Any | None=None, for_vlm: bool=False) -> None:
    from ..patches.mlx_lm_mtp import set_mtp_active
    set_mtp_active(False)
    _patch_mlx_lm_load_config()
    from ..patches.m5_gather_qmm import apply_m5_gather_qmm_workaround
    if apply_m5_gather_qmm_workaround():
        logger.info('M5 sorted gather_qmm reroute installed (issue #2267)')
    from ..patches.arrays_cache_extract import apply_arrays_cache_extract_guard
    apply_arrays_cache_extract_guard()
    config_path = Path(model_name) / 'config.json'
    if not config_path.exists():
        return
    try:
        config = json.loads(config_path.read_text())
    except Exception as e:
        logger.debug('Could not read %s for pre-load patch dispatch: %s', config_path, e)
        return
    quant_cfg = config.get('quantization') or {}
    quant_bits = quant_cfg.get('bits') if isinstance(quant_cfg, dict) else None
    if quant_bits in (1, 2):
        try:
            from ..patches.bonsai_t5_load import apply_bonsai_t5_load_patch
        except Exception as e:
            logger.debug('bonsai t5 load patch import failed: %s', e)
        else:
            if apply_bonsai_t5_load_patch():
                logger.info('Bonsai t5 load patch applied for %s (t5 uint8 weights allowed past strict shape check)', model_name)
    if quant_bits == 1:
        try:
            from ..patches.bonsai_qmv import apply_bonsai_construct_patch
        except Exception as e:
            logger.debug('bonsai construct patch import failed: %s', e)
        else:
            if apply_bonsai_construct_patch():
                logger.info('Bonsai 1-bit construct patch applied for %s', model_name)
    model_type = config.get('model_type')
    if isinstance(model_type, str) and model_type.startswith('deepseek_v4'):
        from ..patches.deepseek_v4 import apply_deepseek_v4_patch
        if apply_deepseek_v4_patch():
            logger.info('DeepSeek V4 pre-load patch applied for %s', model_name)
    if model_type == 'step3p7':
        from ..patches.step3p7 import apply_step3p7_patch
        if apply_step3p7_patch():
            logger.info('Step 3.7 pre-load patch applied for %s', model_name)
    if model_type == 'mimo_v2':
        from ..patches.mimo_v2 import apply_mimo_v2_patch
        if apply_mimo_v2_patch():
            logger.info('MiMo V2.5 text pre-load patch applied for %s', model_name)
    if model_type == 'bailing_hybrid':
        from ..patches.bailing_hybrid import apply_bailing_hybrid_patch
        if apply_bailing_hybrid_patch():
            logger.info('Ling 3.0 Flash pre-load patch applied for %s', model_name)
    if model_type == 'laguna':
        from ..patches.laguna import apply_laguna_patch
        if apply_laguna_patch():
            logger.info('Laguna pre-load patch applied for %s', model_name)
    if model_type == 'hy_v3':
        from ..patches.hy_v3 import apply_hy_v3_patch
        if apply_hy_v3_patch():
            logger.info('Hy3 pre-load patch applied for %s', model_name)
    text_config = config.get('text_config')
    text_model_type = text_config.get('model_type') if isinstance(text_config, dict) else None
    if model_type == 'llama4' or text_model_type == 'llama4':
        from ..patches.llama4_attention import apply_llama4_attention_patch
        if apply_llama4_attention_patch():
            logger.info('Llama 4 attention patch applied for %s', model_name)
    if model_type == 'glm_moe_dsa':
        from ..patches.glm_moe_dsa import apply_glm_moe_dsa_patch
        if apply_glm_moe_dsa_patch():
            logger.info('GLM MoE DSA pre-load patch applied for %s', model_name)
    minimax_m3_types = {'minimax_m3', 'minimax_m3_vl'}
    if not for_vlm and (model_type in minimax_m3_types or text_model_type in minimax_m3_types):
        from ..patches.minimax_m3_mlx_lm import apply_minimax_m3_mlx_lm_patch
        if apply_minimax_m3_mlx_lm_patch():
            logger.info('MiniMax-M3 mlx-lm registration applied for %s', model_name)
    if for_vlm and (model_type in minimax_m3_types or text_model_type in minimax_m3_types):
        from ..patches.mlx_vlm_minimax_m3_compat import apply_mlx_vlm_minimax_m3_compat_patch
        if apply_mlx_vlm_minimax_m3_compat_patch():
            logger.info('MiniMax M3 mlx-vlm compatibility patch applied for %s', model_name)
        from ..patches.minimax_m3_sparse_attention import apply_minimax_m3_sparse_attention_patch
        if apply_minimax_m3_sparse_attention_patch():
            logger.info('MiniMax M3 sparse attention patch applied for %s', model_name)
    if for_vlm and model_type == 'unlimited-ocr':
        from ..patches.mlx_vlm_unlimited_ocr_compat import apply_mlx_vlm_unlimited_ocr_compat_patch
        if apply_mlx_vlm_unlimited_ocr_compat_patch():
            logger.info('Unlimited-OCR mlx-vlm compatibility patch applied for %s', model_name)
    if for_vlm and model_type in ('inkling', 'inkling_mm_model'):
        from ..patches.mlx_vlm_inkling_compat import apply_mlx_vlm_inkling_compat_patch
        if apply_mlx_vlm_inkling_compat_patch():
            logger.info('Inkling mlx-vlm compatibility patch applied for %s', model_name)
    if for_vlm and model_type == 'muse_glimmer':
        from ..patches.mlx_vlm_muse_glimmer_compat import apply_mlx_vlm_muse_glimmer_compat_patch
        if apply_mlx_vlm_muse_glimmer_compat_patch():
            logger.info('Muse Glimmer mlx-vlm compatibility patch applied for %s', model_name)
    if for_vlm and model_type == 'qwen4_exp':
        from ..patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch, configure_qwen4_exp_runtime
        if apply_mlx_vlm_qwen4_exp_compat_patch():
            logger.info('Qwen4-Exp mlx-vlm compatibility patch applied for %s', model_name)
        mtp_requested = bool(model_settings is not None and getattr(model_settings, 'mtp_enabled', False))
        has_mtp_weights = _checkpoint_has_mtp_weights(model_name)
        mtp_active = mtp_requested and has_mtp_weights
        if mtp_requested and (not has_mtp_weights):
            logger.warning('Qwen4-Exp Lightning MTP was requested for %s, but no embedded MTP tensors were found', model_name)
        from ..patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, set_mtp_active, set_mtp_depth
        set_mtp_active(mtp_active)
        depth = getattr(model_settings, 'mtp_num_draft_tokens', None) if model_settings is not None else None
        set_mtp_depth(int(depth) if depth else 3)
        if mtp_active and (not apply_mlx_lm_mtp_patch()):
            logger.warning('Qwen4-Exp Lightning MTP dispatch patch failed for %s; speculative decoding will remain inactive', model_name)
            set_mtp_active(False)
            mtp_active = False
        configure_qwen4_exp_runtime(model_name, mode='mmap' if model_settings is not None and getattr(model_settings, 'qwen4_ple_ssd_offload', False) else 'resident' if model_settings is not None else None, mtp_enabled=mtp_active)
    if for_vlm and model_type == 'glm5_next':
        from ..patches.mlx_vlm_glm5_next_compat import apply_mlx_vlm_glm5_next_compat_patch
        if apply_mlx_vlm_glm5_next_compat_patch():
            logger.info('GLM-5.3 mlx-vlm compatibility patch applied for %s', model_name)
    if _is_mtp_compatible(config, model_type):
        mtp_enabled = bool(model_settings is not None and getattr(model_settings, 'mtp_enabled', False))
        from ..patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, set_mtp_active, set_mtp_depth
        if apply_mlx_lm_mtp_patch():
            set_mtp_active(mtp_enabled)
            depth = getattr(model_settings, 'mtp_num_draft_tokens', None)
            if depth:
                set_mtp_depth(int(depth))
            elif model_type.startswith('nemotron_h'):
                set_mtp_depth(1)
            elif model_type in ('gemma4', 'gemma4_unified'):
                set_mtp_depth(8)
            elif model_type in ('inkling', 'inkling_mm_model'):
                mtp_cfg = config.get('mtp_config') or {}
                set_mtp_depth(int(mtp_cfg.get('num_nextn_predict_layers', 0) or 0) or 3)
            else:
                set_mtp_depth(3)
            if mtp_enabled:
                backend = 'embedded DSpark' if _has_dspark_heads(config) else 'Lightning MTP'
                logger.info('Speculative backend selected for %s: %s (model_type=%s, active)', model_name, backend, model_type)
            else:
                logger.debug('Native MTP patch applied for %s for sanitize correctness (model has MTP heads but mtp_enabled=False; head not attached)', model_name)
        if for_vlm:
            try:
                from ..patches.mlx_vlm_mtp import apply_mlx_vlm_mtp_patch, apply_mlx_vlm_mtp_runtime_patch, set_mtp_attach_enabled
            except Exception:
                pass
            else:
                has_mtp_weights = _checkpoint_has_mtp_weights(model_name)
                set_mtp_attach_enabled(has_mtp_weights)
                if apply_mlx_vlm_mtp_patch():
                    if mtp_enabled:
                        logger.info('mlx-vlm MTP sanitize patch applied for %s', model_name)
                    else:
                        logger.debug('mlx-vlm MTP sanitize patch applied for %s (mtp_enabled=False; allows persisted mtp.* weights to bind)', model_name)
                if apply_mlx_vlm_mtp_runtime_patch():
                    if not has_mtp_weights:
                        logger.info('mlx-vlm runtime MTP patch applied for %s (config declares mtp heads but checkpoint ships no mtp.* weights; MTPModule attachment skipped to keep strict load_weights happy)', model_name)
                    elif mtp_enabled:
                        logger.info('mlx-vlm runtime MTP patch applied for %s', model_name)
                    else:
                        logger.debug('mlx-vlm runtime MTP patch applied for %s (mtp_enabled=False; head attached for weight load only)', model_name)
    elif model_type != 'qwen4_exp' and model_settings is not None and getattr(model_settings, 'mtp_enabled', False):
        logger.warning('mtp_enabled=True for %s but model is incompatible (model_type=%r, mtp_heads=%s); MTP path will be inactive', model_name, model_type, _has_mtp_heads(config))
    if for_vlm and model_type and model_type.startswith('qwen3_5_moe') and (not _is_mtp_compatible(config, model_type)):
        try:
            from ..patches.mlx_vlm_mtp import apply_mlx_vlm_mtp_patch
        except Exception as e:
            logger.debug('qwen3_6 MoE VLM sanitize patch import failed: %s', e)
        else:
            if apply_mlx_vlm_mtp_patch():
                logger.debug('mlx-vlm qwen3_6 MoE VLM sanitize patch applied for %s (no MTP heads; switch_mlp load correctness)', model_name)
    if for_vlm and model_type and model_type.startswith('qwen3_5_moe'):
        try:
            from ..patches.qwen3_6_nested_visual import apply_qwen3_6_nested_visual_patch
        except Exception as e:
            logger.debug('qwen3_6 nested-visual patch import failed: %s', e)
        else:
            if apply_qwen3_6_nested_visual_patch():
                logger.info('qwen3_6 nested-visual sanitize wrap applied for %s', model_name)
    quant_cfg = config.get('quantization') or {}
    quant_bits = quant_cfg.get('bits') if isinstance(quant_cfg, dict) else None
    if quant_bits in (1, 2):
        try:
            from ..patches.bonsai_qmv import apply_bonsai_qmv_patch
        except Exception as e:
            logger.debug('bonsai qmv patch import failed: %s', e)
        else:
            if apply_bonsai_qmv_patch():
                logger.info('Bonsai %d-bit qmv decode patch applied for %s', quant_bits, model_name)
            else:
                logger.debug('Bonsai qmv patch skipped for %s (native extension not available; stock mlx fallback active)', model_name)
''',
    "apply_deepseek_v4_patch": r'''
def apply_deepseek_v4_patch() -> bool:
    global _APPLIED
    if _APPLIED:
        return False
    try:
        import mlx_lm
    except ImportError:
        logger.debug('mlx_lm not importable — deepseek_v4 patch skipped')
        return False
    apply_pooling_cache_support()
    _register_module('mlx_lm.models.hyper_connection', 'hyper_connection.py')
    _register_module('mlx_lm.models.deepseek_v4', 'deepseek_v4_model.py')
    _register_model_type_aliases()
    from .utils_patch import apply_utils_patch
    apply_utils_patch()
    from .tokenizer_patch import apply_load_patch, apply_tokenizer_patch
    apply_tokenizer_patch()
    apply_load_patch()
    _probe_native_indexer_kernels()
    from omlx.patches.mlx_lm_sharded_load import install_local_sharded_load_fallback
    install_local_sharded_load_fallback()
    _APPLIED = True
    logger.info('DeepSeek V4 patch applied (PR 1192 head %s)', PR_HEAD_SHA[:8])
    return True
''',
    "_build_patched_load_model": r'''
def _build_patched_load_model() -> Callable:
    default_get_classes = _utils._get_classes

    def patched_load_model(model_path: Path, lazy: bool=False, strict: bool=True, model_config: dict[str, Any] | None=None, get_model_classes: Callable=default_get_classes, trust_remote_code: bool=False) -> tuple[nn.Module, dict]:
        config = _utils.load_config(model_path)
        if model_config is not None:
            config.update(model_config)
        if (model_file := config.get('model_file')) is not None and (not trust_remote_code):
            raise ValueError(f'The model at {model_path} requires executing custom model code ({model_file!r}). Pass trust_remote_code=True if you trust this model.')
        weight_files = glob.glob(str(model_path / 'model*.safetensors'))
        if not weight_files and strict:
            raise FileNotFoundError(f'No safetensors found in {model_path}')
        weights = {}
        for wf in weight_files:
            weights.update(_load_safetensors(wf))
        if model_file is not None:
            spec = importlib.util.spec_from_file_location('custom_model', model_path / model_file)
            arch = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(arch)
            model_class, model_args_class = (arch.Model, arch.ModelArgs)
        else:
            model_class, model_args_class = get_model_classes(config=config)
        if 'quantization_config' not in config:
            text_config = config.get('text_config', {})
            if 'quantization_config' in text_config:
                config['quantization_config'] = text_config['quantization_config']
        if str(config.get('model_type', '')).startswith('deepseek_v4'):
            config['use_native_ratio128_attention'] = bool(config.get('use_native_ratio128_attention', True)) and _native_ratio128_attention_enabled(config)
        model_args = model_args_class.from_dict(config)
        model = model_class(model_args)
        if hasattr(model, 'sanitize'):
            weights = model.sanitize(weights)

        def _quantize(quantization):

            def class_predicate(p, m):
                if p in config['quantization']:
                    return config['quantization'][p]
                if not hasattr(m, 'to_quantized'):
                    return False
                return f'{p}.scales' in weights
            nn.quantize(model, group_size=quantization['group_size'], bits=quantization['bits'], mode=quantization.get('mode', 'affine'), class_predicate=class_predicate)
        if (quantization := config.get('quantization', None)) is not None:
            _quantize(quantization)
        elif (quantization_config := config.get('quantization_config', False)):
            quant_method = quantization_config['quant_method']
            if quant_method == 'bitnet':
                from mlx_lm.models.bitlinear_layers import bitnet_quantize
                model = bitnet_quantize(model, quantization_config)
            elif quant_method == 'mxfp4':
                quantization = {'group_size': 32, 'bits': 4, 'mode': 'mxfp4'}
                config['quantization'] = quantization
                config['quantization_config'] = quantization
                _quantize(quantization)
            elif quant_method == 'compressed-tensors':
                quantization = {'group_size': 32, 'bits': 4, 'mode': 'affine'}
                config['quantization'] = quantization
                config['quantization_config'] = quantization
                _quantize(quantization)
            elif quant_method in ('awq', 'gptq'):
                weights, quantization = _utils._transform_awq_weights(weights, quantization_config)
                config['quantization'] = quantization
                config['quantization_config'] = quantization
                _quantize(quantization)
            elif quant_method == 'fp8' and str(config.get('model_type', '')).startswith('deepseek_v4'):
                from mlx_lm.models.deepseek_v4 import make_quantization_config
                quantization = make_quantization_config(model)
                config['quantization'] = quantization
                config['quantization_config'] = quantization
                _quantize(quantization)
        if config.get('quantize_activations', False):

            def _maybe_qq(m):
                if isinstance(m, nn.QuantizedLinear):
                    if m.mode not in ('nvfp4', 'mxfp8'):
                        raise ValueError(f'Mode ({m.mode}) does not support activation quantization')
                    if m.get('bias', False):
                        raise ValueError('Linear layer with bias does not support activation quantization')
                    out_dims, in_dims = m.weight.shape
                    in_dims *= 32 // m.bits
                    return nn.QQLinear(in_dims, out_dims, m.group_size, m.bits, m.mode)
                return m
            leaves = tree_map(_maybe_qq, model.leaf_modules(), is_leaf=nn.Module.is_module)
            model.update_modules(leaves)
        model.eval()
        model.load_weights(list(weights.items()), strict=strict)
        if not lazy:
            mx.eval(model.parameters())
        return (model, config)
    return patched_load_model
''',
    "apply_load_patch": r'''
def apply_load_patch() -> bool:
    global _LOAD_PATCHED
    if _LOAD_PATCHED:
        return False
    _register_chat_template_and_parser_modules()
    orig_load = _tu.load

    def patched_load(model_path, tokenizer_config_extra=None, eos_token_ids=None):
        wrapper = orig_load(model_path, tokenizer_config_extra=tokenizer_config_extra, eos_token_ids=eos_token_ids)
        if not _is_deepseek_v4_model(model_path):
            return wrapper
        from . import chat_template_v4 as _ct
        from . import tool_parser_v4 as _tp
        if wrapper._chat_template is None:
            wrapper._chat_template = _ct.apply_chat_template
            wrapper.has_chat_template = True
        if wrapper._chat_template is _ct.apply_chat_template:
            wrapper._omlx_supports_mid_system_messages = _ct.supports_mid_system_messages
            wrapper._omlx_relocate_mid_system_messages = _ct.relocate_mid_system_messages
        if wrapper._tool_parser is None:
            wrapper._tool_parser = _tp.parse_tool_call
            wrapper._tool_call_start = _tp.tool_call_start
            wrapper._tool_call_end = _tp.tool_call_end
            try:
                wrapper._tool_call_start_tokens = tuple(wrapper._tokenizer.encode(_tp.tool_call_start, add_special_tokens=False))
                wrapper._tool_call_end_tokens = tuple(wrapper._tokenizer.encode(_tp.tool_call_end, add_special_tokens=False))
            except Exception as e:
                logger.warning('Could not encode DSML tool-call markers as tokens: %s', e)
        logger.info('Injected DeepSeek V4 DSML chat_template + tool_parser into TokenizerWrapper for %s', model_path)
        return wrapper
    _tu.load = patched_load
    try:
        import mlx_lm.utils as _mu
        if hasattr(_mu, '_load_tokenizer'):
            _mu._load_tokenizer = patched_load
    except Exception as e:
        logger.warning('Could not patch mlx_lm.utils._load_tokenizer (V4 chat_template injection may not fire): %s', e)
    _LOAD_PATCHED = True
    logger.info('mlx_lm.tokenizer_utils.load wrapped (injects DSML chat_template + tool_parser for deepseek_v4)')
    return True
''',
    "ModelArgs": r'''
@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = 'deepseek_v4'
    vocab_size: int = 129280
    hidden_size: int = 4096
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 1.5
    q_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    num_experts_per_tok: int = 6
    norm_topk_prob: bool = True
    hidden_act: str = 'silu'
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-06
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    head_dim: int = 512
    scoring_func: str = 'sqrtsoftplus'
    compress_ratios: List[int] = field(default_factory=list)
    compress_rope_theta: float = 160000.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-06
    num_hash_layers: int = 3
    swiglu_limit: float = 10.0
    sliding_window: int = 128
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    num_nextn_predict_layers: int = 1
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: List[int] = field(default_factory=list)
    dspark_markov_rank: int = 256
    n_mtp_layers: int = 0
    tie_word_embeddings: bool = False
    topk_method: str = 'noaux_tc'
    use_native_ratio128_attention: bool = True

    def __post_init__(self):
        if not self.compress_ratios:
            n = self.num_hidden_layers
            self.compress_ratios = [0] + [4 if i % 2 else 128 for i in range(max(n - 2, 0))] + ([0] if n >= 2 else [])
        self.compress_ratios = list(self.compress_ratios[:self.num_hidden_layers])
        if len(self.compress_ratios) != self.num_hidden_layers:
            raise ValueError(f'`compress_ratios` must have one entry per hidden layer, got {len(self.compress_ratios)} for {self.num_hidden_layers} layers.')
        bad = [r for r in self.compress_ratios if r not in (0, 4, 128)]
        if bad:
            raise ValueError(f'Unsupported DeepSeek-V4 compress ratios: {bad}')
''',
    "make_quantization_config": r'''
def make_quantization_config(model):
    mxfp4 = {'group_size': 32, 'bits': 4, 'mode': 'mxfp4'}
    mxfp8 = {'group_size': 32, 'bits': 8, 'mode': 'mxfp8'}
    flat_modules = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    experts = {k: mxfp4 for k, _ in flat_modules if '.ffn.switch_mlp.' in k and k.endswith('_proj')}
    shared_experts = {k: mxfp8 for k, _ in flat_modules if '.ffn.shared_experts.' in k}
    attn = {k: mxfp8 for k, _ in flat_modules if '.attn.w' in k or '.attn.indexer.wq' in k}
    mtp_projs = {k: mxfp8 for k, _ in flat_modules if k.startswith('mtp.') and (k.endswith('.e_proj') or k.endswith('.h_proj') or k.endswith('.main_proj'))}
    return {'group_size': 64, 'bits': 8, 'mode': 'affine', **experts, **shared_experts, **attn, **mtp_projs}
''',
    "_patch_model_args": r'''
def _patch_model_args(dsv4: Any) -> None:
    args_cls = dsv4.ModelArgs
    if '_omlx_mtp_args_patched' in args_cls.__dict__:
        return
    original_from_dict = args_cls.from_dict.__func__

    def patched_from_dict(cls, params):
        args = original_from_dict(cls, params)
        n_main = int(getattr(args, 'num_hidden_layers', 0) or 0)
        is_dspark = deepseek_v4_dspark.is_dspark_config(args)
        n_mtp = deepseek_v4_dspark.stage_count(args) if is_dspark else int(getattr(args, 'num_nextn_predict_layers', 0) or 0)
        if n_mtp > 0 and hasattr(args, 'compress_ratios'):
            source_ratios = list(params.get('compress_ratios') or ())
            ratios = list(args.compress_ratios)
            if is_dspark and len(source_ratios) >= n_main + n_mtp:
                ratios = source_ratios[:n_main + n_mtp]
            if len(ratios) < n_main + n_mtp:
                ratios = ratios + [0] * (n_main + n_mtp - len(ratios))
            args.compress_ratios = ratios
        return args
    args_cls.from_dict = classmethod(patched_from_dict)
    args_cls._omlx_mtp_args_patched = True
''',
    "BaseModelArgs": r'''
@dataclass
class BaseModelArgs:

    @classmethod
    def from_dict(cls, params):
        return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})
''',
    "_get_classes": r'''
def _get_classes(config: dict):
    model_type = config['model_type']
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        arch = importlib.import_module(f'mlx_lm.models.{model_type}')
    except ImportError:
        msg = f'Model type {model_type} not supported.'
        raise ValueError(msg)
    return (arch.Model, arch.ModelArgs)
''',
    "load_config": r'''
def load_config(model_path: Path) -> dict:
    with open(model_path / 'config.json', 'r') as f:
        config = json.load(f)
    generation_config_file = model_path / 'generation_config.json'
    if generation_config_file.exists():
        generation_config = {}
        try:
            with open(generation_config_file, 'r') as f:
                generation_config = json.load(f)
        except json.JSONDecodeError:
            pass
        if (eos_token_id := generation_config.get('eos_token_id', False)):
            config['eos_token_id'] = eos_token_id
    return config
''',
    "load": r'''
def load(path_or_hf_repo: str, tokenizer_config: Optional[Dict[str, Any]]=None, model_config: Optional[Dict[str, Any]]=None, adapter_path: Optional[str]=None, lazy: bool=False, return_config: bool=False, revision: Optional[str]=None, trust_remote_code: bool=False) -> Union[Tuple[nn.Module, TokenizerWrapper], Tuple[nn.Module, TokenizerWrapper, Dict[str, Any]]]:
    model_path = _download(path_or_hf_repo, revision=revision)
    model, config = load_model(model_path, lazy, model_config=model_config, trust_remote_code=trust_remote_code)
    if adapter_path is not None:
        model = load_adapters(model, adapter_path)
        model.eval()
    tokenizer = load_tokenizer(model_path, tokenizer_config, eos_token_ids=config.get('eos_token_id', None))
    if return_config:
        return (model, tokenizer, config)
    else:
        return (model, tokenizer)
''',
    "parse_tool_call": r'''
def parse_tool_call(text: str, tools: Any | None=None):
    matches = list(_INVOKE_RE.finditer(text))
    if not matches:
        raise ValueError('No <｜DSML｜invoke> block found in DeepSeek V4 tool-call text')
    parsed = [_parse_single_invoke(m.group('name'), m.group('body')) for m in matches]
    if len(parsed) == 1:
        return parsed[0]
    return parsed
''',
}


class Runtime:
    def __init__(self, directory):
        self.base = Path(directory)
        self.site = self.base / "runtime"
        self.site.mkdir()
        self.model_dir = self.base / "model"
        self.model_dir.mkdir()
        self.launcher = self.base / "bin" / "omlx"
        self.launcher.parent.mkdir()
        self.launcher.write_text("#!/fake/venv/bin/python\nimport sys\nfrom omlx.cli import main\nif __name__ == '__main__':\n    sys.exit(main())\n")
        self.modules = {}
        self.quantizations = []
        self.weight_loads = []
        self.args = []
        self.supported_modes = {"affine", "mxfp4", "mxfp8"}
        self.config = {"model_type": "deepseek_v4", "num_hidden_layers": 3, "num_nextn_predict_layers": 1,
                       "quantization_config": {"quant_method": "fp8"}}

    def module(self, name, source="", **values):
        filename = self.site / (name.replace(".", "_") + ".py")
        filename.write_text("from __future__ import annotations\n" + source + "\n")
        module = types.ModuleType(name)
        module.__file__ = str(filename)
        module.__path__ = [str(self.site / name.replace(".", "_"))]
        module.__package__ = name.rpartition(".")[0]
        module.__dict__.update(values)
        self.modules[name] = module
        sys.modules[name] = module
        exec(compile(filename.read_text(), str(filename), "exec"), module.__dict__)
        return module

    def forbidden_load(self, *args, **kwargs):
        self.weight_loads.append((args, kwargs))
        raise AssertionError("probe must never load model weights or instantiate a tokenizer")

    def zeros(self, shape):
        if shape[0] != 1 or shape[1] > 1024:
            raise AssertionError("quantizer probe allocated a model-sized tensor")
        return shape

    def quantize(self, shape, group_size, bits, mode="affine"):
        self.quantizations.append((shape, group_size, bits, mode))
        if mode not in self.supported_modes:
            raise ValueError("unsupported synthetic quantization mode")
        return shape

    def activate_architecture(self, qualname, filename):
        if qualname != "mlx_lm.models.deepseek_v4":
            return
        base_args = self.modules["mlx_lm.models.base"].BaseModelArgs
        source = "from dataclasses import dataclass, field\n" + STOCK["ModelArgs"] + "\n" + STOCK["make_quantization_config"]
        source += "\nclass Model:\n    def __init__(self, *args, **kwargs):\n        raise AssertionError('model constructor called')\n"
        self.module(qualname, source, BaseModelArgs=base_args)

    def install(self):
        for name in ("omlx", "omlx.utils", "omlx.patches", "mlx", "mlx_lm", "mlx_lm.models"):
            self.module(name)
        mx = self.module("mlx.core", metal=types.SimpleNamespace(is_available=lambda: True),
                         zeros=self.zeros, quantize=self.quantize, eval=lambda value: None,
                         load=self.forbidden_load)
        nn = self.module("mlx.nn", """
def quantize(model, group_size=64, bits=4, mode="affine", class_predicate=None):
    raise AssertionError("nn.quantize must not construct a model during the probe")
    if isinstance(params, dict):
        return model.to_quantized(**params)
""")
        self.module("mlx_lm.models.base", "import inspect\nfrom dataclasses import dataclass\n" + STOCK["BaseModelArgs"])
        utils_source = "import json, importlib\nfrom pathlib import Path\n" + STOCK["_get_classes"] + "\n" + STOCK["load_config"] + "\n" + STOCK["load_model"] + "\n" + STOCK["load"]
        utils = self.module("mlx_lm.utils", utils_source, MODEL_REMAPPING={}, _load_tokenizer=self.forbidden_load)
        self.modules["mlx_lm"].load = utils.load
        self.modules["mlx_lm"].utils = utils
        self.modules["mlx"].core = mx
        self.modules["mlx"].nn = nn

        tokenizer = self.module("mlx_lm.tokenizer_utils", load=self.forbidden_load)
        tokenizer_patch = self.module("omlx.patches.deepseek_v4.tokenizer_patch", STOCK["apply_load_patch"],
                                      _LOAD_PATCHED=False, _tu=tokenizer,
                                      logger=types.SimpleNamespace(info=lambda *a: None, warning=lambda *a: None))
        tokenizer_patch.apply_tokenizer_patch = lambda: True

        def register_parser():
            self.module("mlx_lm.tool_parsers")
            self.module("mlx_lm.tool_parsers.deepseek_v4", STOCK["parse_tool_call"],
                        tool_call_start="<｜DSML｜tool_calls>", tool_call_end="</｜DSML｜tool_calls>")
        tokenizer_patch._register_chat_template_and_parser_modules = register_parser

        deepseek_utils = self.module("omlx.patches.deepseek_v4.utils_patch",
                                    STOCK["_build_patched_load_model"], _utils=utils)
        def apply_utils_patch():
            utils.load_model = deepseek_utils._build_patched_load_model()
            return True
        deepseek_utils.apply_utils_patch = apply_utils_patch
        self.module("omlx.patches.mlx_lm_sharded_load", install_local_sharded_load_fallback=lambda: True)

        def register_aliases():
            utils.MODEL_REMAPPING["deepseek_v4_mtp"] = "deepseek_v4"
            self.modules["mlx_lm.models.deepseek_v4_mtp"] = self.modules["mlx_lm.models.deepseek_v4"]
            sys.modules["mlx_lm.models.deepseek_v4_mtp"] = self.modules["mlx_lm.models.deepseek_v4"]

        deepseek = self.module("omlx.patches.deepseek_v4", STOCK["apply_deepseek_v4_patch"],
                    _APPLIED=False, apply_pooling_cache_support=lambda: True,
                    _register_module=self.activate_architecture, _register_model_type_aliases=register_aliases,
                    _probe_native_indexer_kernels=lambda: None, PR_HEAD_SHA="5c10538136b9038b9626c134612b08afc18d697a",
                    logger=types.SimpleNamespace(info=lambda *a: None, debug=lambda *a: None))
        deepseek.__package__ = "omlx.patches.deepseek_v4"

        mtp = self.module("omlx.patches.mlx_lm_mtp", STOCK["_patch_model_args"],
                          deepseek_v4_dspark=types.SimpleNamespace(
                              is_dspark_config=lambda args: bool(args.dspark_block_size and args.dspark_target_layer_ids),
                              stage_count=lambda args: args.n_mtp_layers))
        def apply_mtp():
            mtp._patch_model_args(self.modules["mlx_lm.models.deepseek_v4"])
            return True
        mtp.apply_mlx_lm_mtp_patch = apply_mtp
        mtp.set_mtp_active = lambda flag: None
        mtp.set_mtp_depth = lambda depth: None
        self.module("omlx.patches.m5_gather_qmm", apply_m5_gather_qmm_workaround=lambda: False)
        self.module("omlx.patches.arrays_cache_extract", apply_arrays_cache_extract_guard=lambda: None)

        loading = self.module("omlx.utils.model_loading", "import json\nfrom pathlib import Path\n"
                              + STOCK["_patch_mlx_lm_load_config"] + "\n" + STOCK["maybe_apply_pre_load_patches"]
                              + "\n" + STOCK["lm_load_compat"] + "\n" + STOCK["load_text_model"],
                              _MLX_LM_LOAD_CONFIG_PATCHED=False,
                              _is_mtp_compatible=lambda cfg, model_type: model_type.startswith("deepseek_v4"),
                              logger=types.SimpleNamespace(info=lambda *a: None, debug=lambda *a: None))
        for name in ("normalize_hy_v3_rope_config", "expand_per_layer_quant_keys",
                     "expand_glm_moe_dsa_fused_quant_keys", "normalize_laguna_compressed_quant",
                     "normalize_bailing_hybrid_fp8_quant"):
            setattr(loading, name, lambda cfg: cfg)

    def write_model(self, config=None, dtype="U8"):
        self.config = self.config if config is None else config
        (self.model_dir / "config.json").write_text(json.dumps(self.config))
        (self.model_dir / "tokenizer_config.json").write_text("{}")
        self.write_header(self.model_dir / "model.safetensors", dtype)

    @staticmethod
    def write_header(path, dtype):
        data = json.dumps({"dummy": {"dtype": dtype, "shape": [1], "data_offsets": [0, 1]}}).encode()
        path.write_bytes(struct.pack("<Q", len(data)) + data + b"\0")

    def probe(self, model=True):
        payload = {"launcher": str(self.launcher)}
        if model:
            payload["model_dir"] = str(self.model_dir)
        return probe_module.probe(payload)


class OmlxProbeTests(unittest.TestCase):
    @contextlib.contextmanager
    def runtime(self, config=None):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules), patch.object(sys, "path", list(sys.path)):
            runtime = Runtime(directory)
            runtime.install()
            runtime.write_model(config)
            versions = {"omlx": "0.6.4", "mlx-lm": "0.31.3", "mlx": "0.32.0"}
            with patch.object(probe_module.importlib.metadata, "version", side_effect=lambda name: versions[name]):
                yield runtime
            self.assertEqual(runtime.weight_loads, [])

    def test_stock_source_contracts_survive_python_ast_roundtrip(self):
        names = {"maybe_apply_pre_load_patches": "dispatcher", "_patch_mlx_lm_load_config": "config_patch",
                 "lm_load_compat": "text_load_wrapper", "load_text_model": "text_loader",
                 "apply_deepseek_v4_patch": "deepseek_patch", "_get_classes": "resolver",
                 "load_config": "config_reader", "make_quantization_config": "deepseek_quant",
                 "parse_tool_call": "deepseek_parser", "apply_load_patch": "deepseek_tokenizer_patch"}
        for function, contract in names.items():
            tree = probe_module.ast.parse(STOCK[function])
            self.assertIn(probe_module.ast_digest(tree), probe_module.CONTRACTS[contract], function)

    def test_runtime_only_does_not_load_or_apply_model_patches(self):
        with self.runtime() as runtime:
            result = runtime.probe(model=False)
            self.assertTrue(result["ok"])
            self.assertFalse(result["supports_tool_calling"])
            self.assertEqual(result["runtime"]["omlx_version"], "0.6.4")
            self.assertNotIn("mlx_lm.models.deepseek_v4", sys.modules)
            self.assertEqual(runtime.quantizations, [])

    def test_stock_mlx_namespace_package_without_init_py_is_supported(self):
        # MLX v0.32.0/python/mlx has no __init__.py: origin is None, and
        # installed namespace search locations establish its provenance.
        with self.runtime() as runtime:
            runtime.modules["mlx"].__file__ = None
            self.assertTrue(runtime.probe()["ok"])

    def test_stock_deepseek_patch_resolves_architecture_absent_from_vanilla_mlx(self):
        with self.runtime() as runtime:
            with self.assertRaisesRegex(ValueError, "not supported"):
                runtime.modules["mlx_lm.utils"]._get_classes(runtime.config)
            result = runtime.probe()
            self.assertTrue(result["ok"])
            self.assertTrue(result["supports_tool_calling"])
            self.assertEqual(result["model_type"], "deepseek_v4")
            self.assertEqual([(g, b, m) for _, g, b, m in runtime.quantizations],
                             [(64, 8, "affine"), (32, 4, "mxfp4"), (32, 8, "mxfp8")])

    def test_stock_0731_metadata_and_mtp_wrapper_without_model_construction(self):
        config = {"model_type": "deepseek_v4", "num_hidden_layers": 3, "n_mtp_layers": 2,
                  "dspark_block_size": 4, "dspark_target_layer_ids": [0, 2],
                  "num_nextn_predict_layers": 1, "quantization_config": {"quant_method": "fp8"}}
        with self.runtime(config) as runtime:
            self.assertTrue(runtime.probe()["supports_tool_calling"])
            args = runtime.modules["mlx_lm.models.deepseek_v4"].ModelArgs.from_dict(config)
            self.assertEqual(len(args.compress_ratios), 5)

    def test_explicit_mixed_layout_checks_every_distinct_mode(self):
        config = {"model_type": "deepseek_v4", "quantization": {
            "group_size": 64, "bits": 8, "mode": "affine",
            "model.layers.0.ffn.switch_mlp.gate_proj": {"group_size": 32, "bits": 4, "mode": "mxfp4"},
            "model.layers.0.attn.wq": {"group_size": 32, "bits": 8, "mode": "mxfp8"}}}
        with self.runtime(config) as runtime:
            before = (runtime.model_dir / "config.json").read_bytes()
            self.assertTrue(runtime.probe()["ok"])
            self.assertEqual(len(runtime.quantizations), 3)
            self.assertEqual((runtime.model_dir / "config.json").read_bytes(), before)

    def test_legacy_mxfp4_is_verified_in_actual_loader(self):
        with self.runtime({"model_type": "deepseek_v4", "quantization_config": {"quant_method": "mxfp4"}}) as runtime:
            self.assertTrue(runtime.probe()["ok"])
            self.assertEqual(runtime.quantizations, [((1, 32), 32, 4, "mxfp4")])

    def test_normal_installed_llama_model_is_not_universally_rejected(self):
        # Fields and post-init follow the same pinned mlx-lm's models/llama.py.
        config = {"model_type": "llama", "hidden_size": 128, "num_hidden_layers": 2,
                  "intermediate_size": 256, "num_attention_heads": 4,
                  "rms_norm_eps": 1e-5, "vocab_size": 1024,
                  "quantization": {"group_size": 64, "bits": 4}}
        with self.runtime(config) as runtime:
            source = '''
from dataclasses import dataclass
@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int | None = None
    layer_types: list[str] | None = None
    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.layer_types is None:
            self.layer_types = ['full_attention'] * self.num_hidden_layers
class Model:
    def __init__(self, *args):
        raise AssertionError('probe must never construct Llama')
'''
            runtime.module("mlx_lm.models.llama", source,
                           BaseModelArgs=runtime.modules["mlx_lm.models.base"].BaseModelArgs)
            result = runtime.probe()
            self.assertTrue(result["ok"])
            self.assertFalse(result["supports_tool_calling"])
            self.assertNotIn("mlx_lm.models.deepseek_v4", sys.modules)
            self.assertEqual(runtime.quantizations, [((1, 64), 64, 4, "affine")])

    def test_activation_quantization_is_not_claimed_without_bias_validation(self):
        with self.runtime({"model_type": "deepseek_v4", "quantize_activations": True,
                           "quantization": {"group_size": 64, "bits": 4}}) as runtime:
            with self.assertRaisesRegex(ValueError, "unverified oMLX activation quantization"):
                runtime.probe()

    def test_null_text_config_with_outer_quantization_config_matches_loader(self):
        with self.runtime({"model_type": "deepseek_v4", "text_config": None,
                           "quantization_config": {"quant_method": "mxfp4"}}) as runtime:
            self.assertTrue(runtime.probe()["ok"])

    def test_null_text_config_without_outer_quantization_is_loader_incompatible(self):
        with self.runtime({"model_type": "deepseek_v4", "text_config": None}) as runtime:
            with self.assertRaisesRegex(ValueError, "cannot promote quantization_config"):
                runtime.probe()

    def test_foreign_chat_template_keeps_text_but_does_not_claim_tool_support(self):
        with self.runtime() as runtime:
            (runtime.model_dir / "tokenizer_config.json").write_text('{"chat_template_type":"other"}')
            result = runtime.probe()
            self.assertTrue(result["ok"])
            self.assertFalse(result["supports_tool_calling"])

    def test_changed_text_wrapper_cannot_bypass_default_resolver_contract(self):
        with self.runtime() as runtime:
            runtime.modules["omlx.utils.model_loading"].lm_load_compat = runtime.forbidden_load
            with self.assertRaisesRegex(ValueError, "unverified oMLX loader contract: text_load_wrapper"):
                runtime.probe()

    def test_missing_mxfp8_primitive_fails_before_weight_load(self):
        with self.runtime() as runtime:
            runtime.supported_modes.remove("mxfp8")
            with self.assertRaisesRegex(ValueError, "incompatible oMLX quantization mxfp8"):
                runtime.probe()

    def test_unrecognized_quantization_contract_rejected(self):
        with self.runtime({"model_type": "deepseek_v4", "quantization_config": {"quant_method": "unknown"}}) as runtime:
            with self.assertRaisesRegex(ValueError, "without a supported metadata contract"):
                runtime.probe()

    def test_changed_dispatcher_fails_closed_before_calling_it(self):
        with self.runtime() as runtime:
            runtime.modules["omlx.utils.model_loading"].maybe_apply_pre_load_patches = runtime.forbidden_load
            with self.assertRaisesRegex(ValueError, "unverified oMLX loader contract: dispatcher"):
                runtime.probe()

    def test_comment_claims_do_not_grant_source_contract(self):
        tree = probe_module.ast.parse("# all quantization modes supported\n" + STOCK["make_quantization_config"] + "\nraise RuntimeError('changed implementation')")
        self.assertNotIn(probe_module.ast_digest(tree), probe_module.CONTRACTS["deepseek_quant"])

    def test_unverified_parser_preserves_text_eligibility(self):
        with self.runtime() as runtime:
            patcher = runtime.modules["omlx.patches.deepseek_v4.tokenizer_patch"]
            original = patcher._register_chat_template_and_parser_modules
            def changed_parser():
                original()
                runtime.modules["mlx_lm.tool_parsers.deepseek_v4"].parse_tool_call = runtime.forbidden_load
            patcher._register_chat_template_and_parser_modules = changed_parser
            result = runtime.probe()
            self.assertTrue(result["ok"])
            self.assertFalse(result["supports_tool_calling"])
            self.assertIn("deepseek_parser", result["tool_calling_detail"])

    def test_huge_layer_count_rejected_before_runtime_dispatch(self):
        with self.runtime({"model_type": "deepseek_v4", "num_hidden_layers": 1000000000}) as runtime:
            with self.assertRaisesRegex(ValueError, "bounded probe limit"):
                runtime.probe()
            self.assertNotIn("mlx_lm.models.deepseek_v4", sys.modules)

    def test_model_file_rejected_without_executing_repository_python(self):
        with self.runtime({"model_type": "deepseek_v4", "model_file": "custom.py"}) as runtime:
            (runtime.model_dir / "custom.py").write_text("raise AssertionError('repository code executed')")
            with self.assertRaisesRegex(ValueError, "model_file requires"):
                runtime.probe()
            self.assertNotIn("mlx_lm.models.deepseek_v4", sys.modules)

    def test_custom_tokenizer_requires_unverified_repository_code(self):
        with self.runtime() as runtime:
            (runtime.model_dir / "tokenizer_config.json").write_text('{"auto_map":{"AutoTokenizer":"custom.Tokenizer"}}')
            with self.assertRaisesRegex(ValueError, "custom tokenizer"):
                runtime.probe()

    def test_raw_header_symlink_never_modified(self):
        with self.runtime() as runtime:
            shard = runtime.model_dir / "model.safetensors"
            target = runtime.base / "shared.safetensors"
            Runtime.write_header(target, "F8_E8M0")
            shard.unlink()
            shard.symlink_to(target)
            before = target.read_bytes()
            with self.assertRaisesRegex(ValueError, "in-place safetensors header conversion"):
                runtime.probe()
            self.assertEqual(target.read_bytes(), before)
            self.assertNotIn("mlx_lm.models.deepseek_v4", sys.modules)

    def test_truncated_header_rejected(self):
        with self.runtime() as runtime:
            (runtime.model_dir / "model.safetensors").write_bytes(struct.pack("<Q", 100) + b"{}")
            with self.assertRaisesRegex(ValueError, "truncated safetensors header"):
                runtime.probe()

    def test_oversized_header_rejected_without_payload_read(self):
        with self.runtime() as runtime:
            (runtime.model_dir / "model.safetensors").write_bytes(struct.pack("<Q", 1 << 40))
            with self.assertRaisesRegex(ValueError, "bounded probe limit"):
                runtime.probe()

    def test_invalid_model_type_cannot_import_repo_module(self):
        with self.runtime({"model_type": "../custom"}) as runtime:
            with self.assertRaisesRegex(ValueError, "invalid model_type"):
                runtime.probe()

    def test_symlinked_model_repository_module_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules), patch.object(sys, "path", list(sys.path)):
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            library = root / "site"
            library.mkdir()
            marker = root / "executed"
            (model / "unsafe.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n")
            (library / "unsafe_runtime.py").symlink_to(model / "unsafe.py")
            sys.path.insert(0, str(library))
            with self.assertRaisesRegex(ValueError, "refusing model repository Python module"):
                probe_module.installed_module("unsafe_runtime", model)
            self.assertFalse(marker.exists())

    def test_missing_metal_is_a_runtime_error(self):
        with self.runtime() as runtime:
            runtime.modules["mlx.core"].metal.is_available = lambda: False
            with self.assertRaisesRegex(ValueError, "MLX Metal device is unavailable"):
                runtime.probe(model=False)

    def test_main_emits_single_json_object(self):
        with self.runtime() as runtime:
            data = json.dumps({"launcher": str(runtime.launcher), "model_dir": str(runtime.model_dir)})
            output = io.StringIO()
            with patch.object(sys, "stdin", io.StringIO(data)), contextlib.redirect_stdout(output):
                self.assertEqual(probe_module.main(), 0)
            self.assertTrue(json.loads(output.getvalue())["ok"])

    def test_model_args_rejects_bad_compression_metadata(self):
        with self.runtime({"model_type": "deepseek_v4", "num_hidden_layers": 3, "compress_ratios": [0, 5, 0]}) as runtime:
            with self.assertRaisesRegex(ValueError, "Unsupported DeepSeek-V4 compress ratios"):
                runtime.probe()


if __name__ == "__main__":
    unittest.main()
