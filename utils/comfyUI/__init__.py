"""ComfyUI custom-node exports for Werk1112."""

from .nodes import (
    NODE_CLASS_MAPPINGS as LEGACY_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as LEGACY_NODE_DISPLAY_NAME_MAPPINGS,
)
from .runtime_nodes import (
    NODE_CLASS_MAPPINGS as RUNTIME_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as RUNTIME_NODE_DISPLAY_NAME_MAPPINGS,
)
from .routes import register_routes

NODE_CLASS_MAPPINGS = {
    **LEGACY_NODE_CLASS_MAPPINGS,
    **RUNTIME_NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **LEGACY_NODE_DISPLAY_NAME_MAPPINGS,
    **RUNTIME_NODE_DISPLAY_NAME_MAPPINGS,
}

# Serve the directory containing the module itself.  ComfyUI mounts this as
# /extensions/<module>/, so the standard ../../scripts/*.js imports resolve to
# ComfyUI's /scripts/ directory.  Serving ./web instead would add another
# /js/ path component and incorrectly resolve them below /extensions/scripts/.
WEB_DIRECTORY = "./web/js"

register_routes()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
