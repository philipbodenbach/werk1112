import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CONNECTION_CLASS = "WerkConnection";
const IMAGE_MODELS_CLASS = "WerkImageModels";
const VIDEO_MODELS_CLASS = "WerkVideoModels";
const STATUS_MARGIN = 4;
const STATUS_CONTENT_HEIGHT = 28;
const STATUS_WIDGET_HEIGHT = STATUS_CONTENT_HEIGHT + (2 * STATUS_MARGIN);

function widget(node, name) {
    return node?.widgets?.find((candidate) => candidate.name === name);
}

function makeUnserialized(widgetValue) {
    if (!widgetValue) return widgetValue;
    widgetValue.serialize = false;
    widgetValue.options ??= {};
    widgetValue.options.serialize = false;
    return widgetValue;
}

function chainLifecycle(nodeType, name, callback) {
    const original = nodeType.prototype[name];
    nodeType.prototype[name] = function (...args) {
        const result = original?.apply(this, args);
        callback.apply(this, args);
        return result;
    };
}

function nodeClass(node) {
    return node?.constructor?.comfyClass ?? node?.comfyClass ?? node?.type;
}

function fitWerkNode(node) {
    const minimum = node?.computeSize?.();
    if (!minimum || minimum.length < 2) return;
    const currentWidth = Number(node.size?.[0] ?? minimum[0]);
    node.setSize?.([Math.max(currentWidth, minimum[0]), minimum[1]]);
    node.setDirtyCanvas?.(true, true);
}

function syncStatusWidgetWidth(node, statusWidget, element) {
    const nodeWidth = Number(node?.size?.[0] ?? node?.width);
    if (!Number.isFinite(nodeWidth) || nodeWidth <= 0) return;

    // ComfyUI's DOM-widget store prefers widget.width over the current node
    // width. Keep it explicit and current so a width captured before graph
    // configuration cannot survive loading or resizing the node.
    statusWidget.width = nodeWidth;

    // The Vue DOM-widget wrapper is updated asynchronously. This max width is
    // an immediate visual guard for the frame between a node resize/load and
    // the wrapper receiving its new size.
    const margin = Number(statusWidget.margin ?? STATUS_MARGIN);
    const contentWidth = Math.max(0, nodeWidth - (2 * margin));
    element.style.maxWidth = `${contentWidth}px`;
}

function syncStatusWidgetWidths(node) {
    for (const candidate of node?.widgets ?? []) {
        candidate.syncWerkWidth?.();
    }
}

function createStatusWidget(node, name, initialText) {
    let text = initialText;
    let statusWidget;
    const element = document.createElement("div");
    element.textContent = text;
    element.style.cssText = [
        "box-sizing:border-box",
        "width:100%",
        "height:100%",
        "min-height:0",
        "padding:5px 8px",
        "border:1px solid #555",
        "border-radius:6px",
        "background:#242424",
        "color:#b8b8b8",
        "font:12px sans-serif",
        "line-height:16px",
        "overflow:hidden",
        "text-overflow:ellipsis",
        "white-space:nowrap",
    ].join(";");
    statusWidget = node.addDOMWidget(name, "werk-status", element, {
        serialize: false,
        hideOnZoom: false,
        margin: STATUS_MARGIN,
        getMinHeight: () => STATUS_WIDGET_HEIGHT,
        getMaxHeight: () => STATUS_WIDGET_HEIGHT,
        getHeight: () => STATUS_WIDGET_HEIGHT,
        afterResize: () => syncStatusWidgetWidth(node, statusWidget, element),
        getValue: () => text,
        setValue: (value) => {
            text = String(value ?? "");
            element.textContent = text;
        },
    });
    makeUnserialized(statusWidget);
    statusWidget.syncWerkWidth = () => syncStatusWidgetWidth(node, statusWidget, element);
    statusWidget.syncWerkWidth();
    statusWidget.setWerkStatus = (value, state = "idle") => {
        text = String(value ?? "");
        statusWidget.value = text;
        element.textContent = text;
        const colors = {
            idle: ["#b8b8b8", "#555"],
            checking: ["#ffd166", "#8a6d1d"],
            success: ["#70d98b", "#26743a"],
            error: ["#ff8585", "#8e3434"],
        };
        const [color, border] = colors[state] ?? colors.idle;
        element.style.color = color;
        element.style.borderColor = border;
        element.title = String(value ?? "");
        node.setDirtyCanvas?.(true, true);
    };
    return statusWidget;
}

function connectionPayload(node) {
    return {
        server_url: String(widget(node, "server_url")?.value ?? ""),
        api_key: String(widget(node, "api_key")?.value ?? ""),
        timeout_seconds: Number(widget(node, "timeout_seconds")?.value ?? 900),
        verify_tls: Boolean(widget(node, "verify_tls")?.value ?? true),
    };
}

function isWerkConnection(node) {
    return nodeClass(node) === CONNECTION_CLASS;
}

function linkedConnection(node) {
    const slot = node?.findInputSlot?.("connection") ?? -1;
    if (slot < 0) return null;
    const link = node.getInputLink?.(slot);
    const source = link ? node.graph?.getNodeById?.(link.origin_id) : null;
    return isWerkConnection(source) ? source : null;
}

async function requestDiscovery(connectionNode) {
    const response = await api.fetchApi("/werk1112/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(connectionPayload(connectionNode)),
    });
    let payload;
    try {
        payload = await response.json();
    } catch {
        throw new Error(`Connection verification returned HTTP ${response.status}`);
    }
    if (!response.ok || !payload?.ok) {
        throw new Error(payload?.error || `Connection verification returned HTTP ${response.status}`);
    }
    return payload;
}

function setConnectionStatus(node, message, state) {
    node?._werkStatusWidget?.setWerkStatus?.(message, state);
}

function imageModelValues(discovery, requireAvailable = true) {
    const values = requireAvailable
        ? discovery?.image_models?.available
        : discovery?.image_models?.declared;
    return Array.isArray(values) ? values.filter((value) => typeof value === "string") : [];
}

function setComboValues(node, combo, backing, values) {
    const unique = [...new Set(values)].sort((left, right) => left.localeCompare(right));
    combo.options.values.splice(0, combo.options.values.length, ...unique);
    const previous = String(backing.value ?? "");
    const selected = unique.includes(previous) ? previous : (unique[0] ?? "");
    combo.value = selected;
    backing.value = selected;
    backing.callback?.(selected);
    node.setDirtyCanvas?.(true, true);
}

function updateImageModelsNode(node, discovery) {
    if (!node?._werkModelCombo || !node?._werkModelBacking) return;
    const requireAvailable = Boolean(widget(node, "require_available")?.value ?? true);
    const values = imageModelValues(discovery, requireAvailable);
    setComboValues(node, node._werkModelCombo, node._werkModelBacking, values);
    const declared = discovery?.image_models?.declared?.length ?? 0;
    const available = discovery?.image_models?.available?.length ?? 0;
    const message = values.length
        ? `${available} available · ${declared} declared`
        : requireAvailable && declared
          ? `No runtime-available image model (${declared} declared)`
          : "No image-generation model found";
    node._werkModelStatus?.setWerkStatus?.(message, values.length ? "success" : "error");
}

function updateVideoModelsNode(node, discovery) {
    if (!node?._werkModelCombo || !node?._werkModelBacking) return;
    const requireAvailable = Boolean(widget(node, "require_available")?.value ?? true);
    const task = String(widget(node, "task")?.value ?? "video-generation");
    const taskModels = discovery?.video_models?.by_task?.[task];
    const rawValues = requireAvailable ? taskModels?.available : taskModels?.declared;
    const values = Array.isArray(rawValues)
        ? rawValues.filter((value) => typeof value === "string")
        : [];
    setComboValues(node, node._werkModelCombo, node._werkModelBacking, values);
    const declared = taskModels?.declared?.length ?? 0;
    const available = taskModels?.available?.length ?? 0;
    const message = values.length
        ? `${available} available · ${declared} declared · ${task}`
        : requireAvailable && declared
          ? `No runtime-available ${task} model (${declared} declared)`
          : `No ${task} model found`;
    node._werkModelStatus?.setWerkStatus?.(message, values.length ? "success" : "error");
}

function propagateDiscovery(connectionNode, discovery) {
    connectionNode._werkDiscovery = discovery;
    for (const node of app.graph?._nodes ?? []) {
        if (linkedConnection(node) !== connectionNode) continue;
        const className = nodeClass(node);
        if (className === IMAGE_MODELS_CLASS) updateImageModelsNode(node, discovery);
        if (className === VIDEO_MODELS_CLASS) updateVideoModelsNode(node, discovery);
    }
}

async function verifyConnection(connectionNode) {
    setConnectionStatus(connectionNode, "Checking connection…", "checking");
    try {
        const discovery = await requestDiscovery(connectionNode);
        setConnectionStatus(connectionNode, discovery.status, "success");
        propagateDiscovery(connectionNode, discovery);
        return discovery;
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        setConnectionStatus(connectionNode, `Failed: ${message}`, "error");
        throw error;
    }
}

function installConnectionUi(nodeType) {
    chainLifecycle(nodeType, "onNodeCreated", function () {
        this._werkStatusWidget = createStatusWidget(this, "connection_status", "Not verified");
        const button = makeUnserialized(this.addWidget("button", "Verify Connection", null, async () => {
            button.disabled = true;
            try {
                await verifyConnection(this);
            } catch {
                // The visible status already contains the safe error.
            } finally {
                button.disabled = false;
            }
        }));
        fitWerkNode(this);
    });
}

function installImageModelsUi(nodeType) {
    chainLifecycle(nodeType, "onNodeCreated", function () {
        const backing = widget(this, "preferred_model");
        if (!backing) return;
        backing.hidden = true;
        backing.options ??= {};
        backing.options.hidden = true;
        const values = [];
        const combo = makeUnserialized(this.addWidget("combo", "available_model", backing.value ?? "", (value) => {
            backing.value = value;
            backing.callback?.(value);
        }, { values }));
        this._werkModelBacking = backing;
        this._werkModelCombo = combo;
        this._werkModelStatus = createStatusWidget(this, "model_discovery_status", "Connect and refresh models");
        const button = makeUnserialized(this.addWidget("button", "Refresh Models", null, async () => {
            const connectionNode = linkedConnection(this);
            if (!connectionNode) {
                this._werkModelStatus.setWerkStatus("Connect a WERK Connection first", "error");
                return;
            }
            this._werkModelStatus.setWerkStatus("Refreshing models…", "checking");
            button.disabled = true;
            try {
                const discovery = await verifyConnection(connectionNode);
                updateImageModelsNode(this, discovery);
                const refresh = widget(this, "refresh_token");
                if (refresh) refresh.value = Number(refresh.value ?? 0) + 1;
            } catch (error) {
                const message = error instanceof Error ? error.message : String(error);
                this._werkModelStatus.setWerkStatus(`Failed: ${message}`, "error");
            } finally {
                button.disabled = false;
            }
        }));
        fitWerkNode(this);
    });
    chainLifecycle(nodeType, "onConnectionsChange", function () {
        const connectionNode = linkedConnection(this);
        if (connectionNode?._werkDiscovery) updateImageModelsNode(this, connectionNode._werkDiscovery);
    });
}

function installVideoModelsUi(nodeType) {
    chainLifecycle(nodeType, "onNodeCreated", function () {
        const node = this;
        const backing = widget(node, "preferred_model");
        if (!backing) return;
        backing.hidden = true;
        backing.options ??= {};
        backing.options.hidden = true;
        const values = [];
        const combo = makeUnserialized(node.addWidget("combo", "available_model", backing.value ?? "", (value) => {
            backing.value = value;
            backing.callback?.(value);
        }, { values }));
        node._werkModelBacking = backing;
        node._werkModelCombo = combo;
        node._werkModelStatus = createStatusWidget(node, "model_discovery_status", "Connect and refresh models");

        for (const name of ["task", "require_available"]) {
            const control = widget(node, name);
            if (!control) continue;
            const original = control.callback;
            control.callback = function (...args) {
                const result = original?.apply(this, args);
                const connectionNode = linkedConnection(node);
                if (connectionNode?._werkDiscovery) {
                    updateVideoModelsNode(node, connectionNode._werkDiscovery);
                }
                return result;
            };
        }

        const button = makeUnserialized(node.addWidget("button", "Refresh Models", null, async () => {
            const connectionNode = linkedConnection(node);
            if (!connectionNode) {
                node._werkModelStatus.setWerkStatus("Connect a WERK Connection first", "error");
                return;
            }
            node._werkModelStatus.setWerkStatus("Refreshing models…", "checking");
            button.disabled = true;
            try {
                const discovery = await verifyConnection(connectionNode);
                updateVideoModelsNode(node, discovery);
                const refresh = widget(node, "refresh_token");
                if (refresh) refresh.value = Number(refresh.value ?? 0) + 1;
            } catch (error) {
                const message = error instanceof Error ? error.message : String(error);
                node._werkModelStatus.setWerkStatus(`Failed: ${message}`, "error");
            } finally {
                button.disabled = false;
            }
        }));
        fitWerkNode(node);
    });
    chainLifecycle(nodeType, "onConnectionsChange", function () {
        const connectionNode = linkedConnection(this);
        if (connectionNode?._werkDiscovery) updateVideoModelsNode(this, connectionNode._werkDiscovery);
    });
}

app.registerExtension({
    name: "werk1112.dynamic-discovery",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === CONNECTION_CLASS) installConnectionUi(nodeType);
        if (nodeData.name === IMAGE_MODELS_CLASS) installImageModelsUi(nodeType);
        if (nodeData.name === VIDEO_MODELS_CLASS) installVideoModelsUi(nodeType);
    },
    loadedGraphNode(node) {
        const className = nodeClass(node);
        if (
            className === CONNECTION_CLASS
            || className === IMAGE_MODELS_CLASS
            || className === VIDEO_MODELS_CLASS
        ) {
            fitWerkNode(node);
            syncStatusWidgetWidths(node);
        }
    },
});
