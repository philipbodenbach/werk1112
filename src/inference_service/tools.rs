//! Function schemas for every inference task. Execution uses the same validated
//! requests and runtime selection as the ordinary API; model quality is not a
//! tool capability. Chat tasks use the chat adapter, media tasks use jobs.
use crate::{
    capabilities::{InferenceTask, InputModality},
    inference::{InferenceRequest, ParameterType, parameter_schema},
};
use anyhow::{Result, anyhow, bail};
use serde_json::{Map, Value, json};

pub fn tool_definitions() -> Vec<Value> {
    let mut tools = InferenceTask::ALL
        .iter()
        .copied()
        .map(tool_definition)
        .collect::<Vec<_>>();
    for (name, description) in [
        (
            "get_job",
            "Get a Werk media tool job's status, error and generated outputs. Poll until terminal.",
        ),
        (
            "cancel_job",
            "Request cancellation of a Werk media tool job.",
        ),
    ] {
        tools.push(json!({"type":"function", "function":{
            "name":name, "description":description,
            "parameters":{"type":"object", "properties":{"id":{"type":"string","description":"Job ID returned by the media tool"}}, "required":["id"], "additionalProperties":false}
        }}));
    }
    tools
}

pub fn tool_definition(task: InferenceTask) -> Value {
    let mut parameters = if matches!(
        task,
        InferenceTask::TextGeneration | InferenceTask::ImageUnderstanding
    ) {
        json!({
            "type":"object", "properties":{
                "model":{"type":"string","description":"Installed model ID"},
                "messages":{"type":"array","minItems":1,"description":"OpenAI chat messages, including image_url parts for vision, assistant tool_calls and tool results.","items":{"type":"object","properties":{
                    "role":{"type":"string","enum":["system","developer","user","assistant","tool"]},
                    "content":{"description":"Text, null, or multimodal content parts"},
                    "name":{"type":"string"},"tool_calls":{"type":"array","items":{"type":"object"}},"tool_call_id":{"type":"string"}
                },"required":["role"],"additionalProperties":false}},
                "max_tokens":{"type":"integer","minimum":1},
                "temperature":{"type":"number"},"top_p":{"type":"number"},
                "tools":{"type":"array","items":{"type":"object"}},
                "tool_choice":{"description":"auto, none, required, or a named OpenAI function choice"},
                "parallel_tool_calls":{"type":"boolean"}
            }, "required":["model","messages"], "additionalProperties":false
        })
    } else {
        let mut properties = Map::new();
        for descriptor in parameter_schema(task) {
            let kind = match descriptor.value_type {
                ParameterType::Boolean => "boolean",
                ParameterType::Integer => "integer",
                ParameterType::Number => "number",
                ParameterType::List => "array",
                ParameterType::Object => "object",
                _ => "string",
            };
            let mut property = json!({"type":kind,"description":descriptor.description});
            if let Some(minimum) = descriptor.minimum {
                property["minimum"] = json!(minimum);
            }
            if let Some(maximum) = descriptor.maximum {
                property["maximum"] = json!(maximum);
            }
            if !descriptor.allowed_values.is_empty() {
                property["enum"] = json!(descriptor.allowed_values);
            }
            if kind == "array" {
                property["items"] = json!({});
            }
            properties.insert(descriptor.path, property);
        }
        json!({"type":"object","properties":{
            "model":{"type":"string","description":"Installed model ID. Runtime and model readiness are checked when executing."},
            "prompt":{"type":"string"},"negative_prompt":{"type":"string"},
            "inputs":{"type":"array","items":{
                "type":"object","properties":{
                    "modality":{"type":"string","enum":["text","image","video","audio"]},
                    "role":{"type":"string","description":"Task input role, e.g. image, mask, initial_image, input_video, input_audio, reference_audio"},
                    "mime_type":{"type":"string"},
                    "source":{"oneOf":[
                        {"type":"object","properties":{"kind":{"const":"path"},"path":{"type":"string"}},"required":["kind","path"],"additionalProperties":false},
                        {"type":"object","properties":{"kind":{"const":"url"},"url":{"type":"string"}},"required":["kind","url"],"additionalProperties":false},
                        {"type":"object","properties":{"kind":{"const":"base64"},"data":{"type":"string"}},"required":["kind","data"],"additionalProperties":false},
                        {"type":"object","properties":{"kind":{"const":"text"},"text":{"type":"string"}},"required":["kind","text"],"additionalProperties":false}
                    ]}
                },"required":["modality","role","source"],"additionalProperties":false
            }},
            "parameters":{"type":"object","description":"Task and routing options using canonical dotted paths. Backend/device selection uses routing.backend and routing.device.","properties":properties,"additionalProperties":false}
        },"required":["model"],"additionalProperties":false})
    };
    if !matches!(
        task,
        InferenceTask::TextGeneration | InferenceTask::ImageUnderstanding
    ) {
        let role_groups = required_media_role_groups(task);
        if !role_groups.is_empty() {
            let inputs = &mut parameters["properties"]["inputs"];
            inputs["minItems"] = json!(role_groups.len());
            inputs["description"] = json!(format!(
                "Required inputs: {}. Each group needs a separate input with the indicated modality and one of its roles.",
                role_groups
                    .iter()
                    .map(|(modality, roles)| format!("{modality}: {}", roles.join(" or ")))
                    .collect::<Vec<_>>()
                    .join("; ")
            ));
            inputs["allOf"] = json!(
                role_groups
                    .iter()
                    .map(|(modality, roles)| json!({
                        "contains": {
                            "type":"object",
                            "properties":{"modality":{"const":modality},"role":{"enum":roles}},
                            "required":["modality","role"]
                        }
                    }))
                    .collect::<Vec<_>>()
            );
        }
        let mut required = vec![json!("model")];
        if task.requires_prompt() {
            required.push(json!("prompt"));
            parameters["properties"]["prompt"]["minLength"] = json!(1);
        }
        if task
            .required_input_modalities()
            .iter()
            .any(|modality| *modality != crate::capabilities::InputModality::Text)
        {
            required.push(json!("inputs"));
        }
        parameters["required"] = json!(required);
        let properties = parameters["properties"].as_object_mut().unwrap();
        if !task.accepts_prompt() {
            properties.remove("prompt");
        }
        if !task.accepts_negative_prompt() {
            properties.remove("negative_prompt");
        }
    }
    if task == InferenceTask::ImageUnderstanding {
        let messages = &mut parameters["properties"]["messages"];
        messages["description"] = json!(
            "OpenAI chat messages with at least one image_url or input_image part. Preserve image input in the history when continuing a vision tool conversation."
        );
        messages["contains"] = json!({
            "type":"object", "required":["content"], "properties":{
                "content":{"type":"array","contains":{
                    "type":"object","required":["type","image_url"],"properties":{
                        "type":{"enum":["image_url","input_image"]},
                        "image_url":{"oneOf":[
                            {"type":"string","minLength":1},
                            {"type":"object","properties":{"url":{"type":"string","minLength":1}},"required":["url"]}
                        ]}
                    }
                }}
            }
        });
    }
    let description = if matches!(
        task,
        InferenceTask::TextGeneration | InferenceTask::ImageUnderstanding
    ) {
        format!(
            "Run Werk {task} through a chat/vision adapter. Returns the chat response, including any tool calls; never executes nested model-selected tools automatically."
        )
    } else {
        format!(
            "Submit Werk {task} to any compatible configured backend. Returns a job ID; use get_job to retrieve status and output URLs. See /v1/capabilities for installed models and readiness."
        )
    };
    json!({"type":"function","function":{"name":task.to_string().replace('-',"_"),"description":description,"parameters":parameters}})
}

// Use canonical roles from inference resolution in the published schema. Extra
// reference inputs remain allowed, but cannot stand in for a required source.
fn required_media_role_groups(
    task: InferenceTask,
) -> Vec<(InputModality, &'static [&'static str])> {
    let mut groups: Vec<(InputModality, &'static [&'static str])> = task
        .required_input_modalities()
        .iter()
        .filter_map(|modality| match modality {
            InputModality::Image => {
                Some((*modality, &["image", "input_image", "initial_image"][..]))
            }
            InputModality::Video => {
                Some((*modality, &["source_video", "input_video", "video"][..]))
            }
            InputModality::Audio => {
                Some((*modality, &["input_audio", "source_audio", "audio"][..]))
            }
            InputModality::Text => None,
        })
        .collect();
    match task {
        InferenceTask::ImageInpainting | InferenceTask::ImageOutpainting => {
            groups.push((InputModality::Image, &["mask", "mask_image"]));
        }
        InferenceTask::VideoInpainting => {
            groups.push((InputModality::Video, &["mask_video", "mask"]));
        }
        _ => {}
    }
    groups
}

pub fn tool_inference_request(
    task: InferenceTask,
    mut arguments: Value,
) -> Result<InferenceRequest> {
    if matches!(
        task,
        InferenceTask::TextGeneration | InferenceTask::ImageUnderstanding
    ) {
        bail!("chat tools require the chat adapter");
    }
    let object = arguments
        .as_object_mut()
        .ok_or_else(|| anyhow!("function arguments must be a JSON object"))?;
    for key in object.keys() {
        if !["model", "prompt", "negative_prompt", "inputs", "parameters"].contains(&key.as_str()) {
            bail!("unknown function argument '{key}'");
        }
    }
    object.insert("task".into(), json!(task));
    let request: InferenceRequest = serde_json::from_value(arguments)?;
    if request.model.trim().is_empty() {
        bail!("model must not be empty");
    }
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_task_has_a_complete_unique_function_schema() {
        let definitions = tool_definitions();
        let names = definitions
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(names.len(), InferenceTask::ALL.len() + 2);
        let speech = tool_definition(InferenceTask::SpeechToText);
        assert!(
            speech["function"]["parameters"]["properties"]
                .get("prompt")
                .is_none()
        );
        let image_video = tool_definition(InferenceTask::ImageToVideo);
        assert_eq!(
            image_video["function"]["parameters"]["required"],
            json!(["model", "prompt", "inputs"])
        );
        for task in InferenceTask::ALL {
            let definition = tool_definition(*task);
            assert_eq!(definition["function"]["parameters"]["type"], "object");
            assert!(
                definition["function"]["parameters"]["required"]
                    .as_array()
                    .unwrap()
                    .contains(&json!("model"))
            );
        }
        for name in [
            "image_generation",
            "video_generation",
            "music_generation",
            "text_to_speech",
            "speech_to_text",
            "audio_understanding",
            "voice_conversion",
            "image_understanding",
        ] {
            assert!(names.contains(name), "{name}");
        }
    }
    #[test]
    fn media_arguments_preserve_inputs_and_backend_parameters() {
        let request = tool_inference_request(InferenceTask::ImageToVideo,json!({"model":"video", "prompt":"move", "inputs":[{"modality":"image","role":"initial_image","source":{"kind":"base64","data":"AAEC"}}],"parameters":{"video.frames":24,"routing.backend":"diffusers"}})).unwrap();
        assert_eq!(request.task, InferenceTask::ImageToVideo);
        assert_eq!(request.inputs.len(), 1);
        assert_eq!(
            request.parameters["routing.backend"].as_str(),
            Some("diffusers")
        );
        assert!(
            tool_inference_request(
                InferenceTask::AudioGeneration,
                json!({"model":"audio","task":"image_generation"})
            )
            .is_err()
        );
        assert!(
            tool_inference_request(
                InferenceTask::AudioGeneration,
                json!({"model":"audio","unexpected":true})
            )
            .is_err()
        );
    }
}
