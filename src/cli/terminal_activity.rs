use indicatif::{ProgressBar, ProgressStyle};
use std::{
    env,
    io::{self, IsTerminal},
    time::Duration,
};

use crate::capabilities::InferenceTask;

const SPINNER_FRAMES: &[&str] = &["|", "/", "-", "\\", ""];
const CHAT_FRAMES: &[&str] = &["·  ", "·· ", "···", " ✦ ", " ✧ ", ""];
const IMAGE_FRAMES: &[&str] = &["▖", "▘", "▝", "▗", ""];
const VIDEO_FRAMES: &[&str] = &["▰▱▱", "▱▰▱", "▱▱▰", "▱▰▱", ""];
const AUDIO_FRAMES: &[&str] = &[
    "▁▂▄▆▄▂▁",
    "▂▄▆█▆▄▂",
    "▄▆█▆▄▂▄",
    "▆█▆▄▂▄▆",
    "█▆▄▂▄▆█",
    "▆▄▂▄▆█▆",
    "▄▂▄▆█▆▄",
    "▂▄▆█▆▄▂",
    "",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ActivityKind {
    Spinner,
    Chat,
    Image,
    Video,
    Audio,
}

impl ActivityKind {
    const fn frames(self) -> &'static [&'static str] {
        match self {
            Self::Spinner => SPINNER_FRAMES,
            Self::Chat => CHAT_FRAMES,
            Self::Image => IMAGE_FRAMES,
            Self::Video => VIDEO_FRAMES,
            Self::Audio => AUDIO_FRAMES,
        }
    }

    pub(super) const fn frame(self, index: usize) -> &'static str {
        let frames = self.frames();
        frames[index % (frames.len() - 1)]
    }

    const fn template(self) -> &'static str {
        match self {
            Self::Spinner => "{spinner:.cyan} {msg}",
            Self::Chat => "{spinner:.cyan} {msg} [{elapsed_precise}]",
            Self::Image => "{spinner:.magenta} {msg} [{elapsed_precise}]",
            Self::Video => "{spinner:.blue} {msg} [{elapsed_precise}]",
            Self::Audio => "{spinner:.green} {msg} [{elapsed_precise}]",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ActivitySpec {
    kind: ActivityKind,
    action: &'static str,
}

impl ActivitySpec {
    pub(super) const fn chat() -> Self {
        Self {
            kind: ActivityKind::Chat,
            action: "Werk is thinking with",
        }
    }

    pub(super) const fn for_task(task: InferenceTask) -> Self {
        use InferenceTask::{
            AudioCaptioning, AudioClassification, AudioEditing, AudioEmbedding, AudioEnhancement,
            AudioEventDetection, AudioGeneration, AudioUnderstanding, FrameInterpolation,
            ImageEditing, ImageGeneration, ImageInpainting, ImageOutpainting, ImageToVideo,
            ImageUnderstanding, ImageUpscaling, ImageVariation, LanguageIdentification,
            MusicGeneration, SongContinuation, SongVariation, SpeakerDiarization,
            SpeakerIdentification, SpeechEmotionRecognition, SpeechToText, SpeechTranslation,
            StemGeneration, StemSeparation, TextEmbedding, TextGeneration, TextToSpeech,
            VideoExtension, VideoGeneration, VideoInpainting, VideoToVideo, VideoUpscaling,
            VoiceActivityDetection, VoiceConversion,
        };

        match task {
            TextGeneration => Self::chat(),
            TextEmbedding => Self {
                kind: ActivityKind::Chat,
                action: "Mapping meaning with",
            },
            ImageUnderstanding => Self {
                kind: ActivityKind::Chat,
                action: "Studying the image with",
            },
            ImageGeneration => Self {
                kind: ActivityKind::Image,
                action: "Painting pixels with",
            },
            ImageEditing | ImageVariation => Self {
                kind: ActivityKind::Image,
                action: "Remixing pixels with",
            },
            ImageInpainting | ImageOutpainting => Self {
                kind: ActivityKind::Image,
                action: "Filling the canvas with",
            },
            ImageUpscaling => Self {
                kind: ActivityKind::Image,
                action: "Finding extra pixels with",
            },
            VideoGeneration => Self {
                kind: ActivityKind::Video,
                action: "Teaching frames to move with",
            },
            ImageToVideo => Self {
                kind: ActivityKind::Video,
                action: "Waking up the frame with",
            },
            VideoToVideo | VideoInpainting | VideoExtension => Self {
                kind: ActivityKind::Video,
                action: "Remixing moving pictures with",
            },
            VideoUpscaling => Self {
                kind: ActivityKind::Video,
                action: "Sharpening every frame with",
            },
            FrameInterpolation => Self {
                kind: ActivityKind::Video,
                action: "Dreaming between frames with",
            },
            AudioGeneration | MusicGeneration => Self {
                kind: ActivityKind::Audio,
                action: "Riding the waveform with",
            },
            SongContinuation => Self {
                kind: ActivityKind::Audio,
                action: "Keeping the groove going with",
            },
            SongVariation => Self {
                kind: ActivityKind::Audio,
                action: "Remixing the groove with",
            },
            TextToSpeech => Self {
                kind: ActivityKind::Audio,
                action: "Giving words a voice with",
            },
            SpeechToText | SpeechTranslation => Self {
                kind: ActivityKind::Audio,
                action: "Listening closely with",
            },
            AudioEventDetection
            | VoiceActivityDetection
            | SpeakerIdentification
            | LanguageIdentification
            | SpeechEmotionRecognition
            | AudioClassification => Self {
                kind: ActivityKind::Audio,
                action: "Classifying the waveform with",
            },
            AudioCaptioning | SpeakerDiarization | AudioUnderstanding => Self {
                kind: ActivityKind::Audio,
                action: "Understanding the recording with",
            },
            AudioEmbedding => Self {
                kind: ActivityKind::Audio,
                action: "Mapping the waveform with",
            },
            VoiceConversion => Self {
                kind: ActivityKind::Audio,
                action: "Changing the voice with",
            },
            StemGeneration => Self {
                kind: ActivityKind::Audio,
                action: "Building the stems with",
            },
            StemSeparation => Self {
                kind: ActivityKind::Audio,
                action: "Untangling the waveform with",
            },
            AudioEnhancement | AudioEditing => Self {
                kind: ActivityKind::Audio,
                action: "Polishing the waveform with",
            },
        }
    }

    pub(super) fn message(self, model: &str) -> String {
        format!("{} '{model}'...", self.action)
    }

    pub(super) const fn kind(self) -> ActivityKind {
        self.kind
    }
}

struct TerminalActivity {
    progress: Option<ProgressBar>,
}

impl TerminalActivity {
    fn start(enabled: bool, kind: ActivityKind, message: impl Into<String>) -> Self {
        let progress = enabled.then(|| {
            let progress = ProgressBar::new_spinner();
            let style = ProgressStyle::with_template(kind.template())
                .expect("terminal activity template is valid")
                .tick_strings(kind.frames());
            progress.set_style(style);
            progress.set_message(message.into());
            progress.enable_steady_tick(Duration::from_millis(110));
            progress
        });
        Self { progress }
    }

    fn clear(&mut self) {
        if let Some(progress) = self.progress.take() {
            progress.finish_and_clear();
        }
    }
}

impl Drop for TerminalActivity {
    fn drop(&mut self) {
        self.clear();
    }
}

pub(super) fn with_activity<T>(
    enabled: bool,
    kind: ActivityKind,
    message: impl Into<String>,
    operation: impl FnOnce() -> T,
) -> T {
    let _activity = TerminalActivity::start(enabled, kind, message);
    operation()
}

pub(super) fn generation_activity_enabled(debug: bool) -> bool {
    generation_activity_enabled_for(
        io::stdout().is_terminal(),
        io::stderr().is_terminal(),
        debug,
        env::var("TERM")
            .map(|term| term.eq_ignore_ascii_case("dumb"))
            .unwrap_or(false),
    )
}

const fn generation_activity_enabled_for(
    stdout_is_terminal: bool,
    stderr_is_terminal: bool,
    debug: bool,
    term_is_dumb: bool,
) -> bool {
    stdout_is_terminal && stderr_is_terminal && !debug && !term_is_dumb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_activity_requires_interactive_output() {
        assert!(generation_activity_enabled_for(true, true, false, false));
        assert!(!generation_activity_enabled_for(false, true, false, false));
        assert!(!generation_activity_enabled_for(true, false, false, false));
        assert!(!generation_activity_enabled_for(true, true, true, false));
        assert!(!generation_activity_enabled_for(true, true, false, true));
    }

    #[test]
    fn activity_frames_are_animated_single_lines() {
        for kind in [
            ActivityKind::Spinner,
            ActivityKind::Chat,
            ActivityKind::Image,
            ActivityKind::Video,
            ActivityKind::Audio,
        ] {
            let frames = kind.frames();
            let _style = ProgressStyle::with_template(kind.template())
                .unwrap()
                .tick_strings(frames);
            assert!(frames.len() >= 3);
            assert_eq!(frames.last(), Some(&""));
            assert!(
                frames[..frames.len() - 1]
                    .iter()
                    .all(|frame| !frame.is_empty()
                        && !frame.contains('\n')
                        && !frame.contains('\r'))
            );
        }
        assert!(
            ActivityKind::Audio
                .frames()
                .iter()
                .any(|frame| frame.contains('█'))
        );
    }

    #[test]
    fn media_tasks_get_modality_specific_activity() {
        assert_eq!(
            ActivitySpec::for_task(InferenceTask::ImageGeneration).kind(),
            ActivityKind::Image
        );
        assert_eq!(
            ActivitySpec::for_task(InferenceTask::VideoGeneration).kind(),
            ActivityKind::Video
        );
        assert_eq!(
            ActivitySpec::for_task(InferenceTask::MusicGeneration).kind(),
            ActivityKind::Audio
        );
        assert_eq!(
            ActivitySpec::for_task(InferenceTask::SpeechToText).kind(),
            ActivityKind::Audio
        );
        assert_eq!(
            ActivitySpec::for_task(InferenceTask::TextGeneration).kind(),
            ActivityKind::Chat
        );
    }

    #[test]
    fn disabled_activity_preserves_operation_result() {
        let result = with_activity(false, ActivityKind::Image, "hidden", || {
            Result::<_, &'static str>::Ok(42)
        });
        assert_eq!(result, Ok(42));

        let error = with_activity(true, ActivityKind::Audio, "visible", || {
            Result::<(), _>::Err("generation failed")
        });
        assert_eq!(error, Err("generation failed"));
    }
}
