import env from "@/config/env";

export type SpeechTranscriptionResult = {
  language?: string;
  language_probability?: number;
  model: string;
  text: string;
};

const CANDIDATE_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",
  "audio/ogg;codecs=opus",
] as const;

export function pickRecordingMimeType() {
  if (typeof MediaRecorder === "undefined") {
    return "";
  }
  return CANDIDATE_TYPES.find((type) => MediaRecorder.isTypeSupported(type)) ?? "";
}

export async function transcribeSpeech(
  audioBlob: Blob
): Promise<SpeechTranscriptionResult> {
  const url = `${env.API_URL.replace(/\/$/, "")}/speech/transcribe`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "content-type": audioBlob.type || "audio/webm",
    },
    body: audioBlob,
  });

  if (!response.ok) {
    let detail = `Speech transcription failed: ${response.status}`;
    try {
      const payload = (await response.json()) as {
        detail?: string | { message?: string };
      };
      if (typeof payload.detail === "string") {
        detail = payload.detail;
      } else if (payload.detail?.message) {
        detail = payload.detail.message;
      }
    } catch {
      // Ignore non-JSON error payloads.
    }
    throw new Error(detail);
  }

  return (await response.json()) as SpeechTranscriptionResult;
}
