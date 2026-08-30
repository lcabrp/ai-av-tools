import os
from pathlib import Path
from faster_whisper import WhisperModel

input_path = r"E:\lcabr\OneDrive\Documents\Audacity\TaxPreparer_KTA_Institute_2026-08-26.mp3"
output_path = Path("Output/transcripts_faster_whisper/TaxPreparer_KTA_Institute_2026-08-26.txt")
output_path.parent.mkdir(parents=True, exist_ok=True)

print("Loading model 'medium'...")
model = WhisperModel("medium", device="auto")
print(f"Model loaded. Device: {model.model.device}, compute_type: {model.model.compute_type}")
print("Transcribing...")
segments, info = model.transcribe(str(input_path), language="es")
text = " ".join(segment.text for segment in segments)

output_path.write_text(text, encoding="utf-8")
print("Transcription completed and saved successfully.")
