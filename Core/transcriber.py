from faster_whisper import WhisperModel
import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None

def load_model():
    global _model

    if _model is None:
        print("Loading Whisper model...")
        _model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",          # "cuda" if GPU
            compute_type="int8"    # fastest on CPU
        )
        print("Whisper model loaded successfully")

    return _model

def transcribe_chunk(chunk_path : str, translate: bool= False) -> str:
    model = load_model()

    task = "translate" if translate else "transcribe"

    segments, info = model.transcribe(chunk_path, task = task)
    text = " ".join(segment.text for segment in segments)

    return text

def transcribe(chunks: list, translate: bool = False, output_file: str = "transcript.txt"):
    # Clear existing file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("")
    
    for i , chunk in enumerate(chunks):
        text = transcribe_chunk(chunk, translate=translate)
        print(f"Transcribed chunk {i+1}")

        # append to file
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(text + "\n\n")
    
    print("Transcription Completed")

