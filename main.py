from audio_processor import process_audio
from transcriber import transcribe
from summarizer import summarize

from dotenv import load_dotenv
load_dotenv()

def run_pipeline(source :str) -> dict:
    print("Starting...")

    chunks = process_audio(source, 5)

    transcript_file_dir = r"transcript.txt"
    transcribe(chunks, output_file= transcript_file_dir)
    
    print("\n" + "-" * 50)
    print("TRANSCRIPT (first 500 char)")
    print("-" * 50)
    with open(transcript_file_dir, "r", encoding="utf-8") as f:
        transcript = f.read()
    
    print(transcript[:500] + "..." if len(transcript) > 500 else transcript)
    print("-" * 50)

    summary = summarize(transcript)
    print("*"*50)
    # print(transcript)

    return {
        "transcript": transcript,
        "summary": summary,
    }

if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()
    result = run_pipeline(source)

    print("\n" + "-" * 50)
    print(f"\nSummary:\n{result['summary']}")
    print("\n" + "-" * 50)