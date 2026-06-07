import hashlib, os, time
import re
from Core.audio_processor import process_audio
from Core.transcriber import transcribe
from Core.summarizer import summarize
from Core.rag_engine import get_rag_chain, ask_question
from dotenv import load_dotenv

TRANSCRIPT_DIR = "Transcripts_Downloaded"
os.makedirs(TRANSCRIPT_DIR, exist_ok = True)
load_dotenv()

def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    raise ValueError("Invalid YouTube URL")

def run_pipeline(source :str) -> dict:
    print("Starting...\n")
    
    if source.startswith("https://") or source.startswith("http://"):
        # Youtube url
        video_id = extract_video_id(source)

    else:
        # Local audio file
        video_id = hashlib.md5(os.path.abspath(source).encode()).hexdigest()
        

    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{video_id}.txt" )

    # Transcript already exists
    if os.path.exists(transcript_path):
        print("Transcript found. Skipping processing.")
    else:
        print("Creating transcript...")

        t1 = time.perf_counter()
        chunks = process_audio(source, 5)
        transcribe(chunks, output_file=transcript_path)
        print("-"*50 + '\n')
        print(f"Transcript ready in {time.perf_counter() - t1:.2f}s\n")
        print("-"*50 + '\n')
    
    print("\n" + "-" * 50)
    print("TRANSCRIPT (first 500 char)")
    print("-" * 50)
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()
    
    print(transcript[:500] + "..." if len(transcript) > 500 else transcript)
    print("-" * 50)

    t2 = time.perf_counter()
    summary = summarize(transcript)
    print("*"*50)
    # print(transcript)
    print("-"*50 + '\n')
    print(f"Summary Generated in {time.perf_counter() - t2:.2f}s\n")
    print("-"*50 + '\n')
    
    t3 = time.perf_counter()
    rag_chain = get_rag_chain(transcript, video_id)
    print("-"*50 + '\n')
    print(f"RAG ready in {time.perf_counter() - t2:.2f}s\n")
    print("-"*50 + '\n')

    return {
        "video_id": video_id,
        "transcript": transcript,
        "summary": summary,
        "rag_chain": rag_chain
    }

if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()

    t0 = time.perf_counter()
    result = run_pipeline(source)
    print("-"*50 + '\n')
    print(f"Pipeline completed in {time.perf_counter() - t0:.2f}s\n")
    print("-"*50 + '\n')

    print(f"\nSummary:\n{result['summary']}")
    print("\n" + "-" * 50)

    # RAG
    rag_chain = result["rag_chain"]
    print('\nEnter ["exit", "quit", "q"] to exit conversation:\n')
    while True:
        print("\n" + "-" * 50)
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("."*50)
            print("Thank You")
            break
        if not question: # empty spaces
            continue
        answer = ask_question(rag_chain, question)
        print("\n" + "-" * 50)
        print(f"\nVideoSage: {answer}\n")
