# # # """
# # # eval_metrics.py

# # # Measures two metrics for VideoSage:

# # # 1. RTF (Real-Time Factor) for transcription
# # #        RTF = transcription_processing_time / audio_duration
# # #        RTF < 1.0 means the pipeline transcribes faster than real time.

# # # 2. RAGAS Faithfulness score for the RAG Q&A system
# # #        Fraction of claims in a generated answer that are actually
# # #        supported by the retrieved transcript context (LLM-judged).
# # #        Directly measures the "grounded, hallucination-minimizing" claim.

# # # Drop this file into the root of the VideoSage repo (next to main.py) and run:

# # #     python eval_metrics.py --source "<youtube_url_or_local_audio_path>" --questions questions.json

# # # questions.json format (ground_truth is optional and unused by faithfulness,
# # # but keep the field if you later want to add answer_correctness etc.):

# # #     [
# # #       {"question": "What is the main topic discussed in the video?"},
# # #       {"question": "What action items were mentioned, if any?"}
# # #     ]

# # # Requires (in addition to VideoSage's existing requirements.txt):
# # #     pip install ragas datasets
# # # """

# # # import argparse
# # # import hashlib
# # # import json
# # # import os
# # # import re
# # # import time

# # # from pydub import AudioSegment

# # # from Core.audio_processor import process_audio
# # # from Core.transcriber import transcribe
# # # from Core.rag_engine import collection_exists, get_prompt, load_llm as load_rag_llm
# # # from Core.vector_store import (
# # #     CHROMA_DIR,
# # #     build_vector_store,
# # #     get_embeddings,
# # #     get_retriever,
# # #     load_vector_store,
# # # )


# # # # ---------------------------------------------------------------------------
# # # # Helpers
# # # # ---------------------------------------------------------------------------

# # # def extract_video_id(source: str) -> str:
# # #     """Mirrors main.py's video_id logic so we reuse the same cache/collection."""
# # #     if source.startswith("http://") or source.startswith("https://"):
# # #         match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([a-zA-Z0-9_-]{11})", source)
# # #         if match:
# # #             return match.group(1)
# # #         raise ValueError("Invalid YouTube URL")
# # #     return hashlib.md5(os.path.abspath(source).encode()).hexdigest()


# # # def get_audio_duration_seconds(mp3_path: str) -> float:
# # #     audio = AudioSegment.from_mp3(mp3_path)
# # #     return len(audio) / 1000.0


# # # # ---------------------------------------------------------------------------
# # # # Metric 1: RTF (Real-Time Factor)
# # # # ---------------------------------------------------------------------------

# # # def measure_rtf(source: str, chunk_minutes: int = 5) -> dict:
# # #     """
# # #     Runs audio extraction + transcription and computes RTF.
# # #     Returns timing info plus the full transcript (reused for the RAGAS step
# # #     so we don't pay for transcription twice).
# # #     """
# # #     t0 = time.perf_counter()
# # #     audio_chunks = process_audio(source, chunk_minutes)
# # #     extract_time = time.perf_counter() - t0

# # #     total_duration = sum(get_audio_duration_seconds(c) for c in audio_chunks)

# # #     transcript_path = "eval_temp_transcript.txt"
# # #     t1 = time.perf_counter()
# # #     transcribe(audio_chunks, output_file=transcript_path)
# # #     transcription_time = time.perf_counter() - t1

# # #     rtf = transcription_time / total_duration if total_duration > 0 else float("nan")

# # #     with open(transcript_path, "r", encoding="utf-8") as f:
# # #         transcript = f.read()

# # #     # Clean up temp chunk files (not the cached transcript)
# # #     for chunk_path in audio_chunks:
# # #         try:
# # #             os.remove(chunk_path)
# # #         except OSError:
# # #             pass

# # #     return {
# # #         "audio_duration_sec": round(total_duration, 2),
# # #         "extract_time_sec": round(extract_time, 2),
# # #         "transcription_time_sec": round(transcription_time, 2),
# # #         "rtf": round(rtf, 4),
# # #         "transcript": transcript,
# # #     }


# # # # ---------------------------------------------------------------------------
# # # # Metric 2: RAGAS Faithfulness
# # # # ---------------------------------------------------------------------------

# # # def collect_rag_samples(transcript: str, video_id: str, questions: list) -> list:
# # #     """
# # #     Runs each test question through VideoSage's actual retriever + prompt,
# # #     capturing (question, answer, retrieved_contexts) — the three fields
# # #     RAGAS faithfulness needs.
# # #     """
# # #     from langchain_core.output_parsers import StrOutputParser

# # #     if collection_exists(video_id):
# # #         vector_store = load_vector_store(video_id)
# # #     else:
# # #         vector_store = build_vector_store(transcript, video_id)

# # #     retriever = get_retriever(vector_store)
# # #     llm = load_rag_llm()
# # #     prompt = get_prompt()
# # #     answer_chain = prompt | llm | StrOutputParser()

# # #     samples = []
# # #     for item in questions:
# # #         question = item["question"]
# # #         docs = retriever.invoke(question)
# # #         contexts = [d.page_content for d in docs]

# # #         answer = answer_chain.invoke({"context": "\n\n".join(contexts), "question": question})

# # #         sample = {"question": question, "answer": answer, "contexts": contexts}
# # #         if "ground_truth" in item:
# # #             sample["ground_truth"] = item["ground_truth"]

# # #         samples.append(sample)

# # #     return samples


# # # def measure_ragas_faithfulness(samples: list) -> dict:
# # #     """
# # #     Scores faithfulness with RAGAS, using the SAME Mistral LLM + embeddings
# # #     VideoSage already uses (so no separate OpenAI key is required).
# # #     """
# # #     from datasets import Dataset
# # #     from ragas import evaluate
# # #     from ragas.embeddings import LangchainEmbeddingsWrapper
# # #     from ragas.llms import LangchainLLMWrapper
# # #     from ragas.metrics import faithfulness

# # #     judge_llm = LangchainLLMWrapper(load_rag_llm())
# # #     judge_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

# # #     dataset = Dataset.from_list(samples)
# # #     result = evaluate(
# # #         dataset,
# # #         metrics=[faithfulness],
# # #         llm=judge_llm,
# # #         embeddings=judge_embeddings,
# # #     )

# # #     df = result.to_pandas()
# # #     return {
# # #         "mean_faithfulness": round(float(df["faithfulness"].mean()), 4),
# # #         "per_question": [
# # #             {"question": row["question"], "faithfulness": round(float(row["faithfulness"]), 4)}
# # #             for _, row in df.iterrows()
# # #         ],
# # #     }


# # # # ---------------------------------------------------------------------------
# # # # Main
# # # # ---------------------------------------------------------------------------

# # # def main():
# # #     parser = argparse.ArgumentParser(description="Measure RTF and RAGAS faithfulness for VideoSage")
# # #     parser.add_argument("--source", required=True, help="YouTube URL or local audio file path")
# # #     parser.add_argument("--questions", required=True, help="Path to a JSON file of test questions")
# # #     parser.add_argument("--chunk-minutes", type=int, default=5, help="Audio chunk size in minutes (matches main.py default)")
# # #     args = parser.parse_args()

# # #     with open(args.questions, "r", encoding="utf-8") as f:
# # #         questions = json.load(f)

# # #     video_id = extract_video_id(args.source)

# # #     print("=" * 60)
# # #     print("Metric 1: RTF (Real-Time Factor)")
# # #     print("=" * 60)
# # #     rtf_result = measure_rtf(args.source, args.chunk_minutes)
# # #     print(f"Audio duration:      {rtf_result['audio_duration_sec']}s")
# # #     print(f"Transcription time:  {rtf_result['transcription_time_sec']}s")
# # #     print(f"RTF:                 {rtf_result['rtf']}  (lower is faster than real-time)")
# # #     print()

# # #     print("=" * 60)
# # #     print("Metric 2: RAGAS Faithfulness")
# # #     print("=" * 60)
# # #     samples = collect_rag_samples(rtf_result["transcript"], video_id, questions)
# # #     faithfulness_result = measure_ragas_faithfulness(samples)
# # #     print(f"Mean Faithfulness Score: {faithfulness_result['mean_faithfulness']}  (0-1, higher = more grounded)")
# # #     for row in faithfulness_result["per_question"]:
# # #         q_preview = row["question"][:55]
# # #         print(f"  - {q_preview:<55} {row['faithfulness']}")

# # #     output = {
# # #         "video_id": video_id,
# # #         "rtf": {k: v for k, v in rtf_result.items() if k != "transcript"},
# # #         "faithfulness": faithfulness_result,
# # #     }
# # #     with open("eval_results.json", "w", encoding="utf-8") as f:
# # #         json.dump(output, f, indent=2)

# # #     print("\nSaved full results to eval_results.json")


# # # if __name__ == "__main__":
# # #     main()
# # """
# # eval_metrics.py

# # Measures two metrics for VideoSage:

# # 1. RTF (Real-Time Factor) for transcription
# #        RTF = transcription_processing_time / audio_duration
# #        RTF < 1.0 means the pipeline transcribes faster than real time.

# # 2. RAGAS Faithfulness score for the RAG Q&A system
# #        Fraction of claims in a generated answer that are actually
# #        supported by the retrieved transcript context (LLM-judged).
# #        Directly measures the "grounded, hallucination-minimizing" claim.

# # Drop this file into the root of the VideoSage repo (next to main.py) and run:

# #     python eval_metrics.py --source "<youtube_url_or_local_audio_path>" --questions questions.json

# # questions.json format (ground_truth is optional and unused by faithfulness,
# # but keep the field if you later want to add answer_correctness etc.):

# #     [
# #       {"question": "What is the main topic discussed in the video?"},
# #       {"question": "What action items were mentioned, if any?"}
# #     ]

# # Requires (in addition to VideoSage's existing requirements.txt):
# #     pip install ragas datasets
# # """

# # import argparse
# # import hashlib
# # import json
# # import os
# # import re
# # import time

# # from pydub import AudioSegment

# # from Core.audio_processor import process_audio
# # from Core.transcriber import transcribe
# # from Core.rag_engine import collection_exists, get_prompt, load_llm as load_rag_llm
# # from Core.vector_store import (
# #     CHROMA_DIR,
# #     build_vector_store,
# #     get_embeddings,
# #     get_retriever,
# #     load_vector_store,
# # )


# # # ---------------------------------------------------------------------------
# # # Helpers
# # # ---------------------------------------------------------------------------

# # def extract_video_id(source: str) -> str:
# #     """Mirrors main.py's video_id logic so we reuse the same cache/collection."""
# #     if source.startswith("http://") or source.startswith("https://"):
# #         match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([a-zA-Z0-9_-]{11})", source)
# #         if match:
# #             return match.group(1)
# #         raise ValueError("Invalid YouTube URL")
# #     return hashlib.md5(os.path.abspath(source).encode()).hexdigest()


# # def get_audio_duration_seconds(mp3_path: str) -> float:
# #     audio = AudioSegment.from_mp3(mp3_path)
# #     return len(audio) / 1000.0


# # # ---------------------------------------------------------------------------
# # # Metric 1: RTF (Real-Time Factor)
# # # ---------------------------------------------------------------------------

# # def measure_rtf(source: str, chunk_minutes: int = 5) -> dict:
# #     """
# #     Runs audio extraction + transcription and computes RTF.
# #     Returns timing info plus the full transcript (reused for the RAGAS step
# #     so we don't pay for transcription twice).
# #     """
# #     t0 = time.perf_counter()
# #     audio_chunks = process_audio(source, chunk_minutes)
# #     extract_time = time.perf_counter() - t0

# #     total_duration = sum(get_audio_duration_seconds(c) for c in audio_chunks)

# #     transcript_path = "eval_temp_transcript.txt"
# #     t1 = time.perf_counter()
# #     transcribe(audio_chunks, output_file=transcript_path)
# #     transcription_time = time.perf_counter() - t1

# #     rtf = transcription_time / total_duration if total_duration > 0 else float("nan")

# #     with open(transcript_path, "r", encoding="utf-8") as f:
# #         transcript = f.read()

# #     # Clean up temp chunk files (not the cached transcript)
# #     for chunk_path in audio_chunks:
# #         try:
# #             os.remove(chunk_path)
# #         except OSError:
# #             pass

# #     return {
# #         "audio_duration_sec": round(total_duration, 2),
# #         "extract_time_sec": round(extract_time, 2),
# #         "transcription_time_sec": round(transcription_time, 2),
# #         "rtf": round(rtf, 4),
# #         "transcript": transcript,
# #     }


# # # ---------------------------------------------------------------------------
# # # Metric 2: RAGAS Faithfulness
# # # ---------------------------------------------------------------------------

# # def collect_rag_samples(transcript: str, video_id: str, questions: list) -> list:
# #     """
# #     Runs each test question through VideoSage's actual retriever + prompt,
# #     capturing (question, answer, retrieved_contexts) — the three fields
# #     RAGAS faithfulness needs.
# #     """
# #     from langchain_core.output_parsers import StrOutputParser

# #     if collection_exists(video_id):
# #         vector_store = load_vector_store(video_id)
# #     else:
# #         vector_store = build_vector_store(transcript, video_id)

# #     retriever = get_retriever(vector_store)
# #     llm = load_rag_llm()
# #     prompt = get_prompt()
# #     answer_chain = prompt | llm | StrOutputParser()

# #     samples = []
# #     for item in questions:
# #         question = item["question"]
# #         docs = retriever.invoke(question)
# #         contexts = [d.page_content for d in docs]

# #         answer = answer_chain.invoke({"context": "\n\n".join(contexts), "question": question})

# #         sample = {"question": question, "answer": answer, "contexts": contexts}
# #         if "ground_truth" in item:
# #             sample["ground_truth"] = item["ground_truth"]

# #         samples.append(sample)

# #     return samples


# # def measure_ragas_faithfulness(samples: list) -> dict:
# #     """
# #     Scores faithfulness with RAGAS, using the SAME Mistral LLM + embeddings
# #     VideoSage already uses (so no separate OpenAI key is required).
# #     """
# #     from datasets import Dataset
# #     from ragas import evaluate
# #     from ragas.embeddings import LangchainEmbeddingsWrapper
# #     from ragas.llms import LangchainLLMWrapper
# #     from ragas.metrics import faithfulness

# #     judge_llm = LangchainLLMWrapper(load_rag_llm())
# #     judge_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

# #     dataset = Dataset.from_list(samples)
# #     result = evaluate(
# #         dataset,
# #         metrics=[faithfulness],
# #         llm=judge_llm,
# #         embeddings=judge_embeddings,
# #     )

# #     df = result.to_pandas()
# #     return {
# #         "mean_faithfulness": round(float(df["faithfulness"].mean()), 4),
# #         "per_question": [
# #             {"question": row["question"], "faithfulness": round(float(row["faithfulness"]), 4)}
# #             for _, row in df.iterrows()
# #         ],
# #     }


# # # ---------------------------------------------------------------------------
# # # Main
# # # ---------------------------------------------------------------------------

# # def main():
# #     parser = argparse.ArgumentParser(description="Measure RTF and RAGAS faithfulness for VideoSage")
# #     parser.add_argument("--source", required=True, help="YouTube URL or local audio file path")
# #     parser.add_argument("--questions", required=True, help="Path to a JSON file of test questions")
# #     parser.add_argument("--chunk-minutes", type=int, default=5, help="Audio chunk size in minutes (matches main.py default)")
# #     args = parser.parse_args()

# #     with open(args.questions, "r", encoding="utf-8") as f:
# #         questions = json.load(f)

# #     video_id = extract_video_id(args.source)

# #     print("=" * 60)
# #     print("Metric 1: RTF (Real-Time Factor)")
# #     print("=" * 60)
# #     rtf_result = measure_rtf(args.source, args.chunk_minutes)
# #     print(f"Audio duration:      {rtf_result['audio_duration_sec']}s")
# #     print(f"Transcription time:  {rtf_result['transcription_time_sec']}s")
# #     print(f"RTF:                 {rtf_result['rtf']}  (lower is faster than real-time)")
# #     print()

# #     print("=" * 60)
# #     print("Metric 2: RAGAS Faithfulness")
# #     print("=" * 60)
# #     samples = collect_rag_samples(rtf_result["transcript"], video_id, questions)
# #     faithfulness_result = measure_ragas_faithfulness(samples)
# #     print(f"Mean Faithfulness Score: {faithfulness_result['mean_faithfulness']}  (0-1, higher = more grounded)")
# #     for row in faithfulness_result["per_question"]:
# #         q_preview = row["question"][:55]
# #         print(f"  - {q_preview:<55} {row['faithfulness']}")

# #     output = {
# #         "video_id": video_id,
# #         "rtf": {k: v for k, v in rtf_result.items() if k != "transcript"},
# #         "faithfulness": faithfulness_result,
# #     }
# #     with open("eval_results.json", "w", encoding="utf-8") as f:
# #         json.dump(output, f, indent=2)

# #     print("\nSaved full results to eval_results.json")


# # if __name__ == "__main__":
# #     main()
# """
# eval_metrics.py

# Measures two metrics for VideoSage across a SET of videos (not just one),
# so the reported numbers are a mean +/- spread rather than a single sample:

# 1. RTF (Real-Time Factor) for transcription
#        RTF = transcription_processing_time / audio_duration
#        RTF < 1.0 means the pipeline transcribes faster than real time.

# 2. RAGAS Faithfulness score for the RAG Q&A system
#        Fraction of claims in a generated answer that are actually
#        supported by the retrieved transcript context (LLM-judged).
#        Directly measures the "grounded, hallucination-minimizing" claim.

# Drop this file into the root of the VideoSage repo (next to main.py).

# --- Recommended: evaluate across multiple videos ---

#     python eval_metrics.py --config eval_config.json

# eval_config.json format (one entry per test video; mix YouTube URLs and
# local file paths freely; 4-6 videos with varied length/content is plenty):

#     [
#       {
#         "source": "https://youtube.com/watch?v=XXXXXXXXXXX",
#         "questions": [
#           {"question": "What is the main topic discussed?"},
#           {"question": "What action items were mentioned, if any?"}
#         ]
#       },
#       {
#         "source": "path/to/local_audio.mp3",
#         "questions": [
#           {"question": "What decisions were made?"}
#         ]
#       }
#     ]

# --- Legacy: evaluate a single video ---

#     python eval_metrics.py --source "<youtube_url_or_local_audio_path>" --questions questions.json

# (questions.json is the same list-of-{"question": ...} format used inside
# each eval_config.json entry above. ground_truth is optional and unused by
# faithfulness, but keep the field if you later add answer_correctness etc.)

# Requires (in addition to VideoSage's existing requirements.txt):
#     pip install ragas datasets
# """

# import argparse
# import hashlib
# import json
# import os
# import re
# import statistics
# import time

# from pydub import AudioSegment

# from Core.audio_processor import process_audio
# from Core.transcriber import transcribe
# from Core.rag_engine import collection_exists, get_prompt, load_llm as load_rag_llm
# from Core.vector_store import (
#     CHROMA_DIR,
#     build_vector_store,
#     get_embeddings,
#     get_retriever,
#     load_vector_store,
# )


# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------

# def extract_video_id(source: str) -> str:
#     """Mirrors main.py's video_id logic so we reuse the same cache/collection."""
#     if source.startswith("http://") or source.startswith("https://"):
#         match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([a-zA-Z0-9_-]{11})", source)
#         if match:
#             return match.group(1)
#         raise ValueError("Invalid YouTube URL")
#     return hashlib.md5(os.path.abspath(source).encode()).hexdigest()


# def get_audio_duration_seconds(mp3_path: str) -> float:
#     audio = AudioSegment.from_mp3(mp3_path)
#     return len(audio) / 1000.0


# # ---------------------------------------------------------------------------
# # Metric 1: RTF (Real-Time Factor)
# # ---------------------------------------------------------------------------

# def measure_rtf(source: str, chunk_minutes: int = 5) -> dict:
#     """
#     Runs audio extraction + transcription and computes RTF.
#     Returns timing info plus the full transcript (reused for the RAGAS step
#     so we don't pay for transcription twice).
#     """
#     t0 = time.perf_counter()
#     audio_chunks = process_audio(source, chunk_minutes)
#     extract_time = time.perf_counter() - t0

#     total_duration = sum(get_audio_duration_seconds(c) for c in audio_chunks)

#     transcript_path = "eval_temp_transcript.txt"
#     t1 = time.perf_counter()
#     transcribe(audio_chunks, output_file=transcript_path)
#     transcription_time = time.perf_counter() - t1

#     rtf = transcription_time / total_duration if total_duration > 0 else float("nan")

#     with open(transcript_path, "r", encoding="utf-8") as f:
#         transcript = f.read()

#     # Clean up temp chunk files (not the cached transcript)
#     for chunk_path in audio_chunks:
#         try:
#             os.remove(chunk_path)
#         except OSError:
#             pass

#     return {
#         "audio_duration_sec": round(total_duration, 2),
#         "extract_time_sec": round(extract_time, 2),
#         "transcription_time_sec": round(transcription_time, 2),
#         "rtf": round(rtf, 4),
#         "transcript": transcript,
#     }


# # ---------------------------------------------------------------------------
# # Metric 2: RAGAS Faithfulness
# # ---------------------------------------------------------------------------

# def collect_rag_samples(transcript: str, video_id: str, questions: list) -> list:
#     """
#     Runs each test question through VideoSage's actual retriever + prompt,
#     capturing (question, answer, retrieved_contexts) — the three fields
#     RAGAS faithfulness needs.
#     """
#     from langchain_core.output_parsers import StrOutputParser

#     if collection_exists(video_id):
#         vector_store = load_vector_store(video_id)
#     else:
#         vector_store = build_vector_store(transcript, video_id)

#     retriever = get_retriever(vector_store)
#     llm = load_rag_llm()
#     prompt = get_prompt()
#     answer_chain = prompt | llm | StrOutputParser()

#     samples = []
#     for item in questions:
#         question = item["question"]
#         docs = retriever.invoke(question)
#         contexts = [d.page_content for d in docs]

#         answer = answer_chain.invoke({"context": "\n\n".join(contexts), "question": question})

#         sample = {"question": question, "answer": answer, "contexts": contexts}
#         if "ground_truth" in item:
#             sample["ground_truth"] = item["ground_truth"]

#         samples.append(sample)

#     return samples


# def measure_ragas_faithfulness(samples: list) -> dict:
#     """
#     Scores faithfulness with RAGAS, using the SAME Mistral LLM + embeddings
#     VideoSage already uses (so no separate OpenAI key is required).
#     """
#     from datasets import Dataset
#     from ragas import evaluate
#     from ragas.embeddings import LangchainEmbeddingsWrapper
#     from ragas.llms import LangchainLLMWrapper
#     from ragas.metrics import faithfulness

#     judge_llm = LangchainLLMWrapper(load_rag_llm())
#     judge_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

#     dataset = Dataset.from_list(samples)
#     result = evaluate(
#         dataset,
#         metrics=[faithfulness],
#         llm=judge_llm,
#         embeddings=judge_embeddings,
#     )

#     df = result.to_pandas()

#     # RAGAS has renamed its output columns across versions:
#     #   older releases: "question" / "answer" / "contexts"
#     #   newer releases: "user_input" / "response" / "retrieved_contexts"
#     # Detect whichever is present instead of hardcoding one.
#     question_col = "question" if "question" in df.columns else "user_input"
#     score_col = "faithfulness"

#     return {
#         "mean_faithfulness": round(float(df[score_col].mean()), 4),
#         "per_question": [
#             {
#                 "question": row[question_col],
#                 "faithfulness": round(float(row[score_col]), 4),
#             }
#             for _, row in df.iterrows()
#         ],
#     }


# # ---------------------------------------------------------------------------
# # Per-video evaluation
# # ---------------------------------------------------------------------------

# def evaluate_video(source: str, questions: list, chunk_minutes: int = 5) -> dict:
#     """Runs both metrics for a single video and returns its results."""
#     video_id = extract_video_id(source)

#     print(f"\n--- {source} ---")

#     rtf_result = measure_rtf(source, chunk_minutes)
#     print(f"  RTF: {rtf_result['rtf']}  ({rtf_result['audio_duration_sec']}s audio, "
#           f"{rtf_result['transcription_time_sec']}s to transcribe)")

#     samples = collect_rag_samples(rtf_result["transcript"], video_id, questions)
#     faithfulness_result = measure_ragas_faithfulness(samples)
#     print(f"  Faithfulness: {faithfulness_result['mean_faithfulness']}  "
#           f"(over {len(questions)} questions)")

#     return {
#         "video_id": video_id,
#         "source": source,
#         "rtf": {k: v for k, v in rtf_result.items() if k != "transcript"},
#         "faithfulness": faithfulness_result,
#     }


# def aggregate(video_results: list) -> dict:
#     """Mean + spread across videos. stdev needs n>=2; falls back to 0.0 for n=1."""
#     rtf_values = [v["rtf"]["rtf"] for v in video_results]
#     faith_values = [v["faithfulness"]["mean_faithfulness"] for v in video_results]

#     def mean_and_std(values):
#         mean = round(statistics.mean(values), 4)
#         std = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
#         return mean, std

#     rtf_mean, rtf_std = mean_and_std(rtf_values)
#     faith_mean, faith_std = mean_and_std(faith_values)

#     return {
#         "num_videos": len(video_results),
#         "rtf_mean": rtf_mean,
#         "rtf_std": rtf_std,
#         "rtf_min": round(min(rtf_values), 4),
#         "rtf_max": round(max(rtf_values), 4),
#         "faithfulness_mean": faith_mean,
#         "faithfulness_std": faith_std,
#         "faithfulness_min": round(min(faith_values), 4),
#         "faithfulness_max": round(max(faith_values), 4),
#     }


# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------

# def main():
#     parser = argparse.ArgumentParser(description="Measure RTF and RAGAS faithfulness for VideoSage")
#     parser.add_argument("--config", help="Path to a JSON file listing multiple videos (recommended)")
#     parser.add_argument("--source", help="Single video: YouTube URL or local audio file path")
#     parser.add_argument("--questions", help="Single video: path to a JSON file of test questions")
#     parser.add_argument("--chunk-minutes", type=int, default=5, help="Audio chunk size in minutes (matches main.py default)")
#     args = parser.parse_args()

#     if not args.config and not (args.source and args.questions):
#         parser.error("Provide either --config (multi-video, recommended) or --source + --questions (single video)")

#     if args.config:
#         with open(args.config, "r", encoding="utf-8") as f:
#             video_configs = json.load(f)
#     else:
#         with open(args.questions, "r", encoding="utf-8") as f:
#             questions = json.load(f)
#         video_configs = [{"source": args.source, "questions": questions}]

#     print("=" * 60)
#     print(f"Evaluating {len(video_configs)} video(s)")
#     print("=" * 60)

#     video_results = []
#     for cfg in video_configs:
#         result = evaluate_video(cfg["source"], cfg["questions"], args.chunk_minutes)
#         video_results.append(result)

#     output = {"videos": video_results}

#     print("\n" + "=" * 60)
#     if len(video_results) > 1:
#         agg = aggregate(video_results)
#         output["aggregate"] = agg
#         print(f"Aggregate over {agg['num_videos']} videos")
#         print("=" * 60)
#         print(f"RTF:          {agg['rtf_mean']} +/- {agg['rtf_std']}  "
#               f"(range {agg['rtf_min']}-{agg['rtf_max']})")
#         print(f"Faithfulness: {agg['faithfulness_mean']} +/- {agg['faithfulness_std']}  "
#               f"(range {agg['faithfulness_min']}-{agg['faithfulness_max']})")
#     else:
#         print("Single video evaluated — run with --config and multiple videos")
#         print("for a mean +/- spread instead of a single sample.")

#     with open("eval_results.json", "w", encoding="utf-8") as f:
#         json.dump(output, f, indent=2)

#     print("\nSaved full results to eval_results.json")


# if __name__ == "__main__":
#     main()
"""
eval_metrics.py

Measures two metrics for VideoSage across a SET of videos (not just one),
so the reported numbers are a mean +/- spread rather than a single sample:

1. RTF (Real-Time Factor) for transcription
       RTF = transcription_processing_time / audio_duration
       RTF < 1.0 means the pipeline transcribes faster than real time.

2. RAGAS Faithfulness score for the RAG Q&A system
       Fraction of claims in a generated answer that are actually
       supported by the retrieved transcript context (LLM-judged).
       Directly measures the "grounded, hallucination-minimizing" claim.

Drop this file into the root of the VideoSage repo (next to main.py).

--- Recommended: evaluate across multiple videos ---

    python eval_metrics.py --config eval_config.json

eval_config.json format (one entry per test video; mix YouTube URLs and
local file paths freely; 4-6 videos with varied length/content is plenty):

    [
      {
        "source": "https://youtube.com/watch?v=XXXXXXXXXXX",
        "questions": [
          {"question": "What is the main topic discussed?"},
          {"question": "What action items were mentioned, if any?"}
        ]
      },
      {
        "source": "path/to/local_audio.mp3",
        "questions": [
          {"question": "What decisions were made?"}
        ]
      }
    ]

--- Legacy: evaluate a single video ---

    python eval_metrics.py --source "<youtube_url_or_local_audio_path>" --questions questions.json

(questions.json is the same list-of-{"question": ...} format used inside
each eval_config.json entry above. ground_truth is optional and unused by
faithfulness, but keep the field if you later add answer_correctness etc.)

Requires (in addition to VideoSage's existing requirements.txt):
    pip install ragas datasets
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import time

from pydub import AudioSegment

from Core.audio_processor import process_audio
from Core.transcriber import transcribe
from Core.rag_engine import collection_exists, get_prompt, load_llm as load_rag_llm
from Core.vector_store import (
    CHROMA_DIR,
    build_vector_store,
    get_embeddings,
    get_retriever,
    load_vector_store,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_video_id(source: str) -> str:
    """Mirrors main.py's video_id logic so we reuse the same cache/collection."""
    if source.startswith("http://") or source.startswith("https://"):
        match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([a-zA-Z0-9_-]{11})", source)
        if match:
            return match.group(1)
        raise ValueError("Invalid YouTube URL")
    return hashlib.md5(os.path.abspath(source).encode()).hexdigest()


def get_audio_duration_seconds(mp3_path: str) -> float:
    audio = AudioSegment.from_mp3(mp3_path)
    return len(audio) / 1000.0


# ---------------------------------------------------------------------------
# Metric 1: RTF (Real-Time Factor)
# ---------------------------------------------------------------------------

def measure_rtf(source: str, chunk_minutes: int = 5) -> dict:
    """
    Runs audio extraction + transcription and computes RTF.
    Returns timing info plus the full transcript (reused for the RAGAS step
    so we don't pay for transcription twice).
    """
    t0 = time.perf_counter()
    audio_chunks = process_audio(source, chunk_minutes)
    extract_time = time.perf_counter() - t0

    total_duration = sum(get_audio_duration_seconds(c) for c in audio_chunks)

    transcript_path = "eval_temp_transcript.txt"
    t1 = time.perf_counter()
    transcribe(audio_chunks, output_file=transcript_path)
    transcription_time = time.perf_counter() - t1

    rtf = transcription_time / total_duration if total_duration > 0 else float("nan")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    # Clean up temp chunk files (not the cached transcript)
    for chunk_path in audio_chunks:
        try:
            os.remove(chunk_path)
        except OSError:
            pass

    return {
        "audio_duration_sec": round(total_duration, 2),
        "extract_time_sec": round(extract_time, 2),
        "transcription_time_sec": round(transcription_time, 2),
        "rtf": round(rtf, 4),
        "transcript": transcript,
    }


# ---------------------------------------------------------------------------
# Metric 2: RAGAS Faithfulness
# ---------------------------------------------------------------------------

def collect_rag_samples(transcript: str, video_id: str, questions: list) -> list:
    """
    Runs each test question through VideoSage's actual retriever + prompt,
    capturing (question, answer, retrieved_contexts) — the three fields
    RAGAS faithfulness needs.
    """
    from langchain_core.output_parsers import StrOutputParser

    if collection_exists(video_id):
        vector_store = load_vector_store(video_id)
    else:
        vector_store = build_vector_store(transcript, video_id)

    retriever = get_retriever(vector_store)
    llm = load_rag_llm()
    prompt = get_prompt()
    answer_chain = prompt | llm | StrOutputParser()

    samples = []
    for item in questions:
        question = item["question"]
        docs = retriever.invoke(question)
        contexts = [d.page_content for d in docs]

        answer = answer_chain.invoke({"context": "\n\n".join(contexts), "question": question})

        sample = {"question": question, "answer": answer, "contexts": contexts}
        if "ground_truth" in item:
            sample["ground_truth"] = item["ground_truth"]

        samples.append(sample)

    return samples


def measure_ragas_faithfulness(samples: list) -> dict:
    """
    Scores faithfulness with RAGAS, using the SAME Mistral LLM + embeddings
    VideoSage already uses (so no separate OpenAI key is required).

    NOTE on robustness: RAGAS has repeatedly renamed its dataset columns
    across releases (question/answer/contexts -> user_input/response/
    retrieved_contexts, with more churn planned before v1.0). Rather than
    guessing which naming your installed version uses, we get the question
    text back from our OWN `samples` list (same order fed into the
    Dataset) and only look up the numeric score column in RAGAS's output.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import faithfulness

    judge_llm = LangchainLLMWrapper(load_rag_llm())
    judge_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    dataset = Dataset.from_list(samples)
    result = evaluate(
        dataset,
        metrics=[faithfulness],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    df = result.to_pandas()

    # Resolve the score column. It's named "faithfulness" in essentially
    # every RAGAS release; fall back to "whatever numeric column isn't one
    # of the known input fields" in case a future release renames it too.
    known_input_cols = {
        "question", "answer", "contexts",
        "user_input", "response", "retrieved_contexts",
        "reference", "ground_truth",
    }
    if "faithfulness" in df.columns:
        score_col = "faithfulness"
    else:
        candidates = [c for c in df.columns if c not in known_input_cols]
        if len(candidates) != 1:
            raise RuntimeError(
                "Could not identify the faithfulness score column in RAGAS's "
                f"output. Columns returned: {list(df.columns)}. "
                "Inspect `df` and adjust `known_input_cols` above."
            )
        score_col = candidates[0]

    scores = df[score_col].tolist()

    # Pair scores with OUR OWN samples (by position) instead of trusting
    # RAGAS's input-column naming for the question text.
    per_question = [
        {"question": sample["question"], "faithfulness": round(float(score), 4)}
        for sample, score in zip(samples, scores)
    ]

    return {
        "mean_faithfulness": round(float(df[score_col].mean()), 4),
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# Per-video evaluation
# ---------------------------------------------------------------------------

def evaluate_video(source: str, questions: list, chunk_minutes: int = 5) -> dict:
    """Runs both metrics for a single video and returns its results."""
    video_id = extract_video_id(source)

    print(f"\n--- {source} ---")

    rtf_result = measure_rtf(source, chunk_minutes)
    print(f"  RTF: {rtf_result['rtf']}  ({rtf_result['audio_duration_sec']}s audio, "
          f"{rtf_result['transcription_time_sec']}s to transcribe)")

    samples = collect_rag_samples(rtf_result["transcript"], video_id, questions)
    faithfulness_result = measure_ragas_faithfulness(samples)
    print(f"  Faithfulness: {faithfulness_result['mean_faithfulness']}  "
          f"(over {len(questions)} questions)")

    return {
        "video_id": video_id,
        "source": source,
        "rtf": {k: v for k, v in rtf_result.items() if k != "transcript"},
        "faithfulness": faithfulness_result,
    }


def aggregate(video_results: list) -> dict:
    """Mean + spread across videos. stdev needs n>=2; falls back to 0.0 for n=1."""
    rtf_values = [v["rtf"]["rtf"] for v in video_results]
    faith_values = [v["faithfulness"]["mean_faithfulness"] for v in video_results]

    def mean_and_std(values):
        mean = round(statistics.mean(values), 4)
        std = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
        return mean, std

    rtf_mean, rtf_std = mean_and_std(rtf_values)
    faith_mean, faith_std = mean_and_std(faith_values)

    return {
        "num_videos": len(video_results),
        "rtf_mean": rtf_mean,
        "rtf_std": rtf_std,
        "rtf_min": round(min(rtf_values), 4),
        "rtf_max": round(max(rtf_values), 4),
        "faithfulness_mean": faith_mean,
        "faithfulness_std": faith_std,
        "faithfulness_min": round(min(faith_values), 4),
        "faithfulness_max": round(max(faith_values), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Measure RTF and RAGAS faithfulness for VideoSage")
    parser.add_argument("--config", help="Path to a JSON file listing multiple videos (recommended)")
    parser.add_argument("--source", help="Single video: YouTube URL or local audio file path")
    parser.add_argument("--questions", help="Single video: path to a JSON file of test questions")
    parser.add_argument("--chunk-minutes", type=int, default=5, help="Audio chunk size in minutes (matches main.py default)")
    args = parser.parse_args()

    if not args.config and not (args.source and args.questions):
        parser.error("Provide either --config (multi-video, recommended) or --source + --questions (single video)")

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            video_configs = json.load(f)
    else:
        with open(args.questions, "r", encoding="utf-8") as f:
            questions = json.load(f)
        video_configs = [{"source": args.source, "questions": questions}]

    print("=" * 60)
    print(f"Evaluating {len(video_configs)} video(s)")
    print("=" * 60)

    video_results = []
    for cfg in video_configs:
        result = evaluate_video(cfg["source"], cfg["questions"], args.chunk_minutes)
        video_results.append(result)

    output = {"videos": video_results}

    print("\n" + "=" * 60)
    if len(video_results) > 1:
        agg = aggregate(video_results)
        output["aggregate"] = agg
        print(f"Aggregate over {agg['num_videos']} videos")
        print("=" * 60)
        print(f"RTF:          {agg['rtf_mean']} +/- {agg['rtf_std']}  "
              f"(range {agg['rtf_min']}-{agg['rtf_max']})")
        print(f"Faithfulness: {agg['faithfulness_mean']} +/- {agg['faithfulness_std']}  "
              f"(range {agg['faithfulness_min']}-{agg['faithfulness_max']})")
    else:
        print("Single video evaluated — run with --config and multiple videos")
        print("for a mean +/- spread instead of a single sample.")

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\nSaved full results to eval_results.json")


if __name__ == "__main__":
    main()