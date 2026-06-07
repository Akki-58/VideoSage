# VideoSage 🎥🧠

AI-Powered YouTube & Audio Summarization + RAG Q&A System

VideoSage is an AI-powered application that can:

* Extract audio from YouTube videos or local files
* Transcribe speech into text
* Generate concise AI summaries
* Build a Retrieval-Augmented Generation (RAG) pipeline
* Allow interactive question-answering over video transcripts

Built using Python, Whisper/OpenAI-style transcription workflows, embeddings, and vector databases.

---

# 🚀 Features

✅ YouTube video support
✅ Local audio file support
✅ Automatic transcription generation
✅ AI-powered summarization
✅ RAG-based contextual Q&A
✅ Persistent transcript storage
✅ Vector database integration
✅ Modular architecture
✅ Interactive CLI chatbot

---

# 📂 Project Structure

```bash
VideoSage/
│
├── Core/
│   ├── audio_processor.py     # Audio extraction & chunking
│   ├── extractor.py           # YouTube/audio extraction utilities
│   ├── rag_engine.py          # RAG pipeline + QA
│   ├── summarizer.py          # AI summarization
│   ├── transcriber.py         # Speech-to-text transcription
│   └── vector_store.py        # Embedding & vector DB handling
│
├── Audio_Downloaded/          # Downloaded audio files
├── Transcripts_Downloaded/    # Generated transcripts
├── vector_db/                 # Stored embeddings/vector DB
├── Tests/                     # Test files
│
├── .env
├── .env.example
├── .gitignore
├── LICENSE
├── main.py                    # Main application entry point
├── requirements.txt
└── README.md
```

---

# ⚙️ Tech Stack

* Python
* LangChain
* RAG (Retrieval-Augmented Generation)
* Vector Database
* Whisper/OpenAI APIs
* YouTube Processing
* Embeddings
* dotenv

---

# 🔄 Workflow

```text
YouTube URL / Audio File
            │
            ▼
    Audio Processing
            │
            ▼
      Transcription
            │
            ▼
      AI Summarization
            │
            ▼
   Embedding Generation
            │
            ▼
       Vector Store
            │
            ▼
      RAG Q&A System
```

---

# 🛠️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/VideoSage.git
cd VideoSage
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Configure Environment Variables

Create a `.env` file.
See `.env.example` file to setup API Keys.

---

# ▶️ Running the Project

```bash
python main.py
```

---

# 📥 Input Options

The application supports:

## YouTube URL (Videos, Shorts)

```text
https://www.youtube.com/watch?v=example
```

## Local Audio File

```text
sample.mp3
```

---

# 💡 Example Usage

```bash
Enter YouTube URL or local file path:
https://youtube.com/shorts/h8pIMQVvtaM
```

---

# 🧠 RAG Question Answering

After processing completes, you can ask questions interactively:

```text
You: What is the main topic discussed?
VideoSage: The video mainly discusses...
```

Exit conversation using:

```text
exit
quit
q
```

---

# 📌 Core Functionalities

## 🎵 Audio Processing

* Downloads/extracts audio
* Splits audio into chunks
* Optimized for transcription pipelines

---

## 📝 Transcription

* Converts speech to text
* Stores transcripts locally
* Avoids redundant processing

---

## ✨ Summarization

* Generates concise AI summaries
* Quickly captures key insights

---

## 🔎 RAG Pipeline

* Creates embeddings from transcript chunks
* Stores vectors in vector DB
* Retrieves relevant context for answering queries

---

# 📊 Performance Optimizations

* Transcript caching
* Chunked processing
* Persistent vector storage
* Reusable embeddings
* Modular architecture

---

# 📸 Sample Output

```text
--------------------------------------------------
TRANSCRIPT (first 500 char)
--------------------------------------------------
Today we are going to discuss...
--------------------------------------------------

Summary:
This video explains...

You: What are the key points?
VideoSage: The key points are...
```

---

# 🤝 Contributing

Contributions are welcome!

```bash
Fork the repo
Create a feature branch
Commit changes
Submit a pull request
```

---

# 📄 License

This project is licensed under the MIT License.
