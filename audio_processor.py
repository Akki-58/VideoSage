import yt_dlp, os
from pydub import AudioSegment

DOWNLOAD_DIR = "Audio_Downloaded"
os.makedirs(DOWNLOAD_DIR, exist_ok = True)

def download_youtube_audio(url : str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio", # ffmpeg should be downloaded - https://ffmpeg.org/download.html
                "preferredcodec": "mp3", #mp3-> compressed form (so less size), wav -> more storage
                "preferredquality": "64", # 64, 128, 192, 256, 320
            }
        ],
        "quiet": True, #supresses all the download progress log in terminal
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".mp3").replace(".m4a", ".mp3")
    
    return filename

def chunk_audio(mp3_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_mp3(mp3_path)
    chunk_millisec = chunk_minutes*60*1000

    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_millisec)): # len(audio) output in millisec
        chunk = audio[start: start+ chunk_millisec]
        chunk_path = f"{mp3_path}_chunk_{i}.mp3"
        chunk.export(chunk_path, format="mp3")

        chunks.append(chunk_path)
    
    return chunks


def process_audio(src : str, chunk_minutes: int=10) -> list:
    if src.startswith("https://") or src.startswith("http://"):
        # Youtube url
        mp3_path = download_youtube_audio(src)
    else:
        # Local audio file
        mp3_path = src
    
    print(mp3_path)
    chunks = chunk_audio(mp3_path, chunk_minutes)
    print(f"{len(chunks)} chunk(s) created...")
    return chunks

