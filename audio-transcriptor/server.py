"""server.py.

===============================================================================
Real-time audio transcription backend built with **Flask-SocketIO** and
`faster-whisper`.

The server accepts raw PCM chunks from the browser, performs streaming
speech-to-text, and sends two kinds of WebSocket events back to the client:

* **``partial_text``** – updates the current subtitle line in place.
* **``text``** – appends the finished subtitle line to the full transcript.

After the user stops recording the full audio is saved, transcribed again for
maximum accuracy, and (optionally) passed to a GPT-like model to build a tidy
lecture summary.

This file follows **PEP 8** coding style and **PEP 257** docstring conventions.
===============================================================================

TODO
-------------------------------------------------------------------------------
* Improve transcription time.
* Add prompt splitting for longer lectures.
* Add backup DeepSeek integration.
* Add live subtitling
-------------------------------------------------------------------------------
"""

from __future__ import annotations

# Standard library imports. ---------------------------------------------------
import datetime
import os
import re
import wave
from queue import Queue
from threading import Thread
from typing import List

# Third-party imports. --------------------------------------------------------
import numpy as np
from faster_whisper import WhisperModel
from flask import Flask, render_template, send_file
from flask_socketio import SocketIO
from g4f import Provider
from g4f.client import Client

# ----------------------------------------------------------------------------‐
# Flask application setup.
# ----------------------------------------------------------------------------‐
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

# ----------------------------------------------------------------------------‐
# Constants.
# ----------------------------------------------------------------------------‐
DATE_FORMAT: str = '%Y-%m-%d_%H-%M-%S'

WHISPER_MODEL_NAME = os.getenv('WHISPER_MODEL', 'small')
GPT_MODEL_NAME = 'gpt-4o'

# Audio stream configuration (48 kHz, 16-bit mono PCM). -----------------------
FRAME_RATE: int = 48_000  # Hz – provided by Web Audio API.
BYTES_PER_SAMPLE: int = 4  # 16-bit PCM => 2 bytes per sample.
CHUNK_SECONDS: float = 1.2  # Time window for each recognition pass.
TAIL_SECONDS: float = 0.5  # Overlap between windows for context.
BEAM_SIZE: int = 5  # Beam size to use for decoding.

# Simple heuristic filters for obviously broken GPT responses. ----------------
BANNED_KEYWORDS: List[str] = [
    'api key',
    'access denied',
    'unauthorized',
    'subscription expired',
    'forbidden',
    'sign in',
    'upgrade',
    'buy key',
    '<!-- generated images start -->',
]

# Path configuration. ---------------------------------------------------------
# Directories.
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPTS_DIR: str = os.path.join(BASE_DIR, 'transcripts')
RAW_DIR: str = os.path.join(TRANSCRIPTS_DIR, 'raw')
CLEANED_DIR: str = os.path.join(TRANSCRIPTS_DIR, 'cleaned')

# Files.
PROMPT_PATH: str = os.path.join(BASE_DIR, 'prompt.txt')
TEST_PROMPT_PATH: str = os.path.join(BASE_DIR, 'test_prompt.txt')
AUDIO_PATH: str = os.path.join(BASE_DIR, 'lecture.wav')

# Ensure required folders exist. ----------------------------------------------
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)

# ----------------------------------------------------------------------------‐
# Global runtime state.
# ----------------------------------------------------------------------------‐
audio_buffer = bytearray()  # Stores the full lecture until “stop”.
audio_queue: 'Queue[bytes]' = Queue()  # Streaming worker input queue.
stream_buffer = bytearray()  # Buffer for the current 4-second window.
last_sent_end: float = 0.0  # Deduplication helper between windows.


# ----------------------------------------------------------------------------‐
# Utility functions.
# ----------------------------------------------------------------------------‐
def is_response_invalid(text: str) -> bool:
    """Return *True* if *text* contains obvious signs of an invalid GPT reply.

    The check is intentionally simple – only a handful of keywords, HTML tags
    and Chinese ideographs are filtered out.  This function is used to skip
    broken providers during the «find a free GPT» loop.

    Parameters
    ----------
    text:
        Response candidate returned by a GPT provider.

    Returns
    -------
    bool
        *True* if the response should be discarded, *False* otherwise.
    """
    if not text:
        return True

    text_lower = text.lower()

    # Strip non-ASCII characters and collapse whitespace for easier matching. -
    text_ascii = re.sub(r'[^\x00-\x7F]', ' ', text_lower)
    text_ascii = re.sub(r'\s+', ' ', text_ascii)

    combined = f'{text_lower}\n{text_ascii}'
    if any(keyword in combined for keyword in BANNED_KEYWORDS):
        return True

    # Basic HTML detection. ---------------------------------------------------
    html_tokens = ('<html', '<!doctype html', '<body', '</html>', '</body>')
    if any(token in text_lower for token in html_tokens):
        return True

    # Exclude Chinese ideographs (U+4E00–U+9FFF). -----------------------------
    if re.search(r'[\u4e00-\u9fff]', text):
        print('⚠️  Rejected: Chinese characters detected in GPT response.')
        return True

    return False


# ----------------------------------------------------------------------------‐
# Background worker – low-latency subtitles.
# ----------------------------------------------------------------------------‐
def streaming_worker() -> None:
    """Continuously transcribe small overlapping windows from *audio_queue*.

    For every finished segment two events are emitted:

    * ``partial_text`` – replaces the current subtitle (shown in *italics*).
    * ``text`` – appends the segment to the growing full transcript.
    """
    global stream_buffer, last_sent_end

    while True:
        # 1. Wait for a new raw PCM chunk from the WebSocket. -----------------
        data: bytes = audio_queue.get()
        stream_buffer.extend(data)

        # 2. Check if we have enough audio for one window. --------------------
        target_bytes = int(CHUNK_SECONDS * FRAME_RATE * BYTES_PER_SAMPLE)
        if len(stream_buffer) < target_bytes:
            continue  # Collect more data first.

        # 3. Convert raw int16 PCM to float32 in the −1.0 … 1.0 range. --------
        pcm = np.frombuffer(stream_buffer[:target_bytes], dtype=np.int16)
        float_audio = pcm.astype(np.float32) / 32768.0

        # 4. Run Whisper (small beam, VAD enabled for speed). -----------------
        segments, _info = model.transcribe(
            float_audio,
            beam_size=BEAM_SIZE,
            vad_filter=True,
            language='ru',
        )

        # 5. Emit segments to the client. -------------------------------------
        for seg in segments:
            # Deduplicate overlapping segments (0.05 s tolerance).
            if seg.end <= last_sent_end + 0.05:
                continue

            payload = {
                'text': seg.text.strip(),
                'time': f'{seg.start:.1f}-{seg.end:.1f}',
            }

            socketio.emit('partial_text', payload)  # Update current line.
            socketio.emit('text', payload)  # Append to transcript.

            last_sent_end = seg.end

        # 6. Keep a small tail of audio for the next window. ------------------
        tail_bytes = int(TAIL_SECONDS * FRAME_RATE * BYTES_PER_SAMPLE)
        stream_buffer = stream_buffer[-tail_bytes:]


# ----------------------------------------------------------------------------‐
# Heavy post-processing after the «Stop» button.
# ----------------------------------------------------------------------------‐
def process_transcription() -> None:
    """Generate raw markdown and (optionally) a GPT-cleaned summary.

    This function runs in a background thread so the UI stays responsive.
    """
    print('🔄  Starting background transcription …')

    # 1. Full Whisper pass (no windowing, highest quality). -------------------
    segments, _info = model.transcribe(AUDIO_PATH)
    full_text = ''.join(segment.text for segment in segments)

    # 2. Save the raw transcript. ---------------------------------------------
    lecture_end = datetime.datetime.now().strftime(DATE_FORMAT)
    raw_path = os.path.join(RAW_DIR, f'transcript_of_{lecture_end}.md')
    with open(raw_path, 'w', encoding='utf-8') as file:
        file.write('# Транскрипт лекции\n\n')
        file.write(full_text)
    print('✅  Saved raw transcript.')

    # 3. Try to find a working free-GPT provider. -----------------------------
    client: Client | None = None

    for provider in Provider.__providers__:
        try:
            print(f'🔍  Testing provider: {provider.__name__}')
            temp_client = Client(provider=provider)

            with open(TEST_PROMPT_PATH, 'r', encoding='utf-8') as file:
                gpt_template = file.read()

            test_prompt = f'{gpt_template.strip()}\n\n{full_text}'
            test_resp = temp_client.chat.completions.create(
                model=GPT_MODEL_NAME,
                messages=[{'role': 'user', 'content': test_prompt}],
                web_search=False,
            )
            test_text = test_resp.choices[0].message.content or ''

            if is_response_invalid(test_text):
                print(f'⚠️  Provider {provider.__name__} rejected.')
                continue

            print(f'✅  Provider {provider.__name__} selected.')
            client = temp_client
            break
        except Exception as error_info:
            error = str(error_info).lower()
            if any(tok in error for tok in ('522', 'cloudflare', 'timeout')):
                print(f'⚠️  Provider {provider.__name__} timed out.')
            else:
                print(f'❌  Provider {provider.__name__} failed: {error_info}')

    # 4. Generate a cleaned summary if a provider was found. ------------------
    if client is None:
        print('⚠️  Skipping summary – no working GPT provider found.')
        return

    try:
        with open(PROMPT_PATH, 'r', encoding='utf-8') as file:
            gpt_template = file.read()

        gpt_prompt = f'{gpt_template.strip()}\n\n{full_text}'
        resp = client.chat.completions.create(
            model=GPT_MODEL_NAME,
            messages=[{'role': 'user', 'content': gpt_prompt}],
            web_search=False,
        )
        cleaned = resp.choices[0].message.content or ''

        cleaned_path = os.path.join(
            CLEANED_DIR,
            f'lecture_from_{lecture_end}.md',
        )
        with open(cleaned_path, 'w', encoding='utf-8') as file:
            file.write('# Конспект лекции\n\n')
            file.write(cleaned)
        print('✅  Saved cleaned summary.')
    except Exception as error_info:
        print(f'⚠️  Error while generating summary: {error_info}')


# ----------------------------------------------------------------------------‐
# Whisper model initialisation & stream thread.
# ----------------------------------------------------------------------------‐
print('⏳  Loading faster-whisper model …')
model = WhisperModel(WHISPER_MODEL_NAME, device='cpu', compute_type='int8')
print('✅  Model loaded.')
Thread(target=streaming_worker, daemon=True).start()


# ----------------------------------------------------------------------------‐
# Flask routes.
# ----------------------------------------------------------------------------‐
@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')


@app.route('/download')
def download_transcript():
    """Send the latest raw transcript (.md) as a file download."""
    files = sorted(
        (f for f in os.listdir(RAW_DIR) if f.startswith('transcript_of_')),
        reverse=True,
    )
    if not files:
        return 'No transcript found', 404

    return send_file(os.path.join(RAW_DIR, files[0]), as_attachment=True)


@app.route('/download_cleaned')
def download_cleaned():
    """Send the latest cleaned summary (.md) as a file download."""
    files = sorted(
        (f for f in os.listdir(CLEANED_DIR) if f.startswith('lecture_from_')),
        reverse=True,
    )
    if not files:
        return 'No cleaned transcript found', 404

    return send_file(os.path.join(CLEANED_DIR, files[0]), as_attachment=True)


# ----------------------------------------------------------------------------‐
# WebSocket event handlers.
# ----------------------------------------------------------------------------‐
@socketio.on('connect')
def handle_connect():
    """Notify the client that the server is ready to record."""
    print('🔌  Client connected')
    socketio.emit('ready')


@socketio.on('audio')
def handle_audio(data: bytes):
    """Receive raw PCM chunks and feed them to *audio_queue*."""
    global audio_buffer

    if data:
        audio_buffer += data
        audio_queue.put(data)


@socketio.on('stop')
def handle_stop():
    """Finish the recording, persist audio, and start heavy processing."""
    global audio_buffer

    print('🛑  Received stop. Persisting audio and starting post-processing …')

    # 1. Persist raw PCM into a WAV file. -------------------------------------
    with wave.open(AUDIO_PATH, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(FRAME_RATE)
        wav_file.writeframes(audio_buffer)

    audio_buffer = bytearray()  # Clear buffer for the next recording.

    # 2. Launch the heavy transcription in the background. --------------------
    Thread(target=process_transcription, daemon=True).start()


@socketio.on('disconnect')
def handle_disconnect():
    """Log client disconnection."""
    print('❌  Client disconnected')


# ----------------------------------------------------------------------------‐
# Main entry-point.
# ----------------------------------------------------------------------------‐
if __name__ == '__main__':
    socketio.run(app, port=5000, debug=True)
