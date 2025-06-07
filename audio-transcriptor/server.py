# TODO: Improve transcription time.
# TODO: Add prompt splitting for longer lectures.
# TODO: Add backup DeepSeek integration.
# TODO: Add live subtitling

import datetime
import os
import re
import wave

from faster_whisper import WhisperModel
from flask import Flask, render_template, send_file
from flask_socketio import SocketIO
from g4f import Provider
from g4f.client import Client

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

DATE_FORMAT = '%Y-%m-%d_%H-%M-%S'
BANNED_KEYWORDS = [
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPTS_DIR = os.path.join(BASE_DIR, 'transcripts')

RAW_FILE_PATH = os.path.join(TRANSCRIPTS_DIR, 'raw')
CLEANED_FILE_PATH = os.path.join(TRANSCRIPTS_DIR, 'cleaned')
PROMPT_PATH = os.path.join(BASE_DIR, 'prompt.txt')
TEST_PROMPT_PATH = os.path.join(BASE_DIR, 'test_prompt.txt')
AUDIO_PATH = os.path.join(BASE_DIR, 'lecture.wav')

os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(RAW_FILE_PATH, exist_ok=True)
os.makedirs(CLEANED_FILE_PATH, exist_ok=True)

audio_buffer = bytearray()


def is_response_invalid(text):
    if not text:
        return True

    text_lower = text.lower()

    # Удаляем не-ASCII символы и нормализуем пробелы
    text_ascii = re.sub(r'[^\x00-\x7F]', ' ', text_lower)
    text_ascii = re.sub(r'\s+', ' ', text_ascii)

    # Объединённый текст: оригинальный + ASCII версия
    combined_text = text_lower + '\n' + text_ascii

    # Базовая фильтрация по ключевым словам
    for keyword in BANNED_KEYWORDS:
        if keyword in combined_text:
            return True

    # Проверка на наличие HTML
    html_signatures = [
        '<html',
        '<!doctype html',
        '<body',
        '</html>',
        '</body>',
    ]
    if any(tag in text_lower for tag in html_signatures):
        return True

    # 🚫 Исключаем китайские иероглифы (диапазоны: \u4e00–\u9fff)
    if re.search(r'[\u4e00-\u9fff]', text):
        print('⚠️ Rejected: Chinese characters detected.')
        return True

    return False


def process_transcription():
    print('🔄 Starting background transcription...')

    segments, info = model.transcribe(AUDIO_PATH)
    full_text = ''.join(segment.text for segment in segments)

    lecture_end = datetime.datetime.now().strftime(DATE_FORMAT)
    raw_path = os.path.join(RAW_FILE_PATH, f'transcript_of_{lecture_end}.md')
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write('# Транскрипт лекции\n\n')
        f.write(full_text)
    print('✅ Saved raw transcript.')

    # Try to find a working GPT provider
    client = None
    for provider in Provider.__providers__:
        try:
            print(f'🔍 Testing provider: {provider.__name__}')
            temp_client = Client(provider=provider)

            with open(TEST_PROMPT_PATH, 'r', encoding='utf-8') as f:
                gpt_template = f.read()
            test_prompt = f'{gpt_template.strip()}\n\n{full_text}'

            test_response = temp_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': test_prompt}],
                web_search=False,
            )

            test_content = test_response.choices[0].message.content

            if not test_content:
                print(
                    f'⚠️ Provider {provider.__name__} returned empty content. Skipping.'
                )
                continue
            test_content = str(test_content).strip().lower()

            # Check for suspicious keywords indicating broken access
            if is_response_invalid(test_content):
                print(
                    f'⚠️ Provider {provider.__name__} gave suspicious text. Skipping.'
                )
                continue

            print(f'✅ Provider {provider.__name__} is working!')
            client = temp_client
            break
        except Exception as err_msg:
            # Convert error to string for keyword inspection
            error_str = str(err_msg).lower()

            if (
                '522' in error_str
                or 'cloudflare' in error_str
                or 'timeout' in error_str
            ):
                print(
                    f'⚠️ Provider {provider.__name__} timed out or is behind Cloudflare.'
                )
            else:
                print(f'❌ Provider {provider.__name__} failed: {err_msg}')

    if client:
        try:
            with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
                gpt_template = f.read()
            gpt_prompt = f'{gpt_template.strip()}\n\n{full_text}'

            response = client.chat.completions.create(
                model='gpt-o3',
                messages=[{'role': 'user', 'content': gpt_prompt}],
                web_search=False,
            )
            cleaned = response.choices[0].message.content
            cleaned_path = os.path.join(
                CLEANED_FILE_PATH, f'lecture_from_{lecture_end}.md'
            )
            with open(cleaned_path, 'w', encoding='utf-8') as f:
                f.write('# Конспект лекции\n\n')
                f.write(cleaned)
            print('✅ Saved clean summary.')
        except Exception as err_msg:
            print(f'⚠️ Error while generating summary: {err_msg}')
    else:
        print('⚠️ Skipping summary generation — no available provider.')


print('⏳ Loading faster-whisper model...')
model = WhisperModel('small', device='cpu', compute_type='int8')
print('✅ Model loaded.')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/download')
def download_transcript():
    files = sorted(
        [
            f
            for f in os.listdir(RAW_FILE_PATH)
            if f.startswith('transcript_of_')
        ],
        reverse=True,
    )
    if not files:
        return 'No transcript found', 404
    return send_file(os.path.join(RAW_FILE_PATH, files[0]), as_attachment=True)


@app.route('/download_cleaned')
def download_cleaned():
    files = sorted(
        [
            f
            for f in os.listdir(CLEANED_FILE_PATH)
            if f.startswith('lecture_cleaned_')
        ],
        reverse=True,
    )
    if not files:
        return 'No cleaned transcript found', 404
    return send_file(
        os.path.join(CLEANED_FILE_PATH, files[0]), as_attachment=True
    )


@socketio.on('connect')
def handle_connect():
    print('🔌 Client connected')
    socketio.emit('ready')


@socketio.on('audio')
def handle_audio(data):
    global audio_buffer
    if data:
        audio_buffer += data


@socketio.on('stop')
def handle_stop():
    global audio_buffer

    print('🛑 Received stop. Saving and transcribing...')

    # Save audio immediately
    with wave.open(AUDIO_PATH, 'wb') as lecture_audio:
        lecture_audio.setnchannels(1)
        lecture_audio.setsampwidth(2)
        lecture_audio.setframerate(48000)
        lecture_audio.writeframes(audio_buffer)

    # Clear buffer immediately to allow new recording
    audio_buffer = bytearray()

    # Start background thread for heavy processing
    from threading import Thread

    Thread(target=process_transcription).start()


@socketio.on('disconnect')
def handle_disconnect():
    print('❌ Client disconnected')


if __name__ == '__main__':
    socketio.run(app, port=5000, debug=True)
