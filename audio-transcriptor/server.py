# TODO: Improve transcription time.
# TODO: Add prompt splitting for longer lectures.
# TODO: Add better provider switching.
# TODO: Add backup DeepSeek integration.
# TODO: Add live subtitling

import datetime
import os
import wave

from faster_whisper import WhisperModel
from flask import Flask, render_template, send_file
from flask_socketio import SocketIO
from g4f import Provider
from g4f.client import Client

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

DATE_FORMAT = '%Y-%m-%d_%H-%M-%S'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPTS_DIR = os.path.join(BASE_DIR, 'transcripts')

RAW_FILE_PATH = os.path.join(TRANSCRIPTS_DIR, 'raw')
CLEANED_FILE_PATH = os.path.join(TRANSCRIPTS_DIR, 'cleaned')
PROMPT_PATH = os.path.join(BASE_DIR, 'prompt.txt')
AUDIO_PATH = os.path.join(BASE_DIR, 'lecture.wav')

os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(RAW_FILE_PATH, exist_ok=True)
os.makedirs(CLEANED_FILE_PATH, exist_ok=True)

audio_buffer = bytearray()

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

    with wave.open(AUDIO_PATH, 'wb') as lecture_audio:
        lecture_audio.setnchannels(1)
        lecture_audio.setsampwidth(2)
        lecture_audio.setframerate(48000)
        lecture_audio.writeframes(audio_buffer)

    segments, info = model.transcribe(AUDIO_PATH)
    full_text = ''.join(segment.text for segment in segments)

    lecture_end = datetime.datetime.now().strftime(DATE_FORMAT)
    raw_path = os.path.join(RAW_FILE_PATH, f'transcript_of_{lecture_end}.md')
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write('# Транскрипт лекции\n\n')
        f.write(full_text)
    print('✅ Saved raw transcript.')

    # Seek valid provider.
    client = None
    for provider in Provider.__providers__:
        try:
            print(f'🔍 Testing provider: {provider.__name__}')
            temp_client = Client(provider=provider)
            _ = temp_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': 'Проверка'}],
                web_search=False,
            )
            print(f'✅ Provider {provider.__name__} is working!')
            client = temp_client
            break
        except Exception as err_msg:
            print(
                f'❌ Provider {provider.__name__} is not working!: {err_msg}'
            )

    if not client:
        print('❌ Cannot find a working GPT provider.')

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

    audio_buffer = bytearray()


@socketio.on('disconnect')
def handle_disconnect():
    print('❌ Client disconnected')


if __name__ == '__main__':
    socketio.run(app, port=5000, debug=True)
