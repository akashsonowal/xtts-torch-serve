import time
import requests
import io
from pydub import AudioSegment

url = 'http://localhost:8000/predictions/xttsv2'
input_text = 'Hello, I am Akash Sonowal'

# Set up the headers
headers = {
    'Content-Type': 'application/json'
}

# Prepare the JSON payload
payload = {"data": input_text}

# Record the start time before making the request
start_time = time.time()

# Send the POST request with streaming enabled
response = requests.post(url, headers=headers, json=payload, stream=True, timeout=20)

# Create an in-memory bytes buffer to hold the audio data
audio_buffer = io.BytesIO()

first_chunk_time = None

if response.status_code == 200:
    try:
        # Iterate over the streaming response
        for chunk in response.iter_content():
            if chunk:
                # Record the time when the first chunk arrives
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time
                    print(f"Time to First Chunk (TTFC): {first_chunk_time:.4f} seconds")
                audio_buffer.write(chunk)
    except requests.exceptions.ChunkedEncodingError:
        # This exception may occur if the server closes the connection after streaming.
        pass

    # Calculate total processing time
    end_time = time.time()
    total_processing_time = end_time - start_time

    # Retrieve the collected audio bytes
    audio_bytes = audio_buffer.getvalue()

    # Calculate the duration of the audio in seconds.
    # Assumption: 16-bit audio (2 bytes per sample), mono channel, at 22050 Hz.
    audio_duration = len(audio_bytes) / (2 * 22050)
    rtf = total_processing_time / audio_duration if audio_duration > 0 else 0

    print(f"Total Processing Time: {total_processing_time:.4f} seconds")
    print(f"Audio Duration: {audio_duration:.4f} seconds")
    print(f"Real-Time Factor (RTF): {rtf:.4f}")

    # Reset the buffer's cursor to the beginning
    audio_buffer.seek(0)

    # Load the audio from the buffer using PyDub
    audio_segment = AudioSegment.from_raw(audio_buffer, sample_width=2, frame_rate=22050, channels=1)

    # Save the audio to a file
    audio_segment.export("output_audio.wav", format="wav")
    print("Audio content has been processed and saved to output_audio.wav")
else:
    print(f"Request failed with status code {response.status_code}")