import io
import csv
import time
import logging
import requests
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("torchserve")

BASE_URL = "http://localhost:8000/predictions/xttsv2"

def stream_synthesize_endpoint(text: str) -> dict:
    payload = {"data": text}
    headers = {'Content-Type': 'application/json'}
    try:
        start_time = time.perf_counter()
        response = requests.post(BASE_URL, headers=headers, json=payload, stream=True)
        try:
            audio_buffer = io.BytesIO()
            ttfc = None  # Time to first chunk

            if response.status_code == 200:
                try:
                    # Use a larger chunk size (1024 bytes)
                    for chunk in response.iter_content():
                        if chunk:
                            if ttfc is None:
                                ttfc = time.perf_counter() - start_time
                            audio_buffer.write(chunk)
                except requests.exceptions.ChunkedEncodingError as e:
                    pass 
                
                total_processing_time = time.perf_counter() - start_time
                audio_bytes = audio_buffer.getvalue()

                if not audio_bytes:
                    logger.error("No audio data received.")
                    return None

                # Calculate the duration of the generated audio (assuming 16-bit PCM at 22050 Hz)
                audio_duration = len(audio_bytes) / (2 * 22050)
                rtf = total_processing_time / audio_duration if audio_duration > 0 else float('inf')
                return {"ttfc": ttfc, "rtf": rtf, "audio_data": audio_bytes}
            else:
                logger.error(f"Failed to stream. Status code: {response.status_code}")
                logger.error(f"Error detail: {response.text}")
                return None
        finally:
            response.close()
    except Exception as e:
        logger.exception(f"An error occurred during streaming synthesis: {e}")
        return None

def benchmark_concurrency_http(text: str, max_concurrency: int = 20) -> None:
    """
    Benchmark the streaming endpoint by varying the number of concurrent requests.
    For each concurrency level, compute the average TTFC (Time To First Chunk) and RTF, then plot the results.
    """
    concurrency_levels = list(range(1, max_concurrency + 1))
    ttfc_results = []
    rtf_results = []

    for concurrency in concurrency_levels:
        results = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(stream_synthesize_endpoint, text) for _ in range(concurrency)]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        completed_requests = len(results)
        if completed_requests:
            avg_ttfc = sum(result["ttfc"] for result in results) / completed_requests
            avg_rtf = sum(result["rtf"] for result in results) / completed_requests
        else:
            avg_ttfc = float('nan')
            avg_rtf = float('nan')

        ttfc_results.append(avg_ttfc)
        rtf_results.append(avg_rtf)
        logger.info(f"Concurrency Level: {concurrency}, Average TTFC: {avg_ttfc:.4f} s, Average RTF: {avg_rtf:.4f}")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(concurrency_levels, ttfc_results, marker='o')
    plt.title('Average TTFC vs. Concurrency Level')
    plt.xlabel('Concurrency Level')
    plt.ylabel('Average TTFC (seconds)')

    plt.subplot(1, 2, 2)
    plt.plot(concurrency_levels, rtf_results, marker='o', color='orange')
    plt.title('Average RTF vs. Concurrency Level')
    plt.xlabel('Concurrency Level')
    plt.ylabel('Average RTF')

    plt.tight_layout()
    plt.savefig('benchmark_results_http.png')
    plt.show()

    # Save performance data to CSV
    with open('torchserve_perf.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Concurrency Level', 'Average TTFC', 'Average RTF'])
        for level, ttfc, rtf in zip(concurrency_levels, ttfc_results, rtf_results):
            writer.writerow([level, ttfc, rtf])

def main() -> None:
    # input_text = (
    #     "This is a stress-test of the streaming service by delivering an extensive, multifaceted, "
    #     "and intricately constructed series of words that not only describe the purpose and functionality "
    #     "of the XTTS endpoint in detail but also serve to evaluate its performance under extreme conditions "
    #     "where verbosity and syntactical complexity challenge the system's ability to process and stream "
    #     "large quantities of text efficiently. In order to thoroughly measure performance while ensuring "
    #     "reliability and quality, every aspect of network communication and computational execution is "
    #     "monitored and recorded."
    # )
    input_text = "Hello, I am Akash Sonowal"
    benchmark_concurrency_http(input_text, max_concurrency=20)

if __name__ == "__main__":
    main()