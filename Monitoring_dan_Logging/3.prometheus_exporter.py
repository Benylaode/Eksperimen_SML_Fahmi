#!/usr/bin/env python3
import time
from prometheus_client import Counter, Gauge, Summary, start_http_server

PORT = 9010

REQUEST_COUNT = Counter("model_requests_total", "Total inference request")
LAST_INFERENCE_VALUE = Gauge("last_inference_prediction", "Nilai prediksi terakhir")
INFERENCE_TIME = Summary("inference_duration_seconds", "Durasi inference")

start_http_server(PORT)
print(f"📡 Prometheus exporter berjalan di port {PORT}")

@INFERENCE_TIME.time()
def dummy_inference():
    import random
    pred = random.random()
    LAST_INFERENCE_VALUE.set(pred)
    REQUEST_COUNT.inc()
    return pred

if __name__ == "__main__":
    while True:
        val = dummy_inference()
        print(f"🔥 Dummy inference: {val}")
        time.sleep(5)
