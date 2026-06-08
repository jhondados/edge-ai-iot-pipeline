"""Optimize models for edge deployment."""
import tensorflow as tf
import numpy as np
from pathlib import Path

class EdgeModelOptimizer:
    def convert_to_tflite(self, saved_model_path: str, output_path: str,
                          quantize: bool = True) -> dict:
        """Convert TF model to TFLite with optional INT8 quantization."""
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            # Representative dataset for calibration
            def representative_data():
                for _ in range(100):
                    yield [np.random.randn(1, 224, 224, 3).astype(np.float32)]
            converter.representative_dataset = representative_data
            converter.inference_input_type  = tf.uint8
            converter.inference_output_type = tf.uint8
        tflite_model = converter.convert()
        Path(output_path).write_bytes(tflite_model)
        original_mb = Path(saved_model_path).stat().st_size / 1e6 if Path(saved_model_path).is_file() else 0
        compressed_mb = len(tflite_model) / 1e6
        return {"output": output_path, "size_mb": round(compressed_mb, 2),
                "compression_ratio": round(original_mb / max(compressed_mb, 0.001), 1)}

    def benchmark_on_device(self, tflite_path: str, n_runs: int = 100) -> dict:
        """Benchmark TFLite model on current device."""
        import time
        interp = tf.lite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        input_details = interp.get_input_details()
        dummy_input = np.zeros(input_details[0]["shape"], dtype=input_details[0]["dtype"])
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            interp.set_tensor(input_details[0]["index"], dummy_input)
            interp.invoke()
            latencies.append((time.perf_counter() - t0) * 1000)
        return {"mean_ms": round(np.mean(latencies), 2), "p99_ms": round(np.percentile(latencies, 99), 2),
                "min_ms": round(min(latencies), 2)}
