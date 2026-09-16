"""Lightweight BGE inference for the public API on Render's 512 MB Free plan."""

import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MEDBUDDY_BGE_DIR", os.path.join(BASE_DIR, "bge_model"))


class OnnxBge:
    def __init__(self):
        self.tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_mem_pattern = False
        options.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(
            os.path.join(MODEL_DIR, "onnx", "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {item.name for item in self.session.get_inputs()}

    def encode(self, texts, normalize_embeddings=True):
        vectors = []
        for text in texts:
            encoded = self.tokenizer.encode(text)
            inputs = {
                "input_ids": np.asarray([encoded.ids], dtype=np.int64),
                "attention_mask": np.asarray([encoded.attention_mask], dtype=np.int64),
                "token_type_ids": np.asarray([encoded.type_ids], dtype=np.int64),
            }
            outputs = self.session.run(None, {
                name: value for name, value in inputs.items() if name in self.input_names
            })
            vector = outputs[0][0, 0].astype(np.float32)
            if normalize_embeddings:
                vector /= max(float(np.linalg.norm(vector)), 1e-12)
            vectors.append(vector)
        return np.stack(vectors)


def load_bge_model():
    return OnnxBge()
