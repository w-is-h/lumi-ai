"""ASR server for lumi's remote backend: one FastAPI app, three switchable models.

Pick the backend with the ASR_MODEL env var (parakeet | qwen | ark) and install
the matching extra first — see README.md. The model loads once at startup.

API (matches what the lumi client expects):
    GET  /            health check
    GET  /health      health + model info
    POST /transcribe  multipart WAV in "file" -> {"text": ...}
"""

import logging
import os
import tempfile
import time

from fastapi import FastAPI, HTTPException, UploadFile

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BACKEND = os.environ.get("ASR_MODEL", "parakeet")

MODEL_IDS = {
    "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
    "qwen": "Qwen/Qwen3-ASR-1.7B",
    "ark": "AutoArk-AI/ARK-ASR-0.6B",
}


def load_parakeet():
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_IDS["parakeet"])

    def transcribe(path: str) -> str:
        return model.transcribe([path])[0].text

    return transcribe


def load_qwen():
    import torch
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        MODEL_IDS["qwen"],
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=1024,
    )

    def transcribe(path: str) -> str:
        results = model.transcribe(audio=path, language=None)
        return results[0].text

    return transcribe


def load_ark():
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    model_id = MODEL_IDS["ark"]
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).to("cuda")

    def transcribe(path: str) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": path},
                    {"type": "text", "text": "Please transcribe this audio."},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            sampling_rate=16000,
            audio_padding="longest",
        ).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=1024)
        generated = outputs[:, inputs["input_ids"].shape[1] :]
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

    return transcribe


LOADERS = {"parakeet": load_parakeet, "qwen": load_qwen, "ark": load_ark}

if BACKEND not in LOADERS:
    raise SystemExit(f"Unknown ASR_MODEL={BACKEND!r} (options: {', '.join(LOADERS)})")

logger.info(f"Loading {BACKEND} backend ({MODEL_IDS[BACKEND]})...")
_start = time.time()
transcribe_fn = LOADERS[BACKEND]()
logger.info(f"Model loaded in {time.time() - _start:.1f}s")

app = FastAPI(title="Lumi ASR server", version="0.1.0")


@app.get("/")
def root():
    return {"status": "ok", "message": "Audio Transcription API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True, "model_name": MODEL_IDS[BACKEND]}


@app.post("/transcribe")
def transcribe_audio(file: UploadFile):
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    start = time.time()
    try:
        text = transcribe_fn(tmp_path)
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}") from e
    finally:
        os.unlink(tmp_path)

    duration = time.time() - start
    logger.info(f"Transcribed {file.filename} in {duration:.2f}s: {text[:80]}")
    return {
        "text": text,
        "duration": duration,
        "model_name": MODEL_IDS[BACKEND],
        "file_name": file.filename,
    }
