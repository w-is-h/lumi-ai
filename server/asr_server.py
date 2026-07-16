"""ASR server for lumi's remote backend: one FastAPI app, three switchable models.

Pick the backend with the ASR_MODEL env var (parakeet | qwen | ark) and install
the matching extra first — see README.md. The model loads once at startup.

API (matches what the lumi client expects):
    GET  /                  health check
    GET  /health            health + model info
    POST /transcribe        multipart WAV in "file" -> {"text": ...}
    POST /transcribe_batch  multipart WAVs in "files" -> {"texts": [...]}, one
                            batched forward pass where the backend supports it
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


# Each loader returns transcribe_batch(paths) -> list of texts, same order.


def load_parakeet():
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_IDS["parakeet"])

    def transcribe_batch(paths: list[str]) -> list[str]:
        return [result.text for result in model.transcribe(paths)]

    return transcribe_batch


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

    # qwen_asr batches internally; one call per file keeps us off its unverified
    # list API. Sequential, so no batch speedup on this backend.
    def transcribe_batch(paths: list[str]) -> list[str]:
        return [model.transcribe(audio=path, language=None)[0].text for path in paths]

    return transcribe_batch


def load_ark():
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    model_id = MODEL_IDS["ark"]
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Chunks of different lengths get different audio-placeholder counts, so the
    # prompts need padding to batch. It must be on the left: right padding makes
    # generate continue from pad tokens and misaligns the output slice.
    processor.tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()

    # Ban every special/added token except EOS: without this the model derails
    # mid-transcript and spams <|assistant|> until max_new_tokens (model card recipe).
    eos_ids = tokenizer.eos_token_id
    keep_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
    bad_ids = set(tokenizer.all_special_ids) - keep_ids
    bad_ids.update(
        token_id
        for token, token_id in tokenizer.get_added_vocab().items()
        if token.startswith("<") and token.endswith(">") and token_id not in keep_ids
    )
    bad_words_ids = [[token_id] for token_id in sorted(bad_ids)]

    def transcribe_batch(paths: list[str]) -> list[str]:
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "path": path},
                        {"type": "text", "text": "Please transcribe this audio."},
                    ],
                }
            ]
            for path in paths
        ]
        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            return_tensors="pt",
            sampling_rate=16000,
            audio_padding="longest",
            text_kwargs={"padding": "longest"},
            audio_max_length=30 * 16000,
        ).to(model.device)
        if "audios" in inputs:
            inputs["audios"] = inputs["audios"].to(dtype=model.dtype)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bad_words_ids=bad_words_ids,
            )
        generated = outputs[:, inputs["input_ids"].shape[1] :]
        return [t.strip() for t in tokenizer.batch_decode(generated, skip_special_tokens=True)]

    return transcribe_batch


LOADERS = {"parakeet": load_parakeet, "qwen": load_qwen, "ark": load_ark}

if BACKEND not in LOADERS:
    raise SystemExit(f"Unknown ASR_MODEL={BACKEND!r} (options: {', '.join(LOADERS)})")

logger.info(f"Loading {BACKEND} backend ({MODEL_IDS[BACKEND]})...")
_start = time.time()
transcribe_batch_fn = LOADERS[BACKEND]()
logger.info(f"Model loaded in {time.time() - _start:.1f}s")

app = FastAPI(title="Lumi ASR server", version="0.1.0")


@app.get("/")
def root():
    return {"status": "ok", "message": "Audio Transcription API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True, "model_name": MODEL_IDS[BACKEND]}


def _save_upload(file: UploadFile) -> str:
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        return tmp.name


def _transcribe_uploads(files: list[UploadFile]) -> tuple[list[str], float]:
    paths = [_save_upload(f) for f in files]
    start = time.time()
    try:
        texts = transcribe_batch_fn(paths)
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}") from e
    finally:
        for path in paths:
            os.unlink(path)
    return texts, time.time() - start


@app.post("/transcribe")
def transcribe_audio(file: UploadFile):
    texts, duration = _transcribe_uploads([file])
    logger.info(f"Transcribed {file.filename} in {duration:.2f}s: {texts[0][:80]}")
    return {
        "text": texts[0],
        "duration": duration,
        "model_name": MODEL_IDS[BACKEND],
        "file_name": file.filename,
    }


@app.post("/transcribe_batch")
def transcribe_audio_batch(files: list[UploadFile]):
    texts, duration = _transcribe_uploads(files)
    logger.info(f"Transcribed batch of {len(files)} in {duration:.2f}s")
    return {
        "texts": texts,
        "duration": duration,
        "model_name": MODEL_IDS[BACKEND],
    }
