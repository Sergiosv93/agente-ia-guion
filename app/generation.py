import gc
import json
import logging
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger("script-agent")

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"

_tokenizer = None
_model = None


def is_loaded() -> bool:
    """Verifica si el modelo de IA este cargado"""
    return _model is not None


def load_model():
    """Carga el modelo en VRAM. Se llama al inicio de cada petición, no al arrancar el contenedor."""
    global _tokenizer, _model

    if _model is not None:
        return  # ya cargado (ej. si dos requests llegan casi al mismo tiempo)

    logger.info(f"Cargando modelo '{HF_MODEL_ID}' en VRAM...")

    _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)

    quant_config = None
    if LOAD_IN_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    _model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if torch.cuda.is_available():
        logger.info(f"Modelo cargado. VRAM asignada: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def unload_model():
    """Libera el modelo de VRAM. Se llama al terminar cada petición (éxito o error)."""
    global _tokenizer, _model

    if _model is None:
        return

    logger.info("Liberando modelo de VRAM...")
    del _model
    del _tokenizer
    _model = None
    _tokenizer = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(f"VRAM liberada. VRAM asignada ahora: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


class LLMGenerationError(Exception):
    pass


def _extract_json(text: str) -> dict:
    # El modelo puede envolver el JSON en texto o markdown; extraemos el primer bloque {...}
    """El modelo puede envolver el JSON en texto o markdown; extraemos el primer bloque {...}"""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise LLMGenerationError(f"No se encontró un bloque JSON en la respuesta: {text[:300]}")
    return json.loads(match.group(0))


def generate_json(system_prompt: str, user_prompt: str, max_new_tokens: int = 3000) -> dict:
    """En esta función carga el modelo en función de los prompts y devuerlve en formato json/dict el resultado del guión"""
    if _model is None or _tokenizer is None:
        raise RuntimeError("El modelo aún no ha sido cargado (load_model() no se ejecutó).")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    last_error = None
    for attempt in range(3):
        with torch.no_grad():
            output = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7 + attempt * 0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=_tokenizer.eos_token_id,
            )
        generated = _tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        try:
            return _extract_json(generated)
        except (LLMGenerationError, json.JSONDecodeError) as e:
            last_error = e
            logger.warning(f"Intento {attempt + 1}/3 falló al parsear JSON: {e}")

    raise LLMGenerationError(f"No se pudo obtener JSON válido tras 3 intentos: {last_error}")
