from dotenv import load_dotenv

load_dotenv()  # debe ejecutarse ANTES de importar app.generation (que importa transformers)

import logging

from fastapi import FastAPI, HTTPException

from app.generation import HF_MODEL_ID, LLMGenerationError, generate_json, load_model, unload_model
from app.models import Scene, ScriptRequest, ScriptResponse
from app.prompts import build_system_prompt, build_user_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("script-agent")

app = FastAPI(
    title="Agente de Guión IA",
    description=(
        "Primer agente del pipeline de automatización de YouTube faceless. "
        "Genera guiones estructurados en JSON usando un LLM de Hugging Face "
        "(transformers + torch, 4-bit), cargando/liberando el modelo por petición."
    ),
    version="0.3.0",
)

# NOTA: el modelo NO se carga al arrancar el contenedor. Se carga y libera en
# cada petición a /generate-script, para no ocupar VRAM cuando este agente
# no está siendo usado (necesario porque el pipeline levanta varios agentes
# a la vez y comparten la misma GPU de 8GB).


@app.get(
    "/health",
    tags=["Monitoreo"],
    summary="Estado del servicio",
    description="Verifica que el agente esté vivo y qué modelo tiene configurado (no implica que esté cargado en VRAM).",
)
async def health():
    return {"status": "ok", "model": HF_MODEL_ID}


@app.post(
    "/generate-script",
    response_model=ScriptResponse,
    tags=["Guión"],
    summary="Genera un guión completo para un video",
    description=(
        "Carga el modelo en VRAM, genera un guión estructurado (hook, escenas, CTA) "
        "y libera el modelo antes de responder. Ideal para invocarse desde n8n como "
        "primer paso del pipeline de generación de video."
    ),
)
async def generate_script(req: ScriptRequest):
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        niche=req.niche,
        topic=req.topic,
        duration_minutes=req.duration_minutes,
        tone=req.tone,
        extra_context=req.extra_context,
    )

    # max_new_tokens: aprox. 2 tokens por palabra en español + margen para el JSON
    max_new_tokens = min(req.duration_minutes * 150 * 2 + 500, 4096)

    try:
        load_model()
        logger.info(f"Generando guión | nicho='{req.niche}' topic='{req.topic}'")
        raw = generate_json(system_prompt, user_prompt, max_new_tokens=max_new_tokens)
    except LLMGenerationError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado generando el guión: {e}")
    finally:
        # Se libera SIEMPRE, haya éxito o error, antes de responder al llamador.
        unload_model()

    try:
        scenes = [Scene(**s) for s in raw["scenes"]]
        word_count = sum(len(s.narration.split()) for s in scenes) + len(raw["hook"].split())
        total_seconds = sum(s.estimated_duration_seconds for s in scenes)

        return ScriptResponse(
            title_options=raw["title_options"],
            hook=raw["hook"],
            scenes=scenes,
            cta=raw["cta"],
            word_count=word_count,
            estimated_duration_minutes=round(total_seconds / 60, 2),
            raw_model=HF_MODEL_ID,
            voice=req.voice,
            include_cta=req.include_cta,
        )
    except KeyError as e:
        raise HTTPException(status_code=502, detail=f"El JSON del modelo no tiene el campo esperado: {e}")