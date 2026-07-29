from typing import Optional
from pydantic import BaseModel, Field


class VoiceConfig(BaseModel):
    """Mismo esquema que VoiceConfig del Agente de Audio -- aquí solo se reenvía,
    el Agente de Guión no la usa para nada al generar el texto."""

    gender: str = Field(default="female")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    pitch_semitones: float = Field(default=0.0, ge=-6.0, le=6.0)
    language: str = Field(default="es")


class ScriptRequest(BaseModel):
    niche: str = Field(..., description="Nicho/temática del canal, ej. 'historias de terror'")
    topic: str = Field(..., description="Idea concreta del video, tomada del Sheet")
    duration_minutes: int = Field(default=10, ge=1, le=30)
    tone: str = Field(default="narrativo, envolvente, con suspenso")
    language: str = Field(default="es")
    extra_context: Optional[str] = Field(
        default=None, description="Contexto extra opcional (referencias, restricciones, etc.)"
    )
    voice: Optional[VoiceConfig] = Field(
        default=None,
        description="Configuración de voz del canal -- NO se usa para generar el guión, "
        "solo se reenvía tal cual en la respuesta para que el Agente de Audio la reciba "
        "directo, sin que el orquestador tenga que insertar un paso intermedio.",
    )
    include_cta: bool = Field(
        default=True, description="Se reenvía en la respuesta; el Agente de Audio decide si narra el CTA."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "niche": "historias de misterio y terror narrado",
                "topic": "una leyenda urbana sobre una casa abandonada en un pueblo mexicano",
                "duration_minutes": 10,
                "tone": "narrativo, envolvente, con suspenso creciente",
                "language": "es",
                "voice": {"gender": "female", "speed": 1.0, "pitch_semitones": 0, "language": "es"},
                "include_cta": True,
                "extra_context":"Estilo de narración tipo 'relatos de la noche' o podcast de misterio.",
            }
        }
    }


class Scene(BaseModel):
    scene_number: int
    narration: str
    visual_description: str
    estimated_duration_seconds: int


class ScriptResponse(BaseModel):
    title_options: list[str]
    hook: str
    scenes: list[Scene]
    cta: str
    word_count: int
    estimated_duration_minutes: float
    raw_model: str
    voice: Optional[VoiceConfig] = None
    include_cta: bool = True