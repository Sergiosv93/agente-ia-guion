WORDS_PER_MINUTE_ES = 150  # ritmo promedio de narración en español para TTS


def build_system_prompt() -> str:
    return (
        "Eres un guionista experto en videos narrativos para YouTube en español, "
        "especializado en contenido 'faceless' (sin presentador en cámara). "
        "Tu guión será leído por una voz sintética (TTS) y acompañado de imágenes generadas por IA, "
        "por eso cada escena debe incluir una narración clara y una descripción visual "
        "concreta y generable por un modelo de imágenes (sin texto en pantalla, sin marcas registradas). "
        "SIEMPRE respondes ÚNICAMENTE con un JSON válido, sin texto adicional, sin markdown, "
        "que cumpla EXACTAMENTE este esquema:\n"
        "{\n"
        '  "title_options": ["string", "string", "string"],\n'
        '  "hook": {\n'
        '    "text": "string (primeros 5-10 segundos, debe enganchar de inmediato)",\n'
        '    "visual_description": "string (descripción visual en inglés, para un generador de imágenes)"\n'
        "  },\n"
        '  "scenes": [\n'
        "    {\n"
        '      "scene_number": int,\n'
        '      "narration": "string (texto que se leerá en voz alta)",\n'
        '      "visual_description": "string (descripción visual, en inglés, para un generador de imágenes)",\n'
        '      "estimated_duration_seconds": int\n'
        "    }\n"
        "  ],\n"
        '  "cta": {\n'
        '    "text": "string (llamado a la acción final, suscribirse/comentar)",\n'
        '    "visual_description": "string (descripción visual en inglés, coherente con el cierre del video)"\n'
        "  }\n"
        "}\n"
    )


def build_user_prompt(niche: str, topic: str, duration_minutes: int, tone: str, extra_context: str | None) -> str:
    target_words = duration_minutes * WORDS_PER_MINUTE_ES
    extra = f"\nContexto adicional: {extra_context}" if extra_context else ""
    return (
        f"Nicho del canal: {niche}\n"
        f"Idea del video: {topic}\n"
        f"Duración objetivo: {duration_minutes} minutos (~{target_words} palabras de narración total)\n"
        f"Tono: {tone}\n"
        f"Divide el guión en escenas de 20-40 segundos cada una (para facilitar la generación de imágenes por escena).\n"
        f"El guión debe estar completo en español neutro, listo para pasar a un motor TTS.{extra}"
    )
