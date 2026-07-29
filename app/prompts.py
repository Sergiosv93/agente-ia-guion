WORDS_PER_MINUTE_ES = 150  # ritmo promedio de narración en español para TTS


def build_system_prompt() -> str:
    """Función que devuelve el system prompt del Agente"""
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
        '  "hook": "string (primeros 5-10 segundos, debe enganchar de inmediato)",\n'
        '  "scenes": [\n'
        "    {\n"
        '      "scene_number": int,\n'
        '      "narration": "string (texto que se leerá en voz alta)",\n'
        '      "visual_description": "string (descripción visual, en inglés, para un generador de imágenes)",\n'
        '      "estimated_duration_seconds": int\n'
        "    }\n"
        "  ],\n"
        '  "cta": "string (llamado a la acción final, suscribirse/comentar)"\n'
        "}\n"
    )


def build_user_prompt(niche: str, topic: str, duration_minutes: int, tone: str, extra_context: str | None) -> str:
    """Función prompt que recibe los parametros del usuario en tipo String para el Agente, \n
    {niche}: El nicho o tema a generar el contenido.\n
    {topic}: La premisa central del relato. Debe ser específico: incluye un nombre.\n
    o concepto clave, contexto geográfico/histórico o un gancho argumental.
    {duration_minutes}:Tiempo deseado de la historia interpretado en minutos por el agente.\n
    (No se asegura el tiempo establecido pero esta en función en una gran descripción de los parametrós de entrada).
    {tone}: La atmósfera, ritmo y emoción que debe transmitir la voz narrativa. Ayuda a definir la intensidad y las pausas dramáticas.\n
    {extra_context}: Instrucciones técnicas o de estilo adicionales. Puedes agregar estructura narrativa esperada, restricciones de vocabulario, 
    recursos sensoriales o referencias estéticas.\n
    devuelve en formato String estructurado para ser interpretado por el Agente"""


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
