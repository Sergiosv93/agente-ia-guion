# Agente de Guión IA

Primer agente del pipeline de automatización de YouTube faceless. Genera un guión
estructurado en JSON (hook, escenas con narración + descripción visual, CTA) usando
un LLM de Hugging Face cargado directamente con `transformers` + `torch` (4-bit,
vía `bitsandbytes`), sin depender de Ollama ni de ninguna capa intermedia.

## Por qué `transformers` y no `diffusers`

`diffusers` es la librería de Hugging Face para **modelos de difusión** (imagen,
video, audio) — la usaremos en los Agentes de Imágenes (paso 3) y Música (paso 5).
Para generación de **texto** (el guión), el LLM es un modelo autoregresivo, y la
librería correcta de HF es `transformers`. Ambas descargan modelos directo del Hub
de Hugging Face, así que se mantiene tu requisito de usar solo modelos de HF.

## Modelo usado

`Qwen/Qwen2.5-7B-Instruct` cargado en 4-bit (NF4 vía bitsandbytes):
- En fp16 pesaría ~14GB de VRAM (no cabe en tus 8GB).
- En 4-bit ocupa ~4.5-5GB, dejando margen cómodo en tu RTX 5050.
- Buen desempeño en español entre los modelos open-weight de ese tamaño.
- En disco (pesos originales sin cuantizar) ocupa ~15GB — por eso es importante
  definir bien dónde se guarda (ver `.env` abajo).

Se puede cambiar sin tocar código: variable `HF_MODEL_ID` en `.env` (ej.
`meta-llama/Llama-3.1-8B-Instruct`, requiere aceptar licencia en HF y usar `HF_TOKEN`).

## Configuración (`.env`)

```bash
cp .env.example .env
```

Edita `.env` y ajusta `MODELS_PATH` (y `HF_HOME`) a la carpeta de tu disco donde
quieres que se guarden los modelos descargados (ideal si tu `C:` anda justo de
espacio, ej. `D:/huggingface_models`):

```
MODELS_PATH=D:/huggingface_models
HF_HOME=D:/huggingface_models
HF_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
LOAD_IN_4BIT=true
```

- **`MODELS_PATH`**: lo usa `docker-compose.yml` para el bind mount hacia la caché
  de Hugging Face dentro del contenedor.
- **`HF_HOME`**: lo usa `transformers`/`huggingface_hub` directamente cuando corres
  el proyecto en el venv local. En Docker se sobreescribe automáticamente a la ruta
  interna del contenedor (no necesitas tocarlo ahí).

`.env` está en `.gitignore` — no se sube a git; cada quien define su propia ruta.

## Documentación interactiva (Swagger/OpenAPI)

FastAPI la genera automáticamente, sin configuración adicional:

- Swagger UI: `http://localhost:8001/docs` — puedes probar los endpoints directo desde el navegador.
- ReDoc: `http://localhost:8001/redoc`
- Esquema OpenAPI crudo: `http://localhost:8001/openapi.json`

## Ciclo de vida del modelo: carga y descarga por petición

Este agente **no carga el modelo al arrancar el contenedor**. Lo carga en VRAM
justo al recibir una petición a `/generate-script`, y lo libera (`del` +
`torch.cuda.empty_cache()`) antes de responder — tanto si la generación tuvo
éxito como si falló (bloque `finally`).

**Por qué:** el pipeline completo levanta varios agentes (guión, imágenes, audio,
etc.) que comparten la misma GPU de 8GB. Si cada uno cargara su modelo al
arrancar el contenedor, se sumarían todos en VRAM al mismo tiempo y no
alcanzaría. Con este patrón, en reposo cada agente ocupa ~0GB de VRAM, y solo
usa memoria mientras procesa su paso del pipeline.

**Costo:** cada petición tarda unos segundos/decenas de segundos extra al
inicio (cargar ~5GB del disco a VRAM) que no existían si el modelo se quedara
cargado. Para un pipeline que corre 1-2 veces al día (no en tiempo real), es
un costo aceptable a cambio de poder correr todos los agentes en la misma PC.

Verifica en los logs (`docker logs -f script-agent` o la consola del venv) los
mensajes `Cargando modelo...` / `Liberando modelo de VRAM...` con el uso de
VRAM reportado en cada paso, para confirmar que efectivamente vuelve a ~0GB
después de cada llamada.

## Opción A: correrlo en un venv local (recomendado para iterar rápido)

```powershell
python -m venv venv   # usa Python 3.13 (misma versión que la imagen de Docker)
.\venv\Scripts\Activate.ps1

pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8001
```

El `.env` se carga automáticamente al iniciar (`load_dotenv()` en `main.py`), así
que los modelos se descargarán en la ruta que definiste en `HF_HOME`.

Verifica que Python vea tu GPU antes de arrancar:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> Si `bitsandbytes` da error de CUDA no encontrado en Windows nativo, corre estos
> mismos comandos desde una terminal de **WSL2** (Ubuntu) — ahí hay paridad total
> con Linux. Docker resuelve esto automáticamente porque la imagen ya es Linux.

## Opción B: Docker

```bash
docker compose up -d --build
```

La primera vez tardará varios minutos: instala `torch` (cu128) + dependencias y
descarga el modelo desde HF Hub (~15GB) hacia la carpeta que definiste en
`MODELS_PATH` — si ya lo descargaste antes en el venv con el mismo `MODELS_PATH`/
`HF_HOME`, Docker reutiliza esos archivos y no vuelve a descargar nada.

Revisa el progreso de carga del modelo con:
```bash
docker logs -f script-agent
```

## Probar el agente

```bash
curl -X POST http://localhost:8001/generate-script \
  -H "Content-Type: application/json" \
  -d '{
    "niche": "historias de misterio y terror narrado",
    "topic": "una leyenda urbana sobre una casa abandonada en un pueblo mexicano",
    "duration_minutes": 10,
    "tone": "narrativo, envolvente, con suspenso creciente"
  }'
```

## Health check

```bash
curl http://localhost:8001/health
```

## Notas de diseño

- El endpoint es **agnóstico al nicho**: `niche` y `topic` llegan como parámetros,
  nunca hardcodeados. Cuando conectes el Sheet de Google Drive con las ideas de
  contenido, el orquestador (n8n) solo necesita mapear cada fila a este JSON de
  entrada — cero cambios en este servicio para replicarlo en otro canal/nicho.
- `visual_description` de cada escena viene en inglés (mejor rendimiento en
  modelos de imagen entrenados mayormente con prompts en inglés) — listo para
  conectarse directo al Agente de Imágenes (paso 3).
- El modelo se carga **una sola vez** al iniciar el contenedor (evento `startup`
  de FastAPI), no en cada petición — evita recargar ~5GB de pesos por request.
- Como no hay forzado nativo de JSON (a diferencia de Ollama con `format: "json"`),
  se extrae el bloque `{...}` de la respuesta y se reintenta hasta 3 veces con
  temperatura creciente si el parseo falla.

## Siguiente paso

Con este agente probado y devolviendo JSON válido, seguimos con el **Agente de
Audio** (TTS local desde HF, ej. un modelo Piper/Coqui/Bark disponible en el Hub)
que consumirá `hook` + `narration` de cada escena.
