# Voice Transcriber

Un script de Python para transcribir archivos de audio usando la API de OpenAI Whisper.

## Configuración

1. Instala las dependencias:
```bash
pip install openai python-dotenv
```

2. Crea un archivo `.env` en la raíz del proyecto con tu API key de OpenAI:
```
OPENAI_API_KEY=tu_api_key_aqui
```

## Uso

### Opción 1: Procesar todos los archivos de la carpeta input
```bash
python transcribe.py
```

### Opción 2: Procesar un archivo específico
```bash
python transcribe.py --file "ruta/al/archivo.wav"
```

## Estructura del proyecto

```
VoiceTranscriber/
├── input/          # Coloca aquí tus archivos de audio
├── output/         # Las transcripciones se guardan aquí
├── transcribe.py   # Script principal
└── .env           # Variables de entorno (crear)
```

## Formatos soportados

- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)
- FLAC (.flac)
- OGG (.ogg)
- AAC (.aac)
- WMA (.wma)

## Ejemplo de uso

1. Coloca tus archivos de audio en la carpeta `input/`
2. Ejecuta: `python transcribe.py`
3. Las transcripciones aparecerán en la carpeta `output/` con el formato `nombre_archivo_transcript.txt`

