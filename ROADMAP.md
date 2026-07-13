# Roadmap — de VoiceTranscriber a "Granola propio"

> Documento de visión. No es un compromiso de fechas. Escrito 2026-06-18.
> Meta declarada: llegar a una herramienta tipo **Granola.ai** que capture reuniones
> (incluyendo **Zoom/Teams de escritorio**) y genere actas, no solo transcripciones.

---

## 1. Dónde estamos hoy

`VoiceTranscriber` es una app Streamlit que:
- Recibe un archivo (MP3/WAV/M4A/MP4) **subido manualmente, después de la reunión**.
- Lo transcribe con Deepgram (con diarización) u OpenAI Whisper.
- Devuelve **texto crudo** + descarga `.txt`.

**Activo más valioso:** la lógica de diarización con remapeo de speakers entre chunks
(`transcribe_large_file_with_diarization` en `AppTranscribe.py`). Es el problema difícil
que Granola **todavía no resuelve en vivo**. No lo tires: es reutilizable en cada fase.

**Brecha de paradigma con Granola:**

| | VoiceTranscriber hoy | Granola |
|---|---|---|
| Entrada | Archivo subido *después* | Audio de la reunión *en vivo* |
| Captura | Manual (grabas con otra app) | Audio del sistema (micro + altavoz), **sin bot** |
| Salida | Transcript crudo | **Acta estructurada** = tus notas + transcript, vía LLM |
| Valor central | El texto | La decisión/acción accionable |

El "momento Granola" **no es la transcripción** (eso es commodity; Granola la subcontrata
a Deepgram/AssemblyAI igual que tú). Es: capturar las dos voces sin bot, en vivo, y
convertirlo en acta usando tus propias notas como guía.

---

## 2. La verdad técnica sobre "plugin de Chrome"

Decidiste que quieres capturar **también Zoom/Teams de escritorio**. Hay que ser claro:

- **Una extensión de Chrome NO puede capturar el audio del sistema.** Solo puede capturar:
  - el audio de **una pestaña** (`chrome.tabCapture`), o
  - el **micrófono** (`getUserMedia`).
- El modelo sin-bot de Granola para apps de escritorio (Zoom/Teams nativos) **exige una
  app de escritorio** con permisos de audio del SO (loopback/WASAPI en Windows,
  ScreenCaptureKit/Core Audio en macOS).

**Conclusión:** el plugin de Chrome puede ser una **fase intermedia válida** (cubre Google
Meet en navegador y es el camino más barato para validar el producto en vivo), pero **el
destino para Zoom/Teams es una app de escritorio** (Electron o Tauri). El plugin no es la
meta final; es un peldaño.

---

## 3. Fases

### Fase 0 — Limpieza de base (días)
Prepara el terreno sin cambiar funcionalidad.
- [ ] Extraer la lógica de transcripción/diarización a un módulo reutilizable
      (`core/transcription.py`) — hoy está **duplicada** entre `AppTranscribe.py` y
      `scripts/deepgram_transcribe_cli.py`. Esto es prerequisito para reusarla en una
      extensión o app de escritorio sin copiar-pegar otra vez.
- [ ] Tests mínimos sobre `map_speakers_between_chunks` y `format_diarized_output`
      (no hay tests hoy; estas funciones son el activo a proteger).

### Fase 1 — Acta con IA + plantillas (1-2 semanas) ⭐ mayor valor/esfuerzo
Mueve el producto de "transcriptor" a "tomador de notas" **sin tocar la captura**.
- [ ] Botón "Generar acta" que envía el transcript a Claude/GPT.
- [ ] Plantillas por tipo de reunión (ventas, 1:1, entrevista, daily). Granola vende esto.
- [ ] Caja de "mis notas" donde el usuario escribe apuntes; el LLM **fusiona notas +
      transcript** (este es el diferenciador real de Granola, no la transcripción).
- [ ] Internaliza el GPT externo que el README ya enlaza (`GeneradorActasReunion`).
- Modelo recomendado: Claude (Opus/Sonnet 4.x). Ver skill `claude-api` para ids/precios.
- **Por qué primero:** ataca la brecha de *paradigma*, que es lo que hace especial a
  Granola. Cero infraestructura nueva. Es el mejor ROI disponible.

### Fase 2 — Captura en vivo en navegador (extensión Chrome MV3) (semanas)
"Granola para reuniones-en-navegador" (Google Meet).
- [ ] Extensión Manifest V3.
- [ ] Mezcla **dos streams**: `chrome.tabCapture` (las otras voces de la pestaña) +
      `getUserMedia` (tu micrófono). Sin mezclar ambos solo capturas un lado.
- [ ] Streaming a Deepgram en vivo (WebSocket de streaming, no el endpoint batch actual)
      y panel lateral con transcripción en tiempo real + caja de notas.
- [ ] Al cerrar: transcript + notas → LLM → acta (reusa Fase 1).
- **Límite honesto:** solo cubre reuniones dentro de la pestaña. No Zoom/Teams nativos.
- **Necesita un backend:** la clave de Deepgram/LLM **no puede vivir en la extensión**
  (sería pública). Hace falta un pequeño servicio (FastAPI/serverless) que haga de proxy
  y guarde las actas.

### Fase 3 — App de escritorio (el verdadero Granola) (meses)
El destino para **Zoom/Teams de escritorio**, que es lo que pediste.
- [ ] Electron o Tauri (Tauri = binario más liviano, Rust; Electron = más ecosistema/JS).
- [ ] Captura de audio del sistema:
  - **Windows:** loopback WASAPI (capturar lo que sale por los altavoces) + micro.
  - **macOS:** ScreenCaptureKit / Core Audio (requiere permisos y firma de app).
- [ ] Sin bot: nadie se entera de que grabas (diferenciador #1 de Granola).
- [ ] Reusa: streaming Deepgram (Fase 2) + generación de acta (Fase 1).
- [ ] Descarte de audio tras transcribir (postura de privacidad de Granola).
- **Magnitud real:** permisos por SO, posiblemente un driver de audio virtual en Windows,
  distribución firmada (Apple notarization / firma de código Windows). Es un proyecto
  aparte, no una evolución incremental de la app Streamlit.

### Fase 4 — Integraciones y distribución (continuo)
Lo que convierte una herramienta en producto.
- [ ] Exportar a Notion / Slack / Google Docs (Granola vive de esto).
- [ ] Integración con calendario (auto-vincular acta a evento).
- [ ] Servidor MCP para consultar tus actas desde Claude/ChatGPT (Granola lo lanzó en 2026).
- [ ] Almacenamiento de actas + búsqueda.

---

## 4. Decisiones técnicas que hay que tomar pronto

1. **Backend obligatorio desde Fase 2.** Las claves de API no pueden ir en cliente
   (extensión o app de escritorio publicada). Define ya un servicio mínimo (FastAPI o
   funciones serverless) como proxy + almacén de actas. Diséñalo en Fase 1 aunque no lo
   uses todavía.
2. **Deepgram streaming vs batch.** Hoy usas el endpoint batch (subes el archivo). En vivo
   necesitas el **WebSocket de streaming** de Deepgram. La diarización en streaming se
   comporta distinto al batch — habrá que reescribir el remapeo de speakers para flujo
   continuo, no para chunks.
3. **Privacidad como foso.** Granola compite en privacidad (SOC 2, no almacena audio, no
   entrena con tus datos). Si vas a vender esto, define la postura de privacidad temprano;
   reconstruirla después es caro.
4. **Electron vs Tauri** para Fase 3 — decisión a tomar al llegar, no ahora.

---

## 5. Resumen ejecutivo

- El plugin de Chrome **no es la meta**: para Zoom/Teams de escritorio el destino es una
  **app de escritorio**. El plugin es un peldaño útil para Meet.
- El siguiente paso de mayor valor **no es captura, es el acta con IA (Fase 1)** — porque
  ataca la diferencia de paradigma que hace especial a Granola.
- Tu lógica de diarización es un activo; protégela (Fase 0) y reúsala en cada fase.
- Cada fase en vivo (2 y 3) **requiere un backend** para las claves de API. No lo dejes
  para el final.

**Orden recomendado:** Fase 0 → Fase 1 → Fase 2 → Fase 3, con Fase 4 en paralelo desde
que haya actas que valga la pena exportar.
