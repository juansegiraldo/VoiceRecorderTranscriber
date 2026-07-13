# Guion de prompts para construir la feature en Lovable (MeetingMind)

> Para el Product Owner. Secuencia de prompts en orden, lista para copiar-pegar en Lovable.
> Construye la feature de transcripción de reuniones + acta + acciones abiertas, dentro de
> MeetingMind (Lovable: React + Supabase). Fecha: 2026-06-18.

---

## Cómo usar este documento (léelo antes de empezar)

1. **Lovable trabaja mejor a pasos pequeños, no con un mega-prompt.** Haz **un prompt a la vez**,
   revisa lo que generó, comprueba que funciona, y solo entonces pasa al siguiente. Si algo sale
   raro, corrige con un prompt de ajuste antes de seguir.
2. **Orden importante (no lo cambies):** primero la tabla de datos → luego el backend seguro
   (Deepgram) → luego la captura de audio → luego el acta con IA → al final el pulido. Si haces
   la captura antes que el backend, te quedas a medias.
3. **Hay cosas que NO se hacen por prompt:** poner las claves de API (Deepgram, LLM) como
   *secrets*. Eso lo haces tú a mano en la configuración de Lovable/Supabase. Lo marco con ⚙️.
4. **Decisión de enfoque (ya tomada para la PoC):** **grabar la reunión y subir el archivo al
   terminar** (no streaming en vivo). Es más simple, más robusto y suficiente para demostrar el
   valor. El streaming en vivo se puede añadir después.
5. **Pega primero el bloque "Contexto" de abajo en la sección Knowledge de Lovable.** Eso le da
   a Lovable el marco para que los prompts siguientes salgan coherentes.

---

## Paso 0 — Contexto para la sección "Knowledge" de Lovable

> Pega esto en la sección **Knowledge** del proyecto (no como prompt suelto). Es el marco que
> Lovable tendrá presente en todos los prompts.

```
PROYECTO: MeetingMind — tracker de reuniones con clientes (React + Supabase).
Cada reunión tiene: cliente, título, fecha, asistentes y "acciones abiertas" (open actions).

NUEVA FEATURE: transcripción de reuniones.
- El usuario graba el audio de una reunión (audio de la pestaña del navegador + su micrófono),
  lo transcribe, y obtiene un acta con IA que rellena automáticamente las "acciones abiertas".
- La transcripción y el acta se guardan asociadas a la reunión y al cliente.

DECISIONES TÉCNICAS:
- Transcripción: API de Deepgram, llamada SIEMPRE desde una Supabase Edge Function (la clave
  de Deepgram nunca debe estar en el frontend).
- Acta y extracción de acciones: un LLM, también desde una Edge Function.
- Enfoque: grabar y subir el archivo de audio al terminar (no streaming en tiempo real).
- Idiomas: español principalmente, a veces inglés.
- Captura de audio en el navegador con getDisplayMedia (audio de pestaña) + getUserMedia (micro).

ESTÁNDARES: TypeScript, Tailwind, shadcn/ui, componentes pequeños y reutilizables.
```

---

## Paso 1 — Modelo de datos (la tabla de transcripciones)

> Primero los datos. Sin esto, no hay dónde guardar nada.

**Prompt:**
```
En la base de datos, crea una tabla "meeting_transcriptions" para guardar las transcripciones
de reuniones, relacionada con la tabla de reuniones que ya existe. Campos:
- id
- meeting_id (relación con la reunión existente)
- audio_duration_seconds
- transcript_text (el texto completo con etiquetas de hablante)
- summary_text (el acta/resumen generado con IA, puede estar vacío al principio)
- status (uno de: 'recording', 'processing', 'completed', 'error')
- language
- created_at

Aplica las políticas de seguridad (RLS) para que cada usuario solo vea las transcripciones de
las reuniones a las que tiene acceso, igual que el resto de la app. No cambies todavía ninguna
pantalla; solo crea la tabla y sus políticas.
```

**Verifica:** que la tabla aparezca en el backend con sus campos y que las políticas RLS estén
puestas, antes de seguir.

---

## Paso 2 — Backend seguro: Edge Function que llama a Deepgram

> El proxy a Deepgram. La clave de API vive aquí, jamás en el navegador.

⚙️ **ANTES del prompt:** consigue una clave de API de Deepgram (console.deepgram.com) y, cuando
Lovable te lo pida (o en la configuración de secrets de Supabase), guárdala como secret llamado
`DEEPGRAM_API_KEY`. **No la pegues en el prompt ni en el código.**

**Prompt:**
```
Crea una Supabase Edge Function llamada "transcribe-meeting" que:
1. Reciba un archivo de audio (subido por el usuario) y un meeting_id.
2. Llame a la API de Deepgram para transcribir el audio, con estas opciones activadas:
   diarización (diarize, para separar hablantes), utterances, smart_format y puntuación.
   El idioma por defecto es español ('es'), pero permite recibir 'en' como parámetro.
3. Formatee el resultado como líneas tipo "[Hablante 0]: texto", agrupando frases seguidas
   del mismo hablante.
4. Guarde el transcript_text en la tabla meeting_transcriptions y ponga status='completed'
   (o 'error' si algo falla).
La clave de Deepgram está en el secret DEEPGRAM_API_KEY; léela del entorno, nunca la pongas en
el código ni la expongas al frontend.
```

**Verifica:** que la función se despliegue sin error. Si puedes, pruébala con un MP3 pequeño y
mira que escriba el transcript en la tabla.

> Nota: para audios muy largos puede hacer falta partir el archivo en trozos. Para la PoC, usa
> reuniones cortas (5-15 min) y no te preocupes por eso todavía. Si más adelante hace falta, hay
> lógica de "chunking" ya probada en el proyecto VoiceTranscriber para reutilizar.

---

## Paso 3 — Captura de audio en el navegador

> Ahora sí, el componente que graba. Lo más delicado; revísalo con calma.

**Prompt:**
```
Crea un componente React "MeetingRecorder" para grabar el audio de una reunión que ocurre en
otra pestaña del navegador (Google Meet o Teams web). Comportamiento:
1. Un botón "Transcribir reunión". Al pulsarlo:
   - Pide al usuario compartir la pestaña de la reunión CON su audio, usando
     navigator.mediaDevices.getDisplayMedia({ video: true, audio: true }).
   - Pide también el micrófono con navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false } }).
   - Mezcla ambas fuentes de audio en un solo stream usando la Web Audio API (un AudioContext
     con un MediaStreamDestination al que se conectan las dos fuentes).
2. Graba el stream mezclado con MediaRecorder en formato audio/webm.
3. Muestra un indicador de "grabando" con el tiempo transcurrido y un botón "Detener".
4. Al detener, produce un Blob de audio y lo deja listo para subir (todavía no lo subas, solo
   expón el Blob mediante un callback onRecordingComplete(blob)).
Maneja el caso de que el usuario cancele el selector o no marque "compartir audio": muestra un
mensaje claro explicando que debe marcar la casilla de audio.
```

**Verifica (las trampas conocidas):**
- Que al grabar **el usuario siga oyendo la reunión**. Si se queda en silencio, falta conectar
  el audio capturado de vuelta a la salida. Prompt de ajuste si pasa:
  ```
  Al capturar el audio de la pestaña, el usuario deja de oír la reunión. Conecta también la
  fuente de audio de la pestaña a audioContext.destination para que el usuario siga oyendo
  mientras grabamos.
  ```
- Que el archivo grabado **tenga sonido de la reunión** (no solo tu micro). Si no, el usuario no
  marcó "compartir audio" → el mensaje de aviso debe saltar.

---

## Paso 4 — Conectar grabación → subida → transcripción

> Unir el componente con el backend.

**Prompt:**
```
Conecta el componente MeetingRecorder con el backend. Cuando termine la grabación
(onRecordingComplete):
1. Crea una fila en meeting_transcriptions con status='processing' para la reunión actual.
2. Sube el Blob de audio a Supabase Storage (bucket privado "meeting-audio").
3. Llama a la Edge Function "transcribe-meeting" pasándole la referencia del audio y el meeting_id.
4. Mientras procesa, muestra un spinner con "Transcribiendo...".
5. Cuando la transcripción termine (status='completed'), muéstrala en pantalla con los hablantes
   separados, dentro de la ficha de la reunión.
Maneja errores: si la transcripción falla, muestra un mensaje y deja reintentar.
```

**Verifica:** flujo completo de punta a punta con una reunión corta de prueba: grabar → detener
→ ver la transcripción guardada en la ficha.

---

## Paso 5 — Acta con IA + extracción de acciones abiertas (el diferenciador)

> Aquí está el valor que vende el pitch: rellenar las "acciones abiertas" solas.

⚙️ **ANTES del prompt:** guarda la clave del LLM como secret. Si usas Claude, `ANTHROPIC_API_KEY`;
si usas OpenAI, `OPENAI_API_KEY`. (Recomendado: Claude por calidad en español; cualquiera sirve.)

**Prompt:**
```
Crea una Supabase Edge Function "generate-meeting-minutes" que reciba un meeting_id, lea el
transcript_text de meeting_transcriptions, y llame a un LLM (usa la clave del secret
correspondiente) para generar, en español:
1. Un acta estructurada: contexto, temas tratados, decisiones, y próximos pasos.
2. Una lista de "acciones abiertas": cada una con descripción, responsable (si se menciona) y
   fecha límite (si se menciona).
Guarda el acta en summary_text. Para las acciones, créalas en la estructura de "acciones
abiertas/open actions" que la reunión ya usa, asociadas a la reunión.
Devuelve el acta y las acciones al frontend.

Luego añade un botón "Generar acta" en la ficha de la reunión que llame a esta función y muestre
el acta y las acciones extraídas, permitiendo al usuario editarlas antes de confirmar.
```

**Verifica:** que el acta salga coherente y que las acciones extraídas aparezcan en la columna de
"acciones abiertas" de la reunión. **Este es el momento "wow" de la demo.**

---

## Paso 6 — Pulido para la demo

> Pequeños arreglos que hacen que la demo se vea profesional.

**Prompts (uno a uno, según haga falta):**
```
Añade un aviso la primera vez que el usuario usa "Transcribir reunión", explicando en 2-3 frases
que debe elegir la pestaña de la reunión y marcar la casilla "compartir audio".
```
```
En la lista de reuniones, muestra un pequeño icono cuando una reunión ya tiene transcripción,
para distinguirlas de un vistazo.
```
```
Añade un botón para copiar la transcripción y otro para descargarla como archivo .txt.
```

---

## Resumen de la secuencia

| Paso | Qué hace | ⚙️ Secret a poner antes |
|---|---|---|
| 0 | Contexto en Knowledge | — |
| 1 | Tabla de transcripciones | — |
| 2 | Edge Function → Deepgram | `DEEPGRAM_API_KEY` |
| 3 | Componente de captura de audio | — |
| 4 | Grabar → subir → transcribir | — |
| 5 | Acta IA + acciones abiertas | `ANTHROPIC_API_KEY` u `OPENAI_API_KEY` |
| 6 | Pulido para la demo | — |

---

## Consejos para que Lovable no se rompa

- **Un prompt, revisar, siguiente.** No encadenes 5 cosas en un prompt; Lovable las hace a
  medias.
- **Si algo se rompe, descríbelo en lenguaje natural** ("el botón X no hace Y"), no pegues
  código. Lovable es prompt-driven.
- **Las claves de API SIEMPRE como secrets**, nunca en un prompt ni en el código. Si Lovable
  alguna vez sugiere poner una clave en el frontend, recházalo: es el error de seguridad #1.
- **Para audios largos** (>15-20 min) puede hacer falta partir el archivo; déjalo fuera de la
  PoC y, si se necesita, reutiliza la lógica de chunking del proyecto VoiceTranscriber.
- **Prueba con reuniones reales con clientes** (español) lo antes posible: la calidad real es lo
  que el PO querrá ver.

---

### Relación con los otros documentos
- Pitch para el PO: [pitch-PO-transcripcion.md](pitch-PO-transcripcion.md)
- PRD técnico-de-decisión: [PRD-feature-transcripcion-en-lovable.md](PRD-feature-transcripcion-en-lovable.md)
