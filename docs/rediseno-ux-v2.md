# Rediseño UX/UI v2 — "menos ruido, más señal"

**Estado: implementado** en [AppTranscribe.py](../AppTranscribe.py) + [.streamlit/config.toml](../.streamlit/config.toml).
Este documento registra la auditoría, los principios y las decisiones para que futuras
iteraciones no las deshagan por accidente.

## Contexto

La v1 nació como un subidor-transcriptor simple y fue acumulando capacidades
(diarización v2, métricas de conversación, sentimiento, Google Drive, recorte) sin
re-pensar la presentación. El resultado: mucha opción visible, mucho emoji y una
jerarquía que ya no contaba la historia de la app. La v2 no añade funciones —
**reordena y calla** para que las capacidades nuevas se entiendan como un producto.

## Auditoría de la v1 (qué estaba mal y por qué)

1. **Arquitectura de información invertida.** Los ajustes (modelo, idioma,
   diarización, métricas, sentimiento) se renderizaban **debajo** de los
   resultados, al fondo de la página. El usuario configuraba *después* de ver el
   botón que usa esa configuración. Causa histórica: los widgets se fueron
   añadiendo al final del script.
2. **Ruido de emojis (~25 distintos).** 🎤📁☁️🔄✂️🎵📊😊ℹ️⚠️❌✅🎉📄🌐🇪🇸🇬🇧🔑…
   en títulos, pestañas, checkboxes, estados y errores. El emoji dejó de significar
   nada porque estaba en todas partes.
3. **Copy bilingüe inconsistente.** "Insert audio file" / "Subir archivo" /
   "Audio Trimming (Optional)" / "Métricas de conversación" conviven en la misma
   pantalla. El informe descargable y el feedback ya eran 100% español; la UI no.
4. **Sin sistema visual.** ~15 bloques de HTML inline con tamaños y grises
   arbitrarios (#444, #666, #888), todo centrado (las frases largas centradas se
   leen peor), CSS muerto (.main-header, .success-box… definidos y nunca usados),
   dos bloques `<style>` separados y un fondo forzado con `!important` que rompía
   el tema oscuro.
5. **Opciones sin priorizar.** Recorte siempre expandido (slider + 2 inputs +
   caption) aunque el 90% de las transcripciones son del archivo completo; cuatro
   controles de configuración al mismo nivel visual que el flujo principal.
6. **Resultados apilados sin resumen.** Transcripción + 3 expanders + captions +
   2 botones de descarga con estilos y idiomas distintos ("Download" /
   "📄 Descargar informe…").
7. **Dos botones primarios** simultáneos (Cargar de Drive y Start Transcription)
   compitiendo por atención.

## Principios v2

- **Un flujo lineal, tres pasos**: Audio → Ajustes → Resultado. La página se lee
  de arriba a abajo en el orden en que se usa.
- **Divulgación progresiva**: visible solo lo que el 80% toca (fuente, idioma,
  botón). Todo lo demás (recorte, modelo, toggles) vive en expanders cerrados.
- **Emoji como dato, nunca como decoración.** Se eliminan todos los emojis de
  chrome (títulos, pestañas, botones, estados). Se conservan los que *codifican
  significado* y vienen de `core/`: los iconos de rating ✅🟡⚠️ (bueno/ok/atención)
  y la línea temporal de sentimiento 😊😐🙁 (es una visualización). Coincide con la
  regla del informe CLI: mismo lenguaje en app y reporte.
- **Español como idioma único de la interfaz.** El producto, el feedback y la
  documentación ya eran español; la UI deja de mezclar.
- **Un solo botón primario por pantalla** (Transcribir). Drive y descargas son
  secundarios.
- **Resumen antes que detalle** (es una herramienta, se escanea): chips de
  metadatos y feedback evaluativo primero; números por hablante después; guía de
  interpretación al final.
- **Identidad por texto, nunca solo por color** (ya era así en v1 — se mantiene:
  "Hablante 0 — 45%" + medidor, no leyendas de color).

## Sistema visual

Streamlit manda: el tema se define donde el framework quiere (config.toml), el CSS
propio se reduce a un único bloque y usa las variables del tema con fallback, para
no romper el modo oscuro si el usuario lo elige en el menú.

**Tokens** ([.streamlit/config.toml](../.streamlit/config.toml)):

| Token | Valor | Uso |
|---|---|---|
| `primaryColor` | `#0F766E` (verde petróleo) | Botón primario, slider, progreso, tabs, toggles |
| `textColor` | `#1F2A2E` (tinta con sesgo frío) | Texto principal |
| `backgroundColor` | `#FFFFFF` | Fondo |
| `secondaryBackgroundColor` | `#F2F6F5` (neutro con sesgo al acento) | Inputs, expanders, code |

Racional del acento: mundo audio sin clichés — el rojo REC connota error, el
violeta "IA" es el default genérico. Un verde petróleo profundo es calmado,
distinguible del verde-éxito de los ratings, y da 5.3:1 de contraste sobre blanco
(AA para texto normal, holgado para UI). Los grises no son puros: llevan el sesgo
del acento (neutro elegido, no heredado).

**Tipografía**: la fuente por defecto de Streamlit (Source Sans), deliberadamente —
una webfont para una utility móvil es coste sin retorno. La jerarquía se construye
con escala y peso, no con familias: wordmark 1.5rem/700 con tracking −0.02em,
eyebrows de sección 0.78rem/600 en mayúsculas con tracking +0.08em y color
apagado, cuerpo por defecto, captions del tema. Texto alineado a la izquierda;
centrado solo el hero y el pie.

**Layout**: una columna de máx. 680px, sin tarjetas ni cajas — la estructura la
dan los eyebrows y el espacio en blanco. Targets táctiles ≥44px y fuente de
inputs ≥16px (evita el auto-zoom de iOS) se conservan de la v1.

## Arquitectura de información v2

```
┌──────────────────────────────────────┐
│  ⌁ (marca SVG)  Voice Transcriber    │  hero compacto, centrado
│  Hablantes, métricas y sentimiento   │  tagline = qué hace la v2
├──────────────────────────────────────┤
│  AUDIO                               │  eyebrow
│  [ Archivo local | Google Drive ]    │  tabs sin emoji
│  (uploader / picker de Drive)        │
│  nombre.mp3 · 12,4 MB · 07:32        │  resumen del archivo cargado
│  ▸ Recortar audio (opcional)         │  expander CERRADO: slider + inicio/fin
├──────────────────────────────────────┤
│  AJUSTES                             │  eyebrow
│  [ Auto | Español | English ]        │  segmented control (idioma)
│  ▸ Opciones avanzadas                │  expander CERRADO: modelo, diarización,
│                                      │  métricas, sentimiento (toggles)
├──────────────────────────────────────┤
│  [        Transcribir        ]       │  ÚNICO botón primario, full-width
│  (st.status: progreso plegable)      │
├──────────────────────────────────────┤
│  RESULTADO                           │  solo si existe transcripción
│  idioma · modelo · duración          │  caption de metadatos
│  [ Transcripción | Métricas | Sent. ]│  tabs (Sent. solo si hay datos)
│  descargas: .txt | informe completo  │
├──────────────────────────────────────┤
│  pie: formatos · GPT de actas · créditos  (caption única)
└──────────────────────────────────────┘
```

Detalle de las pestañas de resultado:

- **Transcripción**: `st.code` (conserva el botón copiar) dentro de un contenedor
  con altura fija y scroll propio — el resultado ya no empuja la página metros
  hacia abajo.
- **Métricas**: feedback evaluativo primero (los bullets en español de `core`),
  luego un bloque por hablante (nombre — % de habla · tiempo · palabras · ppm,
  medidor `st.progress`, ratings como caption), línea global (hablantes, palabras,
  interrupciones, silencio, preguntas) y el expander "Cómo leer estas métricas".
- **Sentimiento**: tono general con score y descripción, línea temporal
  inicio→fin, desglose por hablante, escala como caption.

## Decisiones de implementación (y por qué)

- **`st.segmented_control` para el idioma** (Auto/Español/English, sin banderas):
  control compacto y táctil; exige subir el suelo a `streamlit>=1.40` (Cloud ya
  instala la última). Puede devolver `None` al deseleccionar → se trata como Auto.
  El mapeo por substring ('auto'/'espa') sigue funcionando con las etiquetas nuevas.
- **`st.status` para el progreso** en lugar de progress bar + texto sueltos: un
  contenedor con estados (running/complete/error) y log plegable. Los mensajes del
  `progress_cb` de `core` llegan con emoji inicial ("🔍 Analyzing…"); la app los
  limpia en la capa de presentación (regex al inicio del mensaje) — **`core` no se
  toca**: sus strings son compartidos con el CLI y de presentación se encarga cada
  cliente.
- **Los strings de modelo siguen siendo exactamente** `"Deepgram"` / `"OpenAI
  Whisper"` (contrato documentado en CLAUDE.md) — solo cambian de posición.
- **Se preserva intacta la lógica sensible**: caché de conversión
  (`converted_mp3_*`), sincronización slider↔inputs del recorte (mismas keys y
  callbacks, ahora dentro del expander), duck-typing de `DriveFile`, flujo OAuth
  PKCE, `get_secret`. El diff es de presentación y orden, no de comportamiento.
- **Errores en español y accionables** ("Falta la clave DEEPGRAM_API_KEY. Añádela
  en .env o en los secrets…"), sin ❌ (st.error ya es rojo).
- Se eliminan los `time.sleep(0.5)` cosméticos previos a la conversión y el CSS
  muerto de la v1.

## Detectado pero fuera de alcance (deliberadamente)

- La extracción MP4 usa `ffmpeg` de PATH vía shell en vez del binario de
  `imageio-ffmpeg` ya resuelto (funciona en Cloud gracias a packages.txt; en local
  sin ffmpeg en PATH fallaría). Es un fix funcional, no de diseño — mejor en un
  cambio propio.
- Nada de captura en vivo ni actas: eso es Fase 1+ del [ROADMAP](../ROADMAP.md).
  Este rediseño deja el "hueco" natural (pestañas de resultado) donde un futuro
  tab "Acta" encajará sin reordenar nada.

## Cómo evaluar el resultado

Capturas en móvil (390px) y escritorio: estado inicial, archivo cargado y
resultados con métricas/sentimiento. Criterio de éxito: en el estado inicial se
ven exactamente **una decisión** (fuente del audio) **y una acción** (subir);
en resultados, el resumen cabe en una pantalla de móvil sin scroll.
