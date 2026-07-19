# Rediseño UX/UI v2 — "menos ruido, más señal"

> Plan de diseño. Escrito 2026-07-19, rama `claude/app-visual-ux-redesign-xfbmar`.
> Mockup navegable: [docs/mockups/rediseno-v2.html](mockups/rediseno-v2.html)
> (HTML autocontenido — ábrelo en el navegador).

## 1. Contexto y objetivo

La app funcionaba muy bien cuando hacía una sola cosa: subir un audio y
transcribirlo. Desde entonces ganó capacidades reales (nova-3 con fallback,
diarización v2, métricas de conversación, sentimiento, Google Drive, recorte) y
cada una añadió un control, un emoji y un texto más **a la misma columna**. El
resultado: la pantalla de hoy tiene 11 controles a la vista, 50 emojis (23
distintos, auditados en `AppTranscribe.py`; 23 más llegan desde `core/`) y dos
idiomas de interfaz mezclados.

**Objetivo de la v2:** no quitar ninguna capacidad — reordenarlas. El caso feliz
vuelve a ser *subir → Transcribir* sin tocar nada; el análisis nuevo (lo mejor de
la app) pasa de apéndice en expanders a presentación protagonista.

## 2. Diagnóstico (resumen)

| # | Problema | Evidencia en el código |
|---|----------|------------------------|
| P1 | Orden invertido: los ajustes se renderizan **después** de los resultados | `st.selectbox("Transcription Model", …)` está ~250 líneas después del botón `Start Transcription` |
| P2 | Todo desplegado a la vez: recorte, 2 selectores, 3 casillas siempre visibles | bloque de trimming + selectores al final de `main()` |
| P3 | El emoji es el sistema visual: estados ✅🟡⚠️, timeline 😊😐🙁, labels 🎤📁☁️📊 | `RATING_ICONS`, `sentiment_timeline_emoji`, labels de widgets |
| P4 | Dos idiomas: "Insert audio file" / "Results" vs "Subir archivo" / "Métricas" | strings de `main()` |
| P5 | El resultado no luce: transcript en `st.code` monoespaciado, métricas como viñetas | render de resultados en `main()` |
| P6 | Sin identidad: no existe `.streamlit/config.toml`; 3 bloques CSS sueltos con clases muertas (`.main-header`, `.success-box`…) | raíz del repo + `st.markdown(<style>)` ×3 |

## 3. Principios

1. **Cero decisiones en el caso feliz.** Los defaults actuales ya son los buenos
   (Deepgram + auto + diarización + métricas). La UI debe fiarse de ellos.
2. **Revelación progresiva.** Recorte y ajustes existen, plegados y a un toque,
   con su estado resumido en una línea ("Deepgram · idioma auto · diarización
   activada — Ajustes").
3. **El texto manda, el color acompaña, el emoji se retira.** Semántica en chips
   de texto («Bien», «Justo», «Alto»); color como refuerzo, nunca solo.
4. **Una sola voz, en español.** Todo el microcopy en español.
5. **El resultado es el producto.** Transcripción legible por turnos; métricas y
   sentimiento como tarjetas y gráficos, no como listas.

## 4. Especificación por estados

La app sigue siendo una página Streamlit `layout="centered"` mobile-first con
cuatro estados. Componentes concretos:

### Estado 1 — Inicio (sin archivo)
- Wordmark `VoiceTranscriber` + glifo de onda (texto/CSS, sin emoji 🎤 y sin
  header de 3rem). Tagline corto.
- `st.tabs(["Archivo", "Google Drive"])` (sin 📁☁️).
- Uploader con copy: "Arrastra tu audio aquí o pulsa para elegir — MP3 · WAV ·
  M4A · MP4, hasta 2 GB".
- **Fila de ajustes**: resumen en una línea + `st.popover("Ajustes")` (o
  `st.expander` como fallback) que contiene:
  - Modelo: `st.selectbox` con `options=["Deepgram", "OpenAI Whisper"]` **sin
    cambiar los valores** y `format_func` para etiquetas bonitas
    ("Deepgram — recomendado", "OpenAI Whisper").
  - Idioma: opciones `["Automático", "Español", "Inglés"]` mapeadas con un
    **dict explícito** `{label: code}` — hoy se infiere con `'auto' in
    language_ui.lower()`, frágil ante cambios de label.
  - Toggles diarización / métricas / sentimiento (`st.toggle`), sentimiento con
    caption "solo audio en inglés".

### Estado 2 — Archivo cargado
- Tarjeta de archivo: nombre, `duración · tamaño`, acción "Quitar".
- `st.expander("Recortar audio — completo (00:00–42:18)")` **colapsado**, con el
  slider + 2 number inputs actuales dentro (mismas keys y callbacks).
- Fila de ajustes (idéntica al estado 1) **encima** del CTA.
- CTA único: `st.button("Transcribir", type="primary")` full-width.

### Estado 3 — Progreso
- `st.status("Transcribiendo con Deepgram…")` con subpasos de texto alimentados
  por el `progress_cb` existente: "Audio preparado" → "Enviado a Deepgram
  (nova-3, idioma auto)" → "Identificando hablantes" → "Calculando métricas".
  Sin 🔄📦🔍; la barra `st.progress` puede vivir dentro del status.

### Estado 4 — Resultados
- Cabecera: nombre de archivo + **chips** de metadatos (idioma detectado,
  modelo usado, duración) — sustituye al caption "🌐 Idioma… · modelo…".
- `st.tabs(["Transcripción", "Métricas", "Sentimiento"])`; la pestaña
  Sentimiento solo si `analysis["sentiment"]` existe.
- **Transcripción**: turnos agrupados desde `analysis["utterances"]` — chip de
  color + "Hablante N" + timestamp mono pequeño + párrafo con tipografía de
  lectura (HTML vía `st.markdown`, escapando el texto). Fallback sin
  diarización: texto plano en un contenedor legible (no `st.code`; añadir botón
  Copiar propio o conservar `st.code` solo como fallback).
- **Métricas**: fila de tarjetas de cifra (`insights.overall`: hablantes,
  palabras, ppm, interrupciones, silencio, preguntas) con chips de valoración
  donde exista rating; "Reparto del habla" como barras horizontales por
  hablante (etiqueta directa con %); tarjetas por hablante (talk time, palabras,
  ppm + chip, muletillas/100 + chip); "Observaciones" (los `feedback` bullets);
  expander colapsado "Cómo leer estas métricas" (`METRIC_GUIDE_ES`).
- **Sentimiento**: tono general en palabras + cifra (conserva
  `describe_sentiment_score`); timeline de 12 tramos como **barra segmentada
  divergente** (azul positivo ↔ rojo negativo, gris neutro, leyenda de texto) en
  lugar de la línea de emojis; filas por hablante. Nota: "Deepgram solo analiza
  sentimiento en audio en inglés".
- Acciones agrupadas al final: "Descargar .txt" + "Informe completo" y un hueco
  reservado (botón deshabilitado o placeholder) para **"Generar acta"** — la
  Fase 1 del [ROADMAP](../ROADMAP.md) ya tiene sitio en el diseño.

## 5. Tema e identidad

Crear `.streamlit/config.toml` (hoy no existe):

```toml
[theme]
primaryColor = "#0E7263"            # petróleo — acciones y foco
backgroundColor = "#FAFBFA"         # papel neutro con sesgo frío
secondaryBackgroundColor = "#F1F4F2" # paneles, inputs, expanders
textColor = "#21272A"
font = "sans serif"
# Con streamlit >= 1.46 además:
# baseRadius = "10px"
```

- Subir el mínimo en `requirements.txt` a `streamlit>=1.46` (junio 2025): trae
  `st.popover`, `st.toggle`, `st.status`, pestañas estables y las claves de tema
  extendidas. (Todo lo esencial existe desde 1.32–1.40; 1.46 es el colchón
  cómodo.)
- Un **único** bloque CSS al inicio de `main()` que reemplace los tres actuales:
  tipografía/jerarquía, ancho 720px, tap targets ≥44px (se conserva), chips,
  badges, turnos de transcripción. Borrar las clases muertas.

### Colores de datos (validados para daltonismo sobre blanco)

| Rol | Hex |
|-----|-----|
| Hablante 1 / 2 / 3 / 4 | `#2A78D6` / `#008300` / `#E87BA4` / `#EDA100` (orden fijo, nunca reciclar) |
| Sentimiento + / neutro / − | `#2A78D6` / `#ECEEED` / `#E34948` |
| Chip Bien / Justo / Alto | texto `#175617` sobre `#E5F3E5` / `#7A4E00` sobre `#FBF0D7` / `#8C3E1A` sobre `#FBE7DE` |

Regla: la identidad viaja **siempre en texto** (etiqueta "Hablante N", palabra
en el chip, leyenda en el timeline); el color nunca va solo. Hablantes 3–4 son
más claros: siempre con etiqueta pegada (regla de "relief" de la validación).

### Política de emoji

| Dónde | Política |
|-------|----------|
| Controles, títulos, pestañas | **Cero.** |
| Estados y valoraciones en pantalla | Chips de texto con color; ✅🟡⚠️ desaparecen de la UI. |
| Errores/avisos | Texto directo; `st.error`/`st.warning` ya ponen el color (fuera ❌). |
| Informe `.txt` (`build_report`) | **Se mantienen** ✅🟡⚠️ y el timeline de emojis: en texto plano no hay color, ahí el emoji sí trabaja. `build_report` no se toca (contrato compartido con el CLI). |

## 6. Fases de implementación

Cada fase es entregable por sí sola; la A ya "se siente" como la v2.

- **Fase A — Base serena (~½ día):** `config.toml` + un solo bloque CSS + quitar
  emojis de labels + microcopy 100% español + mover ajustes encima del CTA (en
  popover/expander con resumen) + dict explícito de idiomas + `format_func` en
  el selector de modelo. Sin tocar lógica.
- **Fase B — Flujo por estados (~1 día):** tarjeta de archivo con "Quitar",
  recorte en expander colapsado con estado resumido, `st.status` para progreso,
  chips de metadatos en resultados.
- **Fase C — Resultados que lucen (1–2 días):** pestañas de resultados, turnos
  por hablante, tarjetas de cifra, barras de reparto, chips de valoración,
  timeline divergente, acciones agrupadas + hueco "Generar acta".
- **Fase D — Pulido (continuo):** QA en móvil real, contraste AA, foco visible,
  estados vacíos/error con el mismo tono.

## 7. Invariantes — qué NO se toca

- Valores internos `"Deepgram"` / `"OpenAI Whisper"` (la lógica compara esos
  strings exactos); las etiquetas visibles cambian solo vía `format_func`.
- Keys y callbacks del recorte (`trim_range_slider`, `trim_start_input`,
  `trim_end_input`, `trim_state_key`): cambian de sitio, no de mecánica.
- Caché de conversión (`converted_mp3_path`/`converted_mp3_file_key`, `file_key`
  por nombre+tamaño+tipo) — invalidarla mal re-convierte en cada rerun.
- Flujo OAuth de Drive (callback al inicio de `main()`, PKCE en disco) y
  `DriveFile` con `.name` conservando la extensión real.
- `layout="centered"`, tap targets ≥44px, `get_secret()` para toda clave.
- Orden de imports FFmpeg→pydub al inicio del archivo.
- `core/transcription.py` y `build_report` (formato compartido con el CLI);
  los tests de `tests/` deben seguir pasando sin cambios.

## 8. Criterios de aceptación

- [ ] 0 emojis en labels de widgets, títulos y pestañas de la app.
- [ ] 1 sola acción primaria visible por estado; ajustes a ≤1 toque.
- [ ] Ajustes renderizados antes del CTA (leídos por las mismas keys de
      `session_state`).
- [ ] 100% del microcopy en español.
- [ ] Transcripción diarizada renderizada por turnos con identidad texto+color.
- [ ] Timeline de sentimiento sin emojis en pantalla (el informe .txt intacto).
- [ ] `python -m unittest discover tests` en verde sin modificar tests.
- [ ] Mismo comportamiento funcional: caché de conversión, recorte, Drive y
      fallbacks de modelo intactos.
