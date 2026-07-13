# Extensión de Chrome — captura de audio de Teams/Meet web (investigación)

> Documento de investigación. No es código. Escrito 2026-06-18.
> Objetivo: extensión Manifest V3 que capture **el audio de la reunión** (Teams web / Google
> Meet, dentro del navegador) **+ tu micrófono**, los mezcle, y los mande a transcribir.
> Razón del enfoque: **un solo código para Mac + Windows + Linux** sin app de escritorio.

---

## TL;DR — sí es viable

- **Funciona.** Herramientas comerciales (Fireflies, MeetGeek, Tactiq) capturan Teams web y
  Meet exactamente con esta técnica. No hay bloqueo de Microsoft a nivel de API.
- **La regla del juego:** la reunión debe abrirse en el **navegador** (`teams.microsoft.com`,
  Meet). La app de escritorio de Teams queda fuera de alcance. (Ya lo aceptaste.)
- **Requiere Chrome 116+** (marzo 2024). Funciona en Chrome, Edge, Brave (todos Chromium).
  No Firefox ni Safari.
- **Necesitas un backend mínimo desde el día uno** — la clave de Deepgram no puede ir en la
  extensión.

---

## Cómo funciona (el patrón correcto en MV3)

En Manifest V3 ya no hay "background page" persistente, solo un **service worker** que Chrome
suspende cuando quiere. Eso rompe la captura de audio, que necesita un DOM vivo. La solución
oficial (Chrome 116+) tiene **tres piezas**:

```
1. Service worker  → getMediaStreamId({targetTabId})  [tras click del usuario]
                     ↓ envía streamId
2. Offscreen doc   → getUserMedia({chromeMediaSource:'tab', chromeMediaSourceId})
                     + getUserMedia({audio:true})  [tu micro]
                     → mezcla con Web Audio API → MediaRecorder (chunks 250ms)
                     ↓ WebSocket
3. Deepgram / tu backend → transcripción
```

- **Service worker:** solo escucha el click y pide el `streamId`. Mínimo.
- **Offscreen document** (`chrome.offscreen`): documento DOM oculto y persistente donde sí
  vive el audio. Razón `'USER_MEDIA'` (no se autocierra). **Solo puede haber uno por perfil.**
- **Permisos en manifest:** `"tabCapture"` + `"offscreen"`. No necesitas `activeTab` ni
  host_permissions para capturar.
- **Requiere gesto del usuario:** `getMediaStreamId()` debe llamarse tras un click (botón de
  la extensión o popup). No se puede arrancar desde un timer.

### Mezclar micro + audio de la pestaña

```js
const ctx = new AudioContext();
const tabSrc = ctx.createMediaStreamSource(tabStream);   // las otras voces
const micSrc = ctx.createMediaStreamSource(micStream);   // tu voz
const mixer  = ctx.createMediaStreamDestination();
tabSrc.connect(mixer);
micSrc.connect(mixer);
// mixer.stream = una sola pista con ambos → MediaRecorder
```

---

## ⚠️ Las 5 trampas que pueden hundir el proyecto

Estas no son opinables; son bugs conocidos. Conviene tenerlas presentes desde el prototipo.

### 1. El audio se MUTEA para el usuario
Al capturar la pestaña con `tabCapture`, **el usuario deja de oír la reunión** (Chrome trata
la captura como "consumir" el audio). **Fix obligatorio:** reconectar el stream capturado a
`audioContext.destination` para que el usuario siga oyendo mientras grabas. Es el bug más
difícil de diagnosticar porque parece que "rompiste el audio del PC".

### 2. El streamId caduca en ~5 segundos
Si tardas entre `getMediaStreamId()` (service worker) y `getUserMedia()` (offscreen), el ID
expira y la captura falla **en silencio**. Hay que pasar el ID inmediatamente por mensaje.

### 3. Cancelación de eco + 2 canales = silencio
Si pones `echoCancellation: true` con `channelCount: 2`, MediaRecorder graba **solo silencio**
(quirk de Chromium). **Fix:** desactivar `echoCancellation`, `noiseSuppression` y
`autoGainControl` en el micro cuando vas a mezclar.

### 4. Solo un offscreen document por perfil
Si el offscreen doc se cae o se cierra mal, no puedes crear otro hasta que el anterior se
libere (puede tardar). Hay que cerrar explícitamente con `chrome.offscreen.closeDocument()`
al parar, y manejar reinicios.

### 5. Rechazo en la Chrome Web Store por privacidad
Grabar audio y mandarlo a un tercero (Deepgram) **exige**: política de privacidad pública +
aviso claro **en la UI** ("este audio se envía a Deepgram para transcribir") + declaración de
"Limited Use". Sin esto, rechazan la extensión y reenviarla tarda semanas. Fireflies/MeetGeek
están aprobadas → es cuestión de transparencia, no de prohibición.

---

## Teams web en concreto

- Teams web usa **WebRTC** y reproduce las voces por un elemento de audio del tab → `tabCapture`
  lo captura. Funciona.
- **Caso límite:** si el usuario está en escritorio remoto (RDP/RustDesk), Teams puede mutear
  el audio entrante *antes* de llegar al navegador. No es problema de tu extensión, pero
  conviene saberlo.
- Microsoft **no bloquea** la captura por extensión. Algunas empresas desactivan el cliente
  web por política — ahí no hay nada que hacer.
- Google Meet es más sencillo y predecible que Teams; conviene **probar primero con Meet** y
  luego validar Teams web.

---

## `tabCapture` vs `getDisplayMedia` (la alternativa)

`getDisplayMedia({audio:true})` es el selector de "compartir pestaña + audio". Captura lo mismo
que `tabCapture`, pero **obliga al usuario a elegir la pestaña y marcar "compartir audio" cada
vez**. Para un grabador de reuniones, `tabCapture` gana en UX: un click y graba la pestaña
actual. Misma capacidad, peor experiencia en `getDisplayMedia`.

---

## Salida del audio hacia transcripción

- **Recomendado:** `MediaRecorder` con `audio/webm;codecs=opus`, chunks de **250 ms**, por
  **WebSocket** al endpoint de streaming de Deepgram (`wss://api.deepgram.com/v1/listen`).
  Deepgram acepta webm/opus directo. Da transcripción en vivo (~500ms–1s de latencia).
- **Alternativa:** `AudioWorklet` para PCM crudo (más control, más complejo) — solo si
  necesitas latencia sub-100ms o procesamiento propio.
- **Más simple aún (v1):** grabar todo y subir el blob al final (sin tiempo real). Reutiliza
  tu `transcribe_with_deepgram` batch actual casi sin cambios. **Buen punto de partida.**

---

## Arquitectura recomendada

```
Navegador
├─ Pestaña Teams/Meet web ──(el usuario sigue oyendo vía audioContext.destination)
├─ Service worker  → pide streamId al hacer click → lo manda al offscreen
├─ Offscreen doc   → captura tab + micro, mezcla, graba en chunks
└─ Popup           → botón grabar/parar + transcripción en vivo
                              ↓ WebSocket / HTTPS
        Backend propio (proxy)  → guarda la clave de Deepgram, almacena actas
                              ↓
        Deepgram (transcripción)
```

**Por qué el backend es obligatorio:** la clave de Deepgram en la extensión sería pública
(cualquiera la extrae del paquete). El backend hace de proxy y, más adelante, guarda las actas.

---

## Plan por fases (revisado para extensión)

- **Fase A — Prototipo de captura (lo más incierto primero):** extensión MV3 esqueleto que
  capture audio de la pestaña + micro, los mezcle, y **guarde un archivo**. Objetivo único:
  confirmar que `tabCapture` funciona con **Meet** y luego con **Teams web** en tu máquina.
  Aquí se resuelven las trampas 1–4.
- **Fase B — Transcripción:** mandar el audio a transcribir. Empieza por **grabar-y-subir**
  (reusa tu motor batch actual) antes que streaming en vivo.
- **Fase C — Backend proxy:** mover la clave de Deepgram a un servicio mínimo (FastAPI o
  serverless). Necesario antes de cualquier distribución.
- **Fase D — UI + acta:** panel de transcripción en vivo, caja de "mis notas", y generación
  de acta con LLM (la Fase 1 del [../ROADMAP.md](../ROADMAP.md), ahora encima de la extensión).
- **Fase E — Publicación:** política de privacidad, avisos en UI, Limited Use, y subida a la
  Chrome Web Store (trampa 5).

**Orden:** A → B → C → D → E. La Fase A valida lo único técnicamente incierto; el resto es
trabajo conocido.

---

## Relación con los otros documentos

- Esto **reemplaza** la idea de app de escritorio de [captura-nativa-windows.md](captura-nativa-windows.md)
  (WASAPI). Aquel enfoque sigue siendo el único válido si algún día necesitas la **app de
  escritorio de Teams** (no la web), pero por ahora la extensión gana por multiplataforma.
- El motor de transcripción Deepgram y la futura generación de actas se **reutilizan** tal
  cual; lo único nuevo aquí es la **captura en navegador**.
