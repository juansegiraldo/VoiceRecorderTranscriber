# Captura nativa de audio en Windows 11 — investigación de opciones

> Documento de investigación. No es código. Escrito 2026-06-18.
> Objetivo: grabar **micrófono + audio del sistema** (las otras voces de Zoom/Teams)
> **mezclados en una sola pista**, en Windows 11, **sin instalar drivers extra**, en Python,
> y que más adelante se pueda empaquetar en una app de escritorio.

---

## TL;DR — la buena noticia

**WASAPI loopback es nativo de Windows 10/11.** Puedes capturar lo que sale por tus
altavoces/auriculares (las voces de los demás en la llamada) **sin instalar VB-Cable ni
ningún driver virtual**. Esto significa que tu primer paso —captura nativa— es **viable con
un script de Python**, sin construir todavía una app de escritorio.

- **Elección recomendada: `PyAudioWPatch`** (fork de PyAudio con loopback WASAPI explícito).
- **Respaldo: `python-soundcard`** (más simple, multiplataforma, pero con un bug de mono).
- **Evitar:** `sounddevice` vanilla (no expone loopback), `pycaw` (no graba), FFmpeg +
  cable virtual (añade driver de pago y latencia).

Y lo mejor: el resultado de la captura (un WAV/MP3) **entra directo en tu
`transcribe_with_deepgram` actual**. No reescribes el motor de transcripción.

---

## El problema técnico, en concreto

"Captura nativa" = grabar **dos fuentes a la vez** y mezclarlas:

1. **Tu micrófono** (tu voz) — trivial, cualquier librería lo hace.
2. **El audio que sale por tus altavoces** (las otras voces) — esto es *loopback*, lo difícil.

En Windows 11, el SO expone un "dispositivo de loopback" por cada salida de audio. Una
librería con soporte WASAPI loopback lo enumera como si fuera un micrófono más, y lo grabas
igual que cualquier entrada. **No hace falta "Stereo Mix" ni cables virtuales.**

---

## Comparación de librerías

| Librería | Loopback nativo Win11 | ¿Driver extra? | Empaqueta (PyInstaller) | Mantenimiento | Veredicto |
|---|---|---|---|---|---|
| **PyAudioWPatch** | Sí, explícito | No | Limpio (wheels con DLL) | Activo (ene 2026) | **PRINCIPAL** |
| **python-soundcard** | Sí (sigiloso) | No | Limpio (hook CFFI) | Activo (abr 2026) | **RESPALDO** |
| **sounddevice** | No expuesto | No | Limpio | Activo | Descartar para loopback |
| **comtypes/ctypes (WASAPI crudo)** | Sí (manual) | No | Limpio (Python puro) | Estable | Sobre-ingeniería |
| **pycaw** | No (no graba, solo controla volumen) | No | Limpio | Activo | No sirve para capturar |
| **FFmpeg + dshow/cable** | No sin recompilar | Sí (VB-Cable, pago) | Manual (binario aparte) | Estancado | Descartar |

### 1. PyAudioWPatch — recomendado

- Fork de PyAudio que **añade enumeración explícita de dispositivos loopback** WASAPI.
- Licencia MIT. Último release enero 2026. Usado en grabadores tipo Audacity.
- El loopback aparece como dispositivo de entrada; se localiza con
  `get_loopback_device_info_generator()` buscando el que coincide con tu altavoz por defecto.
- **Gotchas:** usar `exception_on_overflow=False` al leer; capturar a la tasa del
  dispositivo (normalmente 48 kHz); solo modo *shared* soporta loopback (no exclusivo).
- **Empaquetado:** trae wheels precompilados con la DLL de PortAudio → PyInstaller sin dolor.

### 2. python-soundcard — respaldo

- API más simple, CFFI puro, **multiplataforma** (útil si algún día quieres macOS).
- Licencia BSD. Último release abril 2026.
- **Bug importante en Windows:** grabar en **mono da basura**; hay que capturar en
  **estéreo (2 canales)** y luego mezclar a mono en NumPy (`stereo.mean(axis=1)`).
- A veces ignora el `blocksize` en modo shared.

### 3–6. Descartados (y por qué)

- **sounddevice vanilla:** no enumera loopback con los wheels de PyPI; tendrías que cambiar
  el build de PortAudio. Si vas a hacer eso, mejor usa PyAudioWPatch directamente.
- **comtypes/WASAPI crudo:** 100+ líneas de boilerplate COM para algo que PyAudioWPatch
  hace en 20. Solo valdría si necesitaras cancelación de eco o latencia sub-5ms.
- **pycaw:** sirve para *controlar* volumen/mute por app, **no para grabar** audio.
- **FFmpeg + cable virtual:** requiere VB-Cable (driver de pago), añade 20–100ms de
  latencia y problemas de sincronización entre dos procesos. No vale la pena.

---

## Los dos retos transversales (a resolver sí o sí)

### A. Alineación de tasas de muestreo (sample rate)

- El **loopback** corre a la tasa del mezclador del sistema: normalmente **48 kHz**.
- El **micrófono** suele ser 44.1 o 48 kHz (depende del dispositivo).
- Deepgram/Whisper prefieren PCM; Deepgram acepta varias tasas, pero conviene normalizar.
- **Regla práctica:** captura **ambas fuentes a la misma tasa** (ej. 48 kHz) para no
  introducir desfase, **mezcla primero**, y **remuestrea después** (con `scipy.signal.resample`
  o `librosa.resample`). Remuestrear antes de mezclar introduce artefactos.

### B. Sincronización / deriva (drift)

- Dos streams independientes **no comparten reloj**: con el tiempo se desfasan (clicks,
  efecto "chipmunk").
- **Mitigación:** capturarlos a la misma tasa y, para grabaciones largas, leer en bloques
  pequeños e ir mezclando/escribiendo de forma incremental en vez de dos buffers gigantes
  al final. (La sincronización perfecta es un problema conocido; para una v1 de reuniones,
  capturar a igual tasa y mezclar por bloques es suficiente.)

### Mezcla a una pista (lo que pediste)

```python
# pseudo — mezcla simple de dos arrays NumPy a la misma tasa
mixed = mic_audio + loopback_audio          # sumar
mixed = mixed / np.max(np.abs(mixed)) * 0.95 # normalizar para no saturar (clipping)
# luego remuestrear si hace falta y guardar a WAV/MP3 → transcribe_with_deepgram(...)
```

> Nota sobre diarización: al mezclar a **una sola pista** pierdes la info de "qué pista es
> la mía". Deepgram igual separa speakers por voz, así que la diarización sigue funcionando;
> solo que el etiquetado "Speaker 0/1" no sabe cuál eres tú. Si en el futuro quieres
> distinguirte con certeza, ahí conviene grabar **pistas separadas** (decisión para más
> adelante; hoy elegiste mezcladas, que es lo más simple).

---

## Recomendación final y siguiente paso

**Stack para el prototipo (cuando decidas construirlo):**
`PyAudioWPatch` (captura) + `numpy`/`scipy` (mezcla y remuestreo) + `soundfile` o tu
`pydub` actual (guardar) → enchufar a `transcribe_with_deepgram` (ya lo tienes).

**Camino sugerido:**
1. **Prototipo Python (script):** capturar micro + loopback a 48 kHz, mezclar, guardar MP3,
   transcribir con tu motor actual. Valida lo más riesgoso (la captura) sin construir UI.
2. **Bucle de grabación + controles:** botón grabar/parar, indicador de nivel, guardar a
   archivo. Sigue siendo Python; puede vivir junto a la app Streamlit o como script aparte.
3. **App de escritorio:** envolver en Tauri (binario liviano, Rust) o Electron (más
   ecosistema JS) cuando la captura ya esté probada. Aquí entran firma de código y
   distribución. **No empieces por aquí.**

**Por qué este orden:** lo único técnicamente incierto es la captura. PyAudioWPatch en
Windows 11 lo hace sin drivers, así que un script lo confirma en horas. Todo lo demás (UI,
empaquetado) es trabajo conocido que no conviene afrontar hasta validar la captura.

---

## Cómo encaja con el resto del plan

Esto corresponde a la **Fase 3** de [../ROADMAP.md](../ROADMAP.md) (app de escritorio para
Zoom/Teams), pero tú decidiste **adelantarla como primer paso**. Es una decisión defendible:
si la captura nativa es la pieza que más te importa y la más incierta, validarla primero
reduce el riesgo del proyecto entero. El acta con IA y el streaming en vivo (Fases 1 y 2)
quedan **después**, tal como dijiste, y reutilizarán tanto el motor de transcripción como
esta captura.
