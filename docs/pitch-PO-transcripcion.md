# Propuesta: transcripción de reuniones como feature de MeetingMind

> One-pager para el Product Owner + plan de demo. Audiencia: PO con base técnica.
> Fecha: 2026-06-18. App: **MeetingMind** — tracker de reuniones con clientes
> (Lovable, ~20 usuarios, ~2-3 reuniones/día por usuario "en full").

---

## El pitch en 30 segundos

MeetingMind ya registra cada reunión con cliente: título, asistentes y **acciones abiertas**.
Pero hoy esas acciones y el resumen se **teclean a mano** después de cada reunión: el usuario
recuerda lo que se dijo, lo resume, y escribe los próximos pasos. **15-20 minutos de trabajo
manual por reunión, y la mitad de los detalles se pierden.**

Propongo que MeetingMind **escuche la reunión y lo haga solo**: un botón "Transcribir reunión"
→ habla → "Detener" → la transcripción, el acta y las **acciones abiertas extraídas
automáticamente** quedan guardadas en la ficha del cliente. No es una feature al lado: **es lo
que MeetingMind ya hace, pero sin teclear.**

Lo mejor: **se construye sobre la Lovable app que ya tenemos viva** (React + Supabase), sin
app nueva, sin extensión, sin servidor nuevo. Ya tengo la viabilidad técnica investigada y un
motor de transcripción funcionando. No es "a ver si se puede" — es "se monta sobre exactamente
lo que ya corre en producción".

---

## El problema (desde el usuario)

MeetingMind es un tracker de reuniones con clientes: el equipo registra cada reunión (ISDIN,
Casa Tarradellas, Grupo Presidente, CHC Energía...) con sus asistentes y las **acciones
abiertas** que quedan pendientes. El problema está en **cómo se llena esa información hoy**:

- Tras cada reunión, el usuario **teclea a mano** el resumen y las acciones abiertas, de
  memoria. Es trabajo repetitivo y tedioso.
- Lo que no se anota **se pierde**: detalles, acuerdos, matices del cliente. La ficha de la
  reunión queda incompleta o vacía (de hecho, hoy muchas reuniones no tienen acciones
  registradas).
- Capturar bien una reunión exige una herramienta externa de transcripción → exportar →
  copiar → pegar en un GPT → volver a MeetingMind. **Nadie hace ese flujo de 5 pasos**, así
  que simplemente no se captura.

**Tamaño del dolor (a pleno funcionamiento):** ~20 usuarios × ~2-3 reuniones/día × ~15-20 min
de trabajo post-reunión cada una ≈ **3-4 horas/semana por usuario → del orden de 70-80
horas/semana en todo el equipo**. Y, más importante que las horas: **las acciones y acuerdos
que hoy se pierden quedarían capturados**, que es justo la razón de existir de MeetingMind.

---

## La solución

Un flujo de transcripción **nativo dentro de la app**, conectado a la ficha de reunión que ya
existe:

1. Desde una reunión (la entidad que MeetingMind ya maneja: cliente + título + asistentes), el
   usuario pulsa **"Transcribir reunión"**.
2. La app captura el audio de la reunión (en el navegador) + su micrófono.
3. Muestra la transcripción con **hablantes separados** y la **guarda en la ficha del cliente**.
4. Un botón **"Generar acta"** produce con IA el resumen estructurado **y rellena las "acciones
   abiertas" automáticamente** — la columna que hoy se llena a mano.

El resultado vive **dentro de MeetingMind**, alimentando los campos que el producto ya tiene
(resumen, asistentes, acciones abiertas). No es una pantalla nueva aislada: **enriquece el
core**.

---

## Por qué ahora y por qué nosotros

- **El usuario ya paga por esto en otro lado.** Herramientas como Granola ($14/usuario/mes)
  o Fireflies viven de esta funcionalidad. Si la tenemos integrada, el usuario no necesita una
  segunda herramienta — y el dato no se va a un competidor.
- **Ventaja de integración (nuestro foso):** Granola/Fireflies son apps genéricas que escupen
  un acta suelta. MeetingMind ya sabe **quién es el cliente, quiénes asisten y qué acciones
  hay abiertas**. La transcripción cae **en contexto**: el acta se asocia al cliente correcto y
  las acciones extraídas pueblan la columna que ya existe. Una herramienta genérica no puede
  hacer eso porque no conoce nuestro modelo de datos.
- **El ahorro de tiempo es medible y demostrable**, no abstracto — y además **mejora la calidad
  del dato** que MeetingMind existe para capturar.

---

## Factibilidad técnica (resumen para PO técnico)

Ya investigado y validado:

- **Se construye sobre la Lovable app actual** (React + Supabase) — el mismo stack que ya
  corre en producción. No requiere app de escritorio, ni extensión de navegador, ni servidor
  nuevo. Para Lovable, esta feature es "más de lo mismo": componentes React + Edge Functions +
  una tabla nueva. Encaja en el paradigma de prompts incrementales con el que ya construimos.
- **Captura de audio:** APIs estándar del navegador (`getDisplayMedia` para el audio de la
  reunión + micrófono). Funciona en Chrome/Edge.
- **Transcripción:** Deepgram (con detección de hablantes). La clave de API se guarda segura
  en una Edge Function de Supabase — nunca toca el navegador.
- **Acta con IA:** un LLM (Claude/GPT) sobre la transcripción.
- **Ya existe un motor de transcripción funcionando** (proyecto VoiceTranscriber), con la
  lógica difícil —diarización por hablantes— resuelta y reutilizable.

**El único trade-off honesto:** para capturar el audio de la reunión dentro de una web app, el
navegador exige que el usuario, al iniciar, elija la pestaña y marque "compartir audio". Es un
paso visible (a diferencia de Granola, que no pide nada porque es una app de escritorio). Es
aceptable para una v1 y se suaviza con un mini-tutorial la primera vez. *(Detalle técnico en
[PRD-feature-transcripcion-en-lovable.md](PRD-feature-transcripcion-en-lovable.md).)*

---

## ¿Qué motor de transcripción usamos? (Lovable nativo vs. Deepgram)

> Pregunta que probablemente hará el PO: *"¿No trae Lovable algo de transcripción ya?"*
> Respuesta corta: **no, y aunque algo se le aproxima, no sirve para reuniones.**

**Lovable NO tiene transcripción de voz propia.** Su IA integrada (sobre modelos Gemini) hace
texto e imágenes —chat, resúmenes, traducción—, pero **no expone speech-to-text**. Para
transcribir audio, el patrón estándar y recomendado por la propia Lovable es: una **Edge
Function de Supabase que llama a una API de transcripción externa**. O sea, hay que elegir
proveedor sí o sí.

**Por qué Deepgram es la elección correcta para MeetingMind:**

| Criterio | Deepgram | Whisper (OpenAI) | Gemini (vía Lovable) |
|---|---|---|---|
| Detección de hablantes (diarización) | ✅ Nativa | ❌ Requiere otra herramienta | ❌ No |
| Tiempo real (streaming) | ✅ Sí | ❌ Solo por lotes | ❌ No |
| Español + Inglés | ✅ Buena calidad | ✅ Amplio | ➖ No para transcribir |
| **Ya tenemos código funcionando** | ✅ **Sí, reutilizable** | ❌ | ❌ |
| Apto para reuniones | ⭐⭐⭐⭐⭐ | ⭐⭐ | ❌ No viable |

Tres razones de peso:
1. **Diarización (saber quién dijo qué) integrada** — clave en una reunión. Whisper no la trae;
   Gemini tampoco.
2. **Ya tenemos un motor Deepgram funcionando** con la lógica difícil resuelta (separación de
   hablantes incluso en audios largos partidos en trozos). **Cero costo de migración** — se
   reutiliza.
3. **Costo razonable y conocido:** ~**$0.46–0.58 por hora** de audio con diarización. Con ~20
   usuarios es un gasto menor y **medible** (lo cuantificamos en la demo).

**Conclusión:** Lovable aporta el frontend, la base de datos y el backend seguro (la clave de
Deepgram vive en una Edge Function, nunca en el navegador); **Deepgram aporta la transcripción**.
No competimos contra lo que trae Lovable porque Lovable no trae esto. *(Detalle y comparación
completa de proveedores en [PRD-feature-transcripcion-en-lovable.md](PRD-feature-transcripcion-en-lovable.md).)*

---

## Alcance y esfuerzo

| Versión | Qué incluye | Esfuerzo (orden de magnitud) |
|---|---|---|
| **PoC / demo** | Capturar audio + transcribir + mostrar texto. Sin guardar, sin pulir. | Días |
| **v1** | + Guardar en la ficha de reunión/cliente, detección de hablantes, UI básica | Semanas |
| **v1.1** | + "Generar acta" con IA, plantillas, exportar/copiar | Semanas adicionales |

No es un proyecto de meses para tener algo demostrable. **Una demo funcional es cuestión de
días.**

---

## Riesgos y cómo los manejamos

| Riesgo | Mitigación |
|---|---|
| Fricción del selector de "compartir audio" | Mini-tutorial la primera vez; detectar y avisar si falta el audio |
| Privacidad (grabamos reuniones, mandamos audio a Deepgram) | Aviso claro al usuario + política de privacidad; definir retención |
| Costo variable de API (Deepgram/LLM por minuto) | ~$0.46–0.58/hora con diarización; con ~20 usuarios es gasto menor. Medir exacto en la PoC y acotar por cuota si hace falta |
| Calidad de transcripción según idioma/audio | Deepgram + fallback; validar con audio real de reuniones con clientes (español, a veces inglés) en la demo |

---

## Plan de demo / Prueba de Concepto

**Objetivo:** enseñar la feature funcionando con audio real, para decidir con hechos y no con
promesas. Demuestra lo más incierto (captura + transcripción) antes de invertir en pulido.

**Qué mostrará la demo:**
1. Botón "Transcribir reunión" dentro de un prototipo (o de la propia MeetingMind si se permite).
2. Captura en vivo de una reunión de prueba con cliente (Meet/Teams web) + micrófono.
3. Transcripción apareciendo, con hablantes separados.
4. Botón "Generar acta" produciendo el resumen **y las acciones abiertas extraídas** — el campo
   que hoy se llena a mano.

**Qué medir durante la demo (los números que pedirá el PO):**
- Tiempo real ahorrado vs. el flujo actual (cronometrar: teclear a mano vs. botón).
- Calidad de la transcripción con audio real de una reunión con cliente (español, a veces inglés).
- Calidad de las acciones extraídas vs. las que pondría un humano.
- Costo aproximado por reunión (minutos de Deepgram + tokens de LLM).

**Pasos para llegar a la demo:**
1. Elegir 1 reunión-tipo con cliente como caso (p. ej. una de seguimiento con acciones claras).
2. Montar el prototipo de captura + transcripción (reutilizando el motor existente).
3. Conectar Deepgram vía una Edge Function de prueba + un LLM para el acta/acciones.
4. Grabar una reunión de muestra y cronometrar el ahorro.
5. Presentar: demo en vivo + este one-pager + los números medidos.

**Decisión que se le pide al PO tras la demo:** ¿seguimos a v1 (guardado + historial)? ¿con
qué prioridad frente al resto del roadmap?

---

## La pregunta para el PO

No pido construir el producto completo. Pido **luz verde para una PoC de unos días** que
demuestre la feature funcionando con el audio de una reunión real con cliente. Con eso sobre
la mesa, decidimos juntos si entra al roadmap y con qué prioridad.

---

### Notas para ti antes de presentar (borrar esta sección)

- Ya rellenado: nombre **MeetingMind** (tracker de reuniones con clientes), ~20 usuarios,
  ~2-3 reuniones/día por usuario en full, motor **Deepgram** justificado, y el ángulo clave:
  la feature **rellena las "acciones abiertas" que hoy se teclean a mano**.
- Verifica/ajusta el número de PoC ("unos días" → pon los días concretos que te comprometas).
- El argumento más fuerte para este PO no es "ahorrar tiempo" en abstracto, sino **"capturamos
  las acciones y acuerdos que hoy se pierden"** — eso es la razón de existir de MeetingMind.
  Apóyate en la captura de pantalla: muchas reuniones tienen "—" en Open Actions hoy.
- Factibilidad ya confirmada: MeetingMind corre en Lovable (React + Supabase), exactamente el
  stack para el que está escrita esta propuesta. No hay que reescribir nada de arquitectura.
- Material de respaldo si el PO pide profundidad técnica:
  [PRD-feature-transcripcion-en-lovable.md](PRD-feature-transcripcion-en-lovable.md) y el
  [ROADMAP.md](../ROADMAP.md).
```
