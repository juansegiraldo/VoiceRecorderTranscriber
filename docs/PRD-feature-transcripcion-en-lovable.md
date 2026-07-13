# PRD — Feature de transcripción de reuniones dentro de una Lovable app

> Documento de producto. Para decidir si vale la pena meter la feature.
> Audiencia: alguien que ya tiene una **Lovable app montada** y quiere añadir
> transcripción de reuniones como funcionalidad. Escrito 2026-06-18.
> Arquitectura elegida: **todo dentro de Lovable** (sin extensión de Chrome).

---

## 1. En una frase

Añadir a tu Lovable app un botón "Transcribir reunión" que graba el audio de una
reunión en el navegador (Meet / Teams web) **+ tu micrófono**, lo transcribe con
Deepgram, y guarda la transcripción (y opcionalmente un acta generada con IA) en tu
propia base de datos, dentro de la misma app.

---

## 2. El problema que resuelve

Hoy, para tener el acta de una reunión, el flujo típico es: grabar con otra herramienta →
exportar el archivo → subirlo a un transcriptor → copiar el texto → pegarlo en un GPT para
el acta. Cinco pasos y tres herramientas. Esta feature lo colapsa en **un botón dentro de la
app que el usuario ya usa**.

Es el mismo valor que vende Granola.ai (notas de reunión sin fricción), pero viviendo dentro
de tu producto en vez de ser una app aparte.

---

## 3. Por qué "todo dentro de Lovable" (y qué se acepta a cambio)

**Decisión:** construir la feature como parte de la web app de Lovable, sin extensión de
Chrome.

**Lo que esto te da:**
- **Una sola base de código y un solo despliegue.** Nada que publicar en la Chrome Web Store,
  nada que revisar durante semanas, nada que mantener por separado.
- **Lovable lo construye casi entero.** Es una web app React + Supabase, justo lo que Lovable
  hace bien.
- **El backend ya está resuelto.** Supabase (lo que Lovable usa) guarda la clave de Deepgram
  de forma segura en una Edge Function. No hace falta montar servidor aparte.

**Lo que se acepta a cambio (importante, sin letra pequeña):**
- La captura usa `getDisplayMedia`, el **selector de "compartir pantalla/pestaña"** del
  navegador. El usuario, al iniciar, ve el cuadro del navegador y debe **elegir la pestaña de
  la reunión y marcar "compartir audio del sistema"**. Cada vez.
- Esto es **menos fluido** que Granola (que captura sin pedir nada), porque una web app normal
  **no puede** capturar el audio de otra pestaña sin ese selector — es una restricción de
  seguridad del navegador, no un límite de Lovable. La captura sin selector (`chrome.tabCapture`)
  **solo existe en extensiones de Chrome**, y una extensión no puede vivir dentro de Lovable.

**Veredicto del trade-off:** si la prioridad es *llegar rápido y mantener una sola cosa*, el
selector es un precio razonable. Si algún día la fluidez tipo Granola se vuelve crítica, la
salida es construir una extensión **aparte** que use esta misma Lovable app como panel y
backend — pero eso es otro proyecto, no esta feature.

---

## 4. Recorrido del usuario (happy path)

1. El usuario abre tu Lovable app y entra a la reunión en **otra pestaña** (Meet o Teams web).
2. Pulsa **"Transcribir reunión"** en tu app.
3. El navegador muestra el selector → el usuario elige la pestaña de la reunión y marca
   **"compartir audio del sistema"**. La app también pide permiso del **micrófono**.
4. La app mezcla ambos audios y muestra la **transcripción en vivo** (o, en una primera
   versión, simplemente graba).
5. Al terminar, el usuario pulsa **"Detener"**. La transcripción queda **guardada** en su
   historial dentro de la app.
6. (Opcional) Pulsa **"Generar acta"** y la app produce un resumen estructurado con IA a
   partir de la transcripción.

---

## 5. Alcance

### Dentro de alcance (v1)
- Botón iniciar/detener transcripción.
- Captura de audio de pestaña (`getDisplayMedia`) + micrófono (`getUserMedia`), mezclados.
- Transcripción vía Deepgram (a través de una Edge Function de Supabase que guarda la clave).
- Guardado de la transcripción en la base de datos del usuario, con historial.
- Detección de hablantes (diarización) — Deepgram la ofrece; etiquetas tipo "Hablante 1/2".

### Dentro de alcance (v1.1 — siguiente)
- Caja de "mis notas" durante la reunión + botón "Generar acta" (IA mezcla notas +
  transcripción). Este es el diferenciador real de Granola.
- Plantillas de acta por tipo de reunión (ventas, 1:1, entrevista).
- Exportar / copiar / descargar la transcripción y el acta.

### Fuera de alcance (explícito)
- Reuniones en la **app de escritorio** de Teams/Zoom (solo navegador).
- Captura **sin selector** (eso requeriría una extensión de Chrome aparte).
- App móvil nativa.
- Unirse a la reunión como bot.

---

## 6. Cómo encaja en una Lovable app (lo justo para confiar en que es factible)

- **Frontend:** componente React dentro de la app (Lovable lo genera). Usa APIs estándar del
  navegador (`getDisplayMedia`, `getUserMedia`, Web Audio API para mezclar).
- **Backend:** una **Edge Function de Supabase** que actúa de *proxy* a Deepgram. La clave de
  Deepgram vive en los *secrets* de Supabase, nunca en el navegador. Es el patrón de seguridad
  que la propia Lovable recomienda.
- **Base de datos:** tabla de transcripciones (usuario, título, fecha, contenido) que Lovable
  crea sobre Supabase/Postgres como cualquier otra entidad de la app.
- **IA del acta:** otra Edge Function que manda la transcripción a un LLM (Claude / GPT) con la
  plantilla y devuelve el acta.

Nada de esto sale del stack nativo de Lovable. La feature es "más de lo mismo" para Lovable:
componentes React + Edge Functions + una tabla.

---

## 7. Esfuerzo estimado (orden de magnitud, no compromiso)

| Pieza | Esfuerzo | Quién lo hace |
|---|---|---|
| Componente de captura + mezcla de audio | Medio (la parte más delicada) | Prompts a Lovable + ajuste manual |
| Edge Function proxy a Deepgram | Bajo | 1 función, prompt + secret |
| Tabla + historial de transcripciones | Bajo | Lovable casi solo |
| UI de transcripción en vivo | Bajo-medio | Prompts a Lovable |
| "Generar acta" con IA (v1.1) | Bajo | 1 Edge Function + plantillas |

En conjunto: una v1 funcional es cuestión de **un puñado de prompts a Lovable + un par de Edge
Functions**, no de meses. La parte que más cuidado pide es la captura/mezcla de audio (timing
del selector, permisos, mezcla de dos streams).

---

## 8. Riesgos y avisos honestos

1. **El selector de pantalla es fricción visible.** Es el mayor compromiso de UX de esta
   arquitectura. Hay que comunicárselo bien al usuario (un mini-tutorial la primera vez).
2. **El usuario debe marcar "compartir audio del sistema".** Si no lo marca, capturas solo
   imagen sin sonido de la reunión. Hay que detectar y avisar si falta el audio.
3. **Eco entre micro y audio de la pestaña.** Al mezclar, puede haber eco; Deepgram lo maneja
   bastante bien, pero conviene desactivar la cancelación de eco del micro al grabar.
4. **Privacidad.** Estás grabando reuniones y mandando audio a Deepgram. Necesitas aviso claro
   al usuario y política de privacidad, sobre todo si la app es para terceros.
5. **El PRD para Lovable se alimenta distinto.** Lovable trabaja con *intención en lenguaje
   natural*, no con código pegado. Este documento sirve para **decidir**; para construir, se
   traduce a un PRD corto en la sección "Knowledge" de Lovable + prompts pequeños e
   incrementales (no un volcado de código).

---

## 9. Recomendación

Sí vale la pena, **con expectativas claras**: dentro de Lovable consigues la feature rápido,
en una sola base de código, reutilizando Supabase como backend seguro. El precio es el
selector de pantalla en cada reunión — aceptable para una v1 y para validar si la gente
realmente usa la feature.

**Orden sugerido:**
1. **v1 captura + transcripción guardada** (con selector). Valida el flujo completo.
2. **v1.1 acta con IA + notas + plantillas.** Aquí está el valor diferencial (lo de Granola).
3. **Solo si la fluidez se vuelve crítica:** evaluar una extensión de Chrome **aparte** que
   reutilice esta misma app como dashboard/backend. Documento separado:
   [extension-chrome-tabcapture.md](extension-chrome-tabcapture.md).

---

## Apéndice — relación con los otros documentos

- [../ROADMAP.md](../ROADMAP.md): visión general del producto por fases.
- [extension-chrome-tabcapture.md](extension-chrome-tabcapture.md): la ruta de extensión de
  Chrome (captura sin selector, multiplataforma) — relevante solo si se va por la opción B.
- [captura-nativa-windows.md](captura-nativa-windows.md): captura de escritorio (WASAPI) —
  relevante solo si se necesita la app de escritorio de Teams. Fuera de alcance de este PRD.
