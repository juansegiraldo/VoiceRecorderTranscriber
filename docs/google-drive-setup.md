# Configurar Google Drive en Voice Transcriber (guía paso a paso)

> Esto se hace **una sola vez**. Al terminar, en la app aparecerá el botón
> "🔑 Iniciar sesión con Google" y el flujo será: pulsas → sale el pop-up
> "elige tu cuenta" → entras → eliges tu audio → transcribe.

## ¿Qué es el "client secret" y por qué hace falta?

El `GOOGLE_CLIENT_SECRET` **no son tus datos personales ni tu contraseña**. Es el
"DNI" de *esta app* ante Google — identifica a la aplicación, no a ti. Toda app
que muestre el pop-up de "iniciar sesión con Google" tiene uno por detrás; en las
apps que usas a diario ya viene puesto por su desarrollador y tú no lo ves.

- Se guarda **en el servidor** (panel de Secrets de Streamlit Cloud) o en tu `.env`
  local. **Nunca** en el navegador ni en código público. Nadie más lo ve.
- Tú, como usuario, solo pulsas el botón y usas el pop-up. No tocas ningún secret.

---

## Paso 1 — Crear proyecto y habilitar la API

1. Entra en [Google Cloud Console](https://console.cloud.google.com/).
2. Arriba, crea un proyecto nuevo (p. ej. "VoiceTranscriber") o usa uno existente.
3. Ve a **APIs y servicios → Biblioteca**, busca **"Google Drive API"** y pulsa
   **Habilitar**.

## Paso 2 — Pantalla de consentimiento (modo prueba)

1. **APIs y servicios → Pantalla de consentimiento de OAuth**.
2. Tipo de usuario: **Externo** → Crear.
3. Rellena lo mínimo: nombre de la app ("Voice Transcriber"), tu correo de
   soporte y tu correo de contacto. Guarda y continúa.
4. En **Usuarios de prueba**, añade **tu propia dirección de Gmail** (y las de
   quien vaya a usar la app; hasta ~100). Guarda.
   - Déjalo en estado **"Prueba/Testing"**. No necesitas verificación de Google.

> ⚠️ En modo prueba, la sesión de Google caduca cada ~7 días: cada tanto tendrás
> que volver a pulsar el botón e iniciar sesión. Es normal para uso personal.

## Paso 3 — Crear la credencial OAuth

1. **APIs y servicios → Credenciales → Crear credenciales → ID de cliente de OAuth**.
2. Tipo de aplicación: **Aplicación web**.
3. En **URIs de redireccionamiento autorizados**, añade **las dos** (exactamente,
   con la barra final `/`):
   - `http://localhost:8501/`  ← para pruebas en tu PC
   - `https://TU-APP.streamlit.app/`  ← la URL real de tu app en Streamlit Cloud
     (cámbiala por la tuya cuando la sepas)
4. Pulsa **Crear**. Google te muestra el **ID de cliente** y el **Secreto de
   cliente**. Cópialos.

> El error más común es que esta URI **no coincida exacto** con la de la app
> (incluida la barra final). Si ves `redirect_uri_mismatch`, revisa este paso.

## Paso 4 — Poner los secrets

### En local (tu PC) — archivo `.env`
Añade estas tres líneas (ver `.env.example`):

```
GOOGLE_CLIENT_ID=el_id_de_cliente_que_copiaste
GOOGLE_CLIENT_SECRET=el_secreto_de_cliente_que_copiaste
GOOGLE_REDIRECT_URI=http://localhost:8501/
```

Arranca con `py -m streamlit run AppTranscribe.py` (puerto 8501) y prueba el botón.

### En Streamlit Cloud — panel de Secrets
En tu app desplegada: **⋮ → Settings → Secrets**, y pega (formato TOML):

```toml
DEEPGRAM_API_KEY = "tu_clave_deepgram"
OPENAI_API_KEY = "tu_clave_openai"
GOOGLE_CLIENT_ID = "el_id_de_cliente"
GOOGLE_CLIENT_SECRET = "el_secreto_de_cliente"
GOOGLE_REDIRECT_URI = "https://TU-APP.streamlit.app/"
```

Guarda; la app se reinicia sola y el botón de Drive queda activo.

---

## Cómo se usa (una vez configurado)

1. Abre la app → pestaña **☁️ Google Drive**.
2. Pulsa **🔑 Iniciar sesión con Google** → sale el pop-up, eliges tu cuenta.
3. La app lista tus audios recientes de Drive → eliges uno → **☁️ Cargar de Drive**.
4. (Opcional) recortas con el slider o los campos numéricos.
5. **🎤 Start Transcription**.

## Notas de seguridad

- La app solo pide permiso de **lectura** de tu Drive (`drive.readonly`): puede
  listar y descargar, nunca borrar ni modificar.
- Los secrets viven en el servidor; el navegador nunca los ve.
- El token de sesión se guarda solo en tu sesión de la app; cerrar sesión
  ("Salir") lo borra.
