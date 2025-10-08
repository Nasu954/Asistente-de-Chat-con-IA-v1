from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles 
from fastapi.responses import FileResponse 
import requests
import os
import json 
import uvicorn

app = FastAPI()

# serve frontend files 
# Asegúrate de que la carpeta 'static' exista en el mismo directorio que app.py
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ollama settings 
OLLAMA_URL = "http://localhost:11434/api/generate" # acá es donde vive ollama
MODEL_NAME = "mistral" # change to "llama3" if prefered 

@app.get("/")
def serve_homepage():
    """servirá el archivo index.html al acceder a la URL raiz"""
    # Verificamos que el archivo exista para evitar errores
    index_path = os.path.join("static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=500, detail="El archivo 'static/index.html' no fue encontrado.")
    return FileResponse(index_path)

@app.post("/chat")
def chat(prompt: str=Query(..., description="User prompt for AI model")):
    headers = {"Content-Type": "application/json"}
    
    # Validación básica del prompt
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="El prompt no puede estar vacío.")

    try:
        # send request to Ollama
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            headers=headers,
            # timeout=30  Añadir un timeout de 30 segundos
        )
        
        # --- Manejo del código de estado HTTP de Ollama (si está mal configurado) ---
        if response.status_code != 200:
            detail_msg = f"Ollama devolvió un error {response.status_code}. Asegúrate de que el modelo '{MODEL_NAME}' esté descargado."
            try:
                error_json = response.json()
                if 'error' in error_json:
                    detail_msg = f"Ollama Error: {error_json['error']}"
            except json.JSONDecodeError:
                pass
            
            raise HTTPException(status_code=502, detail=detail_msg)

        # --- Validación de la respuesta JSON ---
        response_data = response.text.strip()
        try:
            json_response = json.loads(response_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail=f"Respuesta JSON inválida de Ollama. Respuesta: {response_data[:100]}...")
            
        # Extract AI-generated response 
        ai_response = json_response.get("response")
        if not ai_response:
            raise HTTPException(status_code=500, detail="Ollama no devolvió el campo 'response' en el JSON.")
        
        return {"response": ai_response}
    
    except requests.exceptions.ConnectionError:
         # ESTA ES LA CORRECCIÓN CLAVE PARA DIAGNOSTICAR FALLOS DE OLLAMA
        raise HTTPException(
            status_code=503, 
            detail=f"ERROR DE CONEXIÓN: No se pudo conectar a Ollama en {OLLAMA_URL}. Asegúrate de que Ollama esté ejecutándose."
        )
    except requests.exceptions.RequestException as e: 
        raise HTTPException(status_code=500, detail=f"Solicitud a Ollama falló: {str(e)}")

# Run the API server 
if __name__ == "__main__": 
    # Asegura que el directorio estático exista antes de iniciar el servidor
    if not os.path.exists("static"):
        os.makedirs("static")
        print("Directorio 'static' creado.")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
