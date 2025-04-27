#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install langchain langchain-openai langchain-google-genai langchain-groq langchain-experimental langchain-neo4j neo4j pydantic python-dotenv ipywidgets openai tiktoken langchain-core langchain-community --upgrade')


# In[1]:


# -*- coding: utf-8 -*-
"""
Script para procesar documentos de texto, extraer estructuras de grafo
(nodos y relaciones) usando un LLM (OpenAI, Google, Groq, Ollama),
y guardar esta estructura en archivos JSON detallados.

Opcionalmente, puede cargar la estructura extraída en una base de datos Neo4j.

Diseñado para ejecución modular en Jupyter Notebooks (abrir este .py o usar %run).
"""

# ============================================================
# %% Bloque 1: Importaciones y Configuración Inicial
# ============================================================
# Nota: Si faltan librerías, ejecuta la siguiente línea en una celda separada UNA VEZ:
# !pip install langchain langchain-openai langchain-google-genai langchain-groq langchain-experimental langchain-neo4j neo4j pydantic python-dotenv ipywidgets openai tiktoken langchain-core langchain-community --upgrade

import os
import logging
import time
import webbrowser
import json # Para manejar JSON
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse # Para derivar URL del navegador
import traceback # Para logs de errores detallados

# Librerías de Terceros
import dotenv
from neo4j import GraphDatabase, Driver
from pydantic import BaseModel, Field, ValidationError # Pydantic v2
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser # Para robustez extra
# LLM Providers
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI # Para Gemini API
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
# Graph Components
from langchain_neo4j import Neo4jGraph # Aún necesario si se carga a Neo4j

# Configuración del Logging
log_format = '%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
# Configurar para que también salga en la consola del notebook
logging.basicConfig(level=logging.INFO, format=log_format, handlers=[
    logging.StreamHandler() # Añade salida a consola/stderr
    # Podrías añadir FileHandler si quieres guardar logs a archivo también
    # logging.FileHandler("graph_processor.log")
])
# Silenciar logs muy verbosos si es necesario
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
# Ajustar nivel de logs de Neo4j si es necesario (INFO, WARNING, ERROR)
# logging.getLogger("neo4j").setLevel(logging.WARNING)

logger = logging.getLogger(__name__) # Logger específico para este script

print("Bloque 1: Librerías importadas y logging configurado.")


# In[2]:


# ============================================================
# %% Bloque 2: Carga de Variables de Entorno y Parámetros
# ============================================================

logger.info("Bloque 2: Cargando variables de entorno y definiendo parámetros...")

# --- Carga de Variables de Entorno ---
# Busca .env en directorio actual y superiores
# Contenido esperado: NEO4J_..., OPENAI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY, etc.
env_path = dotenv.find_dotenv(usecwd=True) # Busca primero en el directorio actual
if env_path:
    loaded = dotenv.load_dotenv(dotenv_path=env_path, override=True)
    logger.info(f"Variables .env cargadas desde: {env_path}" if loaded else f".env encontrado ({env_path}) pero carga falló.")
else:
    logger.warning(".env no encontrado en directorio actual o superiores. Usando variables de entorno del sistema o valores por defecto.")

# --- Credenciales y Configuraciones ---
# Asegúrate de que estas variables se carguen correctamente desde .env o el entorno
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE: Optional[str] = os.getenv("NEO4J_DATABASE", None) # None usa la DB por defecto ('neo4j')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Para Gemini API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Parámetros de Configuración del Script ---
# Lista de archivos de entrada a procesar (relativa al directorio donde se ejecuta el script/notebook)
input_filepaths: List[str] = [
    "datasets/Conciencia.md",
    "datasets/Microsoft.md",
    # Añade más archivos aquí
]
# Directorio para guardar los JSON (relativo al directorio de ejecución)
# Asegúrate de que este directorio exista o se pueda crear.
output_directory: str = "output_graphs"

# --- Configuración LLM Principal (Extracción de Grafo JSON) ---
llm_type: str = "groq" # Opciones: "openai", "google", "groq", "ollama"
# Ajusta los modelos según el llm_type elegido y tu disponibilidad/preferencia
openai_main_model_name: str = "gpt-4o-mini" # Modelo OpenAI
google_main_model_name: str = "gemini-1.5-flash-latest" # Modelo Google Gemini
groq_main_model_name: str = "llama3-8b-8192" # Modelo Groq (ej: llama3-70b-8192, mixtral-8x7b-32768)
ollama_main_model_name: str = "gemma3:27b"

# Modelos disponibles en ollama en mi configuración
#NAME                       ID              SIZE      MODIFIED     
#llama3.2-vision:latest     085a1fdae525    7.9 GB    47 hours ago    
#deepseek-coder:1.3b        3ddd2d3fc8d2    776 MB    4 days ago      
#codegemma:2b               926331004170    1.6 GB    4 days ago      
#qwen:4b                    d53d04290064    2.3 GB    4 days ago      
#llava:latest               8dd30f6b0cb1    4.7 GB    5 days ago      
#llama3.1:8b                46e0c10c039e    4.9 GB    2 weeks ago     
#llama3.2:3b                a80c4f17acd5    2.0 GB    2 weeks ago     
#nomic-embed-text:latest    0a109f422b47    274 MB    3 weeks ago     
#gemma3:27b                 a418f5838eaf    17 GB     3 weeks ago     
#gemma3:latest              a2af6cc3eb7f    3.3 GB    3 weeks ago     
#qwq:latest                 009cb3f08d74    19 GB     3 weeks ago     

# " # Modelo principal si llm_type es "ollama"

# --- Configuración LLM para Chunking (SIEMPRE Ollama + JSON Mode) ---
# Se usa Ollama para chunking por la fiabilidad de `format="json"`.
# Asegúrate de que Ollama esté corriendo en OLLAMA_BASE_URL y tenga este modelo.
ollama_chunking_model_name: str = "gemma3:27b" # Modelo Ollama específico para chunking JSON
# Ejecuta `ollama pull gemma:7b` en tu servidor Ollama si no lo tienes.

# --- Control del Flujo ---
# Define qué partes del script se ejecutarán
skip_extraction: bool = False      # True para saltar lectura, chunking y extracción JSON
print_chunks: bool = False         # True para imprimir chunks intermedios de Ollama
visualize_json: bool = True        # True para imprimir el JSON extraído final en la salida
load_into_neo4j: bool = True       # True para intentar cargar el JSON extraído a Neo4j
# --- >>> LÍNEA AÑADIDA/CORREGIDA <<< ---
run_interactive_query: bool = True # True para iniciar el bucle de consultas interactivas (si load_into_neo4j=True)
# ------------------------------------

# --- Opciones Neo4j (SOLO si load_into_neo4j = True) ---
# Estas opciones solo tienen efecto si load_into_neo4j es True
clear_graph_before_load: bool = True # Borrar grafo Neo4j ANTES de cargar los datos
delete_indexes_before_load: bool = True # Borrar índices Neo4j definidos por usuario ANTES de cargar

# --- Opciones de Visualización Externa ---
# Esta opción solo tiene efecto si load_into_neo4j es True
show_neo4j_browser: bool = True    # Intentar abrir Neo4j Browser después de la carga

# --- Verificación de Credenciales Críticas ---
# Comprueba si hay contraseña para Neo4j, necesaria si se carga/consulta
neo4j_configured = bool(NEO4J_PASSWORD)
if load_into_neo4j and not neo4j_configured:
    logger.warning("load_into_neo4j=True pero falta NEO4J_PASSWORD. La carga a Neo4j fallará.")
# Comprueba las API keys según el LLM principal seleccionado
if llm_type == "openai" and not OPENAI_API_KEY: logger.warning("llm_type='openai' pero OPENAI_API_KEY falta.")
if llm_type == "google" and not GOOGLE_API_KEY: logger.warning("llm_type='google' pero GOOGLE_API_KEY falta.")
if llm_type == "groq" and not GROQ_API_KEY: logger.warning("llm_type='groq' pero GROQ_API_KEY falta.")
# No se necesita API key para Ollama local si es el principal

# Crear directorio de salida si no existe y no se salta la extracción
if not skip_extraction:
    try:
        # Crear el directorio si no existe. parents=True crea directorios intermedios si son necesarios.
        os.makedirs(output_directory, exist_ok=True)
        logger.info(f"Directorio de salida para JSON asegurado: '{output_directory}'")
    except OSError as e:
        logger.error(f"No se pudo crear el directorio de salida '{output_directory}': {e}. La escritura de JSON fallará.")
        # Considerar detener el script aquí si la salida es esencial
        # raise OSError(f"Fallo al crear directorio de salida: {e}")

# --- Imprimir Configuración Final ---
# Muestra todos los parámetros tal como se usarán
print("\n--- Configuración Cargada y Verificada ---")
print(f"Neo4j URI: {NEO4J_URI}")
print(f"Neo4j User: {NEO4J_USERNAME}")
print(f"Neo4j Password Cargada: {'Sí' if NEO4J_PASSWORD else 'No'}")
print(f"Neo4j Database: {NEO4J_DATABASE if NEO4J_DATABASE else 'default'}")
print("-" * 20)
print(f"LLM Principal (Extracción JSON): {llm_type}")
if llm_type == "openai": print(f"  Modelo Principal OpenAI: {openai_main_model_name}")
elif llm_type == "google": print(f"  Modelo Principal Google: {google_main_model_name}")
elif llm_type == "groq": print(f"  Modelo Principal Groq: {groq_main_model_name}")
elif llm_type == "ollama": print(f"  Modelo Principal Ollama: {ollama_main_model_name}")
# Indicar qué API keys se encontraron
if OPENAI_API_KEY: print(f"  OpenAI API Key Cargada: Sí")
if GOOGLE_API_KEY: print(f"  Google API Key Cargada: Sí")
if GROQ_API_KEY: print(f"  Groq API Key Cargada: Sí")
print("-" * 20)
print(f"LLM para Chunking (JSON): Ollama (Siempre)")
print(f"  Modelo Chunking Ollama: {ollama_chunking_model_name}")
print(f"  URL Base Ollama (Chunking): {OLLAMA_BASE_URL}")
print("-" * 20)
print(f"Archivos a procesar: {', '.join(input_filepaths)}")
print(f"Directorio de Salida JSON: {output_directory}")
print(f"Saltar Extracción JSON: {skip_extraction}")
print(f"Imprimir Chunks: {print_chunks}")
print(f"Visualizar JSON Extraído: {visualize_json}")
print(f"Ejecutar Consultas Interactivas: {run_interactive_query}") # <-- Mostrar valor
print("-" * 20)
print(f"Cargar a Neo4j (Opcional): {load_into_neo4j}")
if load_into_neo4j:
    print(f"  Neo4j Configurado: {'Sí' if neo4j_configured else 'NO (Falta Password!)'}")
    print(f"  Borrar Grafo Neo4j antes de Cargar: {clear_graph_before_load}")
    print(f"  Borrar Índices Neo4j antes de Cargar: {delete_indexes_before_load}")
    print(f"  Abrir Neo4j Browser (post-carga): {show_neo4j_browser}") # <-- Mostrar valor
print("-----------------------------")
logger.info("Bloque 2: Parámetros definidos y verificados.")


# In[3]:


# ============================================================
# %% Bloque 3: Definición de Modelos Pydantic
# ============================================================
logger.info("Bloque 3: Definiendo modelos Pydantic para estructura de datos...")
# Usamos Pydantic V2 (importado como pydantic sin .v1)

class Chunk(BaseModel):
    """Representa un trozo de texto semánticamente coherente."""
    chunk_id: int = Field(..., description="ID numérico secuencial único del chunk (empezando en 1).")
    text: str = Field(..., description="Contenido textual del chunk, procesado semánticamente.")

class Chunks(BaseModel):
    """Esquema JSON esperado como salida del LLM de chunking (Ollama). Contiene lista de Chunks."""
    chunks: List[Chunk] = Field(..., description="Lista ordenada de los chunks generados del texto original.")

# --- Modelos para Extracción de Grafo (Salida del LLM Principal) ---
class Node(BaseModel):
    """Representa un nodo en el grafo de conocimiento."""
    id: str = Field(..., description="Identificador único y canónico del nodo (e.g., nombre normalizado).")
    label: str = Field(..., description="Etiqueta principal del nodo (e.g., Person, Organization, Concept).")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Propiedades adicionales (clave-valor) extraídas del texto.")

class Relationship(BaseModel):
    """Representa una relación dirigida entre dos nodos identificados por su ID."""
    source: str = Field(..., description="ID del nodo origen.")
    target: str = Field(..., description="ID del nodo destino.")
    type: str = Field(..., description="Tipo de la relación en MAYUSCULAS_CON_GUIONES (e.g., WORKS_AT, LOCATED_IN).")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Propiedades de la relación (clave-valor).")

class ExtractedGraph(BaseModel):
    """Representa la estructura completa del grafo extraída de un texto o chunk."""
    nodes: List[Node] = Field(default_factory=list, description="Lista de nodos únicos identificados.")
    relationships: List[Relationship] = Field(default_factory=list, description="Lista de relaciones identificadas entre los nodos.")

print("Modelos Pydantic definidos: Chunk, Chunks, Node, Relationship, ExtractedGraph.")
logger.info("Bloque 3: Modelos Pydantic completados.")


# In[4]:


# ============================================================
# %% Bloque 4: Inicialización del LLM Principal
# ============================================================
logger.info("Bloque 4: Inicializando el LLM principal según configuración...")

def get_main_llm(
    llm_provider: str,
    # OpenAI
    openai_key: Optional[str], openai_model: str,
    # Google
    google_key: Optional[str], google_model: str,
    # Groq
    groq_key: Optional[str], groq_model: str,
    # Ollama
    ollama_url: str, ollama_model: str
) -> Optional[BaseChatModel]:
    """
    Inicializa y devuelve la instancia del LLM principal (para extracción JSON).
    Maneja OpenAI, Google Gemini, Groq, y Ollama.
    """
    logger.info(f"Intentando inicializar LLM principal: {llm_provider}")
    llm_instance: Optional[BaseChatModel] = None
    try:
        if llm_provider == "openai":
            if not openai_key: raise ValueError("OPENAI_API_KEY es requerida.")
            llm_instance = ChatOpenAI(model=openai_model, temperature=0, api_key=openai_key)
        elif llm_provider == "google":
            if not google_key: raise ValueError("GOOGLE_API_KEY es requerida.")
            llm_instance = ChatGoogleGenerativeAI(model=google_model, google_api_key=google_key, temperature=0, convert_system_message_to_human=True)
        elif llm_provider == "groq":
            if not groq_key: raise ValueError("GROQ_API_KEY es requerida.")
            llm_instance = ChatGroq(groq_api_key=groq_key, model_name=groq_model, temperature=0)
        elif llm_provider == "ollama":
            if not ollama_model: raise ValueError("ollama_main_model_name es requerido.")
            # Asegurarse que la URL base sea correcta
            if not ollama_url or not urlparse(ollama_url).scheme:
                 raise ValueError(f"URL base de Ollama inválida o faltante: '{ollama_url}'")
            llm_instance = ChatOllama(model=ollama_model, base_url=ollama_url, temperature=0)
        else:
            logger.error(f"Tipo de LLM principal no soportado: '{llm_provider}'")
            return None

        # Determinar nombre del modelo para log
        model_name_for_log = 'N/A'
        if hasattr(llm_instance, 'model_name'):
            model_name_for_log = llm_instance.model_name
        elif hasattr(llm_instance, 'model'): # Ollama usa 'model'
             model_name_for_log = llm_instance.model
        elif llm_provider == "google": # Google puede no tenerlo directamente
             model_name_for_log = google_model # Usar el nombre pasado

        logger.info(f"LLM principal inicializado: {llm_provider} (Modelo: {model_name_for_log})")


        # Opcional: Test rápido de conectividad
        # logger.debug(f"Realizando test rápido LLM {llm_provider}...")
        # llm_instance.invoke("Confirma status.")
        # logger.info(f"Test rápido LLM {llm_provider} exitoso.")

        return llm_instance

    except ImportError as e:
         logger.error(f"Fallo import librería para {llm_provider}: {e}. ¿Instalada?", exc_info=False) # No mostrar traceback completo de import
         print(f"ERROR: Librería para {llm_provider} no encontrada. Instálala (ver Bloque 1).")
         return None
    except ValueError as e: # Captura errores de validación (ej. falta de clave/URL)
         logger.error(f"Error configuración {llm_provider}: {e}")
         print(f"ERROR: Configuración para {llm_provider} inválida: {e}")
         return None
    except Exception as e: # Otros errores (conexión, API inválida, modelo no existe)
        logger.error(f"Fallo inesperado inicializando {llm_provider}: {e}", exc_info=True) # Mostrar traceback aquí sí
        print(f"ERROR: Fallo al inicializar {llm_provider}. Verifica conexión, API keys, nombre del modelo y logs.")
        return None

# --- Inicializar el LLM principal ---
llm: Optional[BaseChatModel] = get_main_llm(
    llm_provider=llm_type,
    openai_key=OPENAI_API_KEY, openai_model=openai_main_model_name,
    google_key=GOOGLE_API_KEY, google_model=google_main_model_name,
    groq_key=GROQ_API_KEY, groq_model=groq_main_model_name,
    ollama_url=OLLAMA_BASE_URL, ollama_model=ollama_main_model_name
)

if llm:
    print(f"LLM principal ({llm_type}) inicializado correctamente.")
else:
    print(f"¡FALLO al inicializar el LLM principal ({llm_type})! La extracción JSON no funcionará.")
    # Detener si el LLM es esencial para el flujo deseado
    if not skip_extraction:
         raise RuntimeError(f"Fallo crítico: No se pudo inicializar el LLM principal '{llm_type}'.")

logger.info("Bloque 4: Inicialización LLM principal completada (o fallida).")


# In[5]:


# ============================================================
# %% Bloque 5: Funciones Utilitarias para Neo4j (Opcional)
# ============================================================
logger.info("Bloque 5: Definiendo funciones utilitarias para Neo4j (usadas si load_into_neo4j=True)...")

# Asegurar importaciones necesarias aquí también
from neo4j import GraphDatabase, Driver
from typing import Optional
import logging
import time
import webbrowser
from urllib.parse import urlparse

# Usar el logger principal configurado
logger = logging.getLogger(__name__)

def get_neo4j_driver(uri: str, user: str, passw: str) -> Optional[Driver]:
    """Establece y verifica conexión directa con Neo4j."""
    if not passw:
        logger.error("Se requiere contraseña de Neo4j para crear el driver.")
        print("ERROR: Falta contraseña Neo4j.")
        return None
    try:
        # Timeout de conexión y adquisición más generosos
        driver = GraphDatabase.driver(uri, auth=(user, passw),
                                      connection_timeout=15.0, # segundos
                                      max_connection_lifetime=3600) # 1 hora
        # Verificar conectividad y obtener info básica del servidor
        driver.verify_connectivity()
        server_info = driver.get_server_info()
        # --- CORRECCIÓN: Log simplificado sin .version ---
        # Acceder a atributos que sí existen en ServerInfo
        logger.info(f"Conexión directa con Neo4j en {uri} verificada (Server address: {server_info.address}, Protocol: {server_info.protocol_version}).")
        # También podrías loguear server_info.agent si es útil
        # logger.info(f"Server Agent: {server_info.agent}")
        # ------------------------------------------------
        return driver
    except Exception as e:
        logger.error(f"Fallo al conectar driver Neo4j en {uri}: {e}", exc_info=True)
        print(f"ERROR: Fallo al conectar con Neo4j en {uri}. Verifica URI, credenciales y estado del servidor.")
        return None

def reset_graph_data(driver: Driver, db_name: Optional[str] = None) -> bool:
    """Elimina TODOS los nodos y relaciones. Requiere doble confirmación."""
    effective_db = db_name if db_name else "neo4j" # Neo4j >= 4.0 default es 'neo4j'
    # Usar input() solo si se ejecuta interactivamente
    try:
        print(f"\nADVERTENCIA MUY SERIA:")
        print(f"Estás a punto de BORRAR **TODOS** los nodos y relaciones de la base de datos '{effective_db}'.")
        print(f"Esta acción es IRREVERSIBLE.")
        confirm1 = input(f"Escribe 'SI QUIERO BORRAR TODO' para continuar con la segunda confirmación: ")
        if confirm1 != "SI QUIERO BORRAR TODO":
             logger.warning(f"Primera confirmación para borrado cancelada por el usuario para BD '{effective_db}'.")
             print("Borrado cancelado (Paso 1).")
             return False

        print(f"\nSEGUNDA CONFIRMACIÓN (IRREVERSIBLE):")
        confirm2 = input(f"Escribe 'BORRAR TODO NEO4J AHORA' para proceder con el borrado de '{effective_db}': ")
        if confirm2 != "BORRAR TODO NEO4J AHORA":
            logger.warning(f"Segunda confirmación para borrado cancelada por el usuario para BD '{effective_db}'.")
            print("Borrado cancelado (Paso 2).")
            return False

    except EOFError: # Manejar si no se ejecuta en un TTY interactivo
         logger.error("No se pudo obtener confirmación interactiva para reset_graph_data. Abortando borrado.")
         print("ERROR: No se puede confirmar el borrado en un entorno no interactivo. Operación cancelada.")
         return False

    logger.warning(f"CONFIRMACIÓN DOBLE RECIBIDA. PROCEDIENDO CON BORRADO COMPLETO en BD '{effective_db}'...")
    try:
        with driver.session(database=db_name) as session:
            start_time = time.time()
            logger.info("Ejecutando 'MATCH (n) DETACH DELETE n'...")
            # Ejecutar con timeout por si la base de datos es muy grande
            result = session.run("MATCH (n) DETACH DELETE n", timeout=300.0) # 5 minutos timeout
            summary = result.consume() # Consumir para obtener estadísticas
            duration = time.time() - start_time
            # Acceder a los contadores del summary
            nodes_deleted = summary.counters.nodes_deleted
            rels_deleted = summary.counters.relationships_deleted
            logger.info(f"Datos del grafo reseteados en BD '{effective_db}' en {duration:.2f}s. Nodos borrados: {nodes_deleted}, Relaciones borradas: {rels_deleted}.")
            print(f"Datos del grafo en '{effective_db}' borrados.")
            return True
    except Exception as e:
        logger.error(f"Error al resetear datos en BD '{effective_db}': {e}", exc_info=True)
        print(f"Error al borrar datos en '{effective_db}'. Ver logs.")
        return False

def retrieve_graph_summary(driver: Driver, db_name: Optional[str] = None) -> str:
    """Recupera un resumen del contenido del grafo (conteos, labels, reltypes)."""
    db_log_name = db_name if db_name else 'default'
    summary = f"Resumen del Grafo (Base de Datos: {db_log_name}):\n"
    try:
        with driver.session(database=db_name) as session:
            # Intentar con APOC primero
            apoc_summary = ""
            try:
                # Usar single() y verificar si devuelve None o un registro
                stats_record = session.run("CALL apoc.meta.stats() YIELD nodeCount, relCount, labels, relTypes RETURN *").single()
                if stats_record: # Verificar que no sea None
                    stats = stats_record.data() # Convertir a diccionario
                    apoc_summary += f"- Nodos Totales (APOC): {stats.get('nodeCount', 0)}\n" # Usar .get con default
                    apoc_summary += f"- Relaciones Totales (APOC): {stats.get('relCount', 0)}\n"
                    # Filtrar tipos de relaciones internas de APOC
                    rel_types_map = stats.get('relTypes', {})
                    rel_types_filtered = {k: v for k, v in rel_types_map.items() if not k.startswith('_')}
                    labels_map = stats.get('labels', {})
                    apoc_summary += f"- Tipos de Nodos (Labels): {', '.join(sorted(labels_map.keys())) if labels_map else 'Ninguno'}\n"
                    apoc_summary += f"- Tipos de Relaciones: {', '.join(sorted(rel_types_filtered.keys())) if rel_types_filtered else 'Ninguna'}\n"
                    return apoc_summary.strip() # Devolver si APOC tuvo éxito
                else:
                    logger.debug("apoc.meta.stats() no devolvió resultados (single() fue None).")
            except Exception as apoc_e:
                logger.debug(f"apoc.meta.stats() falló ({type(apoc_e).__name__}), usando conteos manuales.")

            # Conteos manuales como fallback
            manual_summary = ""
            node_count_res = session.run("MATCH (n) RETURN count(n) AS count").single()
            node_count = node_count_res['count'] if node_count_res else 0
            manual_summary += f"- Nodos Totales (Manual): {node_count}\n"
            if node_count == 0: return manual_summary + "- Grafo vacío.\n"

            rel_count_res = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()
            rel_count = rel_count_res['count'] if rel_count_res else 0
            manual_summary += f"- Relaciones Totales (Manual): {rel_count}\n"

            labels_res = session.run("CALL db.labels() YIELD label RETURN collect(label) as labels").single()
            labels = sorted(labels_res['labels']) if labels_res and labels_res.get('labels') else []
            manual_summary += f"- Tipos de Nodos (Labels): {', '.join(labels) if labels else 'Ninguno'}\n"

            rel_types_res = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types").single()
            rel_types = sorted(rel_types_res['types']) if rel_types_res and rel_types_res.get('types') else []
            # Filtrar tipos internos (puede ser redundante si APOC falló, pero seguro)
            rel_types = [rt for rt in rel_types if not rt.startswith("_")]
            manual_summary += f"- Tipos de Relaciones: {', '.join(rel_types) if rel_types else 'Ninguna'}\n"
            return manual_summary.strip()

    except Exception as e:
        logger.error(f"Error recuperando resumen del grafo BD '{db_log_name}': {e}", exc_info=True)
        return summary + f"Error recuperando resumen: {e}"

def print_indexes(driver: Driver, db_name: Optional[str] = None):
    """Imprime los índices existentes en la base de datos de forma detallada."""
    effective_db = db_name if db_name else "default"
    logger.info(f"Consultando índices en BD '{effective_db}'...")
    print(f"\n--- Índices en Base de Datos: {effective_db} ---")
    try:
        with driver.session(database=db_name) as session:
            result = session.run("SHOW INDEXES")
            indexes = [record.data() for record in result] # Convertir a lista
            if indexes:
                indexes = sorted(indexes, key=lambda x: x.get('name', '')) # Ordenar por nombre
                print(f"Se encontraron {len(indexes)} índices:")
                for idx in indexes:
                    name = idx.get('name', 'N/A')
                    idx_type = idx.get('type', 'N/A')
                    entity_type = idx.get('entityType', 'N/A')
                    labels = idx.get('labelsOrTypes', []) # Default a lista vacía
                    props = idx.get('properties', [])   # Default a lista vacía
                    state = idx.get('state', 'N/A')
                    # Usar join solo si la lista no está vacía/None
                    labels_str = ', '.join(labels) if labels else 'N/A'
                    props_str = ', '.join(props) if props else 'N/A'
                    print(f"- Nombre: {name} | Estado: {state} | TipoÍndice: {idx_type}")
                    print(f"    TipoEntidad: {entity_type} | Labels/Tipos: {labels_str} | Propiedades: {props_str}")
                logger.info(f"Se encontraron {len(indexes)} índices en BD '{effective_db}'.")
            else:
                print("No se encontraron índices definidos.")
                logger.info(f"No se encontraron índices en BD '{effective_db}'.")
    except Exception as e:
        # Manejo de fallback para versiones antiguas (igual que antes)
        if "Unknown command `SHOW`" in str(e) or "Invalid input 'SHOW'" in str(e):
             print("Comando 'SHOW INDEXES' no soportado. Usando CALL db.indexes()...")
             try:
                 with driver.session(database=db_name) as session_old:
                     result_old = session_old.run("CALL db.indexes()")
                     indexes_old = [record_old.data() for record_old in result_old]
                     if indexes_old:
                         print(f"Se encontraron {len(indexes_old)} índices (formato antiguo):")
                         # Formato puede variar, imprimir raw
                         for idx_old in indexes_old: print(f"  - {idx_old}")
                     else: print("No se encontraron índices (método alternativo).")
             except Exception as e_old: print(f"Error CALL db.indexes(): {e_old}")
        else:
            print(f"Error al obtener índices: {e}")
            logger.error(f"Error obteniendo índices BD '{effective_db}': {e}", exc_info=True)
    print("---------------------------------------")

def delete_all_user_indexes(driver: Driver, db_name: Optional[str] = None) -> bool:
    """Elimina TODOS los índices definidos por el usuario. Requiere confirmación."""
    effective_db = db_name if db_name else "default"
    logger.warning(f"Intentando eliminar índices de usuario en BD '{effective_db}'...")
    try:
        with driver.session(database=db_name) as session:
            # --- CONSULTA CYPHER CORREGIDA ---
            # Usar YIELD para obtener las columnas y luego filtrar con WHERE y RETURN
            cypher_get_indexes = """
            SHOW INDEXES
            YIELD name, type // Especificar las columnas que necesitas
            WHERE type <> 'LOOKUP' // Filtrar por tipo
              AND NOT name STARTS WITH 'vector_' // Exclusiones por nombre
              AND NOT name STARTS WITH 'graphrag_'
              AND NOT name CONTAINS '_vector_'
              AND NOT name IN $system_index_names // Excluir índices de sistema
            RETURN name AS name_to_drop // Retornar solo el nombre para borrar
            """
            # ---------------------------------------
            # Nombres comunes de índices de sistema/constraint a excluir
            params = {"system_index_names": ["nodes_id_unique", "rels_id_unique", "node_constraint", "relationship_constraint", "token_lookup_nodes", "token_lookup_relationships"]}

            index_names_to_delete = []
            try:
                 result = session.run(cypher_get_indexes, parameters=params)
                 index_names_to_delete = [record["name_to_drop"] for record in result]
                 logger.info(f"Se encontraron {len(index_names_to_delete)} índices de usuario para posible borrado.")
            except Exception as show_e:
                 # Manejo de fallback si SHOW INDEXES no funciona (igual que antes)
                 if "Unknown command `SHOW`" in str(show_e) or "Invalid input 'SHOW'" in str(show_e):
                     logger.warning("SHOW INDEXES falló/no soportado, intentando con db.indexes()...")
                     try:
                         # Este fallback es menos preciso para identificar índices de sistema
                         result_old = session.run("CALL db.indexes() YIELD name, type WHERE type <> 'LOOKUP' RETURN name")
                         index_names_to_delete = [r["name"] for r in result_old if not r["name"].startswith("constraint") and not r["name"].startswith("token") and not r["name"].startswith("vector")]
                         logger.info(f"Fallback db.indexes() encontró {len(index_names_to_delete)} candidatos.")
                     except Exception as dbidx_e:
                         logger.error(f"CALL db.indexes() también falló: {dbidx_e}. No se pueden determinar índices a borrar.")
                         return False # Fallo crítico
                 else:
                     logger.error(f"Error ejecutando SHOW INDEXES: {show_e}", exc_info=True)
                     print("Error listando índices, el borrado de índices fallará.")
                     return False # Indicar fallo

            # --- Lógica de confirmación y borrado ---
            if not index_names_to_delete:
                logger.info(f"No se encontraron índices de usuario para eliminar en BD '{effective_db}'.")
                print(f"No hay índices de usuario para borrar en '{effective_db}'.")
                return True

            print(f"\nSe borrarán los siguientes {len(index_names_to_delete)} índices de usuario en '{effective_db}':")
            for name in index_names_to_delete: print(f"  - {name}")
            # Input interactivo
            try:
                confirm = input(f"ADVERTENCIA: Esta acción puede impactar rendimiento. Escribe 'BORRAR INDICES USUARIO' para confirmar: ")
            except EOFError:
                 logger.error("No se pudo obtener confirmación interactiva para delete_all_user_indexes.")
                 print("ERROR: No se puede confirmar borrado en entorno no interactivo. Operación cancelada.")
                 return False

            if confirm != "BORRAR INDICES USUARIO":
                logger.warning(f"Borrado de índices cancelado por usuario para BD '{effective_db}'.")
                print("Borrado de índices cancelado.")
                return False

            logger.warning(f"PROCEDIENDO A BORRAR {len(index_names_to_delete)} índices en BD '{effective_db}'...")
            deleted_count = 0
            errors = []
            start_time = time.time()
            for index_name in index_names_to_delete:
                # Verificar si el índice todavía existe antes de intentar borrarlo
                check_exists_query = "SHOW INDEXES WHERE name = $index_name RETURN name"
                exists_result = session.run(check_exists_query, parameters={"index_name": index_name}).single()

                if exists_result:
                    try:
                        logger.debug(f"Ejecutando DROP INDEX `{index_name}`...")
                        summary = session.run(f"DROP INDEX `{index_name}`").consume()
                        logger.debug(f"Índice '{index_name}' eliminado de BD '{effective_db}'. Contadores: {summary.counters}")
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Error eliminando índice '{index_name}' BD '{effective_db}': {e}", exc_info=True)
                        errors.append(index_name)
                else:
                    logger.debug(f"Índice '{index_name}' no encontrado, omitiendo borrado.")
                    # Contar como procesado aunque no existiera
                    deleted_count += 1


            duration = time.time() - start_time
            if not errors:
                logger.info(f"{deleted_count} índices procesados para borrado en {duration:.2f}s de BD '{effective_db}'.")
                print(f"Se procesó el borrado de {deleted_count} índices de usuario de '{effective_db}'.")
                return True
            else:
                logger.error(f"Fallo al eliminar índices en BD '{effective_db}': {', '.join(errors)}")
                print(f"Error al borrar índices en '{effective_db}'. Ver logs.")
                return False
    except Exception as e:
        logger.error(f"Error durante eliminación de índices BD '{effective_db}': {e}", exc_info=True)
        print(f"Error general al procesar índices en '{effective_db}'. Ver logs.")
        return False

def display_neo4j_browser(browser_url: str = 'http://localhost:7474/browser/'):
    """Intenta abrir el navegador web en la URL de Neo4j Browser."""
    try:
        is_opened = webbrowser.open(browser_url, new=2) # new=2 intenta nueva pestaña
        if is_opened:
             logger.info(f"Navegador abierto (o intentado) en: {browser_url}")
             print(f"Se intentó abrir el navegador en {browser_url}")
        else:
             # Esto puede pasar si no hay navegador gráfico o por permisos
             logger.warning(f"webbrowser.open devolvió False para {browser_url}. Puede que no se haya abierto.")
             print(f"No se pudo confirmar apertura automática del navegador en {browser_url}. Abre la URL manualmente si es necesario.")
        return is_opened
    except Exception as e:
        logger.error(f"No se pudo abrir navegador para Neo4j Browser: {e}", exc_info=True)
        print(f"Error al intentar abrir el navegador: {e}")
        return False

print("Funciones utilitarias de Neo4j definidas.")
logger.info("Bloque 5: Funciones utilitarias de Neo4j completadas.")


# In[6]:


# ============================================================
# %% Bloque 6: Definición de Prompts
# ============================================================
logger.info("Bloque 6: Definiendo plantillas de prompts...")

# --- Plantilla para Chunking Semántico (Usada por Ollama JSON) ---
prompt_chunking_semantic_template = PromptTemplate.from_template(
    template="""Eres un experto en NLP. Divide el texto en chunks semánticamente coherentes y autocontenidos. Sigue estas reglas ESTRICTAMENTE:
1. Descomposición Proposicional: Divide en las proposiciones (ideas) más simples posibles.
2. Oraciones Simples: Convierte oraciones complejas en varias simples. Mantén redacción original si puedes.
3. Separación de Descripciones: Info descriptiva de entidades -> chunk propio.
4. Descontextualización: Reemplaza pronombres (él, ella, esto) con nombres completos. Añade contexto si es necesario para que el chunk se entienda solo.
5. Coherencia: Cada chunk debe ser gramaticalmente correcto.
6. IDs: Asigna ID numérico secuencial a cada chunk (desde 1).

Texto de Entrada:
--------------------
{input}
--------------------

Instrucción Final: Prepara tu respuesta como JSON siguiendo el esquema [{{"chunk_id": int, "text": string}}]. El sistema forzará este formato, tú genera el contenido correcto."""
)

# --- Plantilla para Extracción de Grafo (Usada por LLM Principal) ---
# (Usa el definido en el Bloque 3 para ExtractedGraph)
prompt_graph_extraction_template = PromptTemplate.from_template(
    template="""Eres un experto analista de texto y modelador de grafos de conocimiento. Tu tarea es leer el texto proporcionado y extraer una estructura de grafo significativa.

**Instrucciones Detalladas:**

1.  **Identifica Nodos:**
    *   Extrae las entidades clave (Personas, Organizaciones, Lugares, Proyectos, Conceptos técnicos, Productos, Fechas importantes, etc.).
    *   Para cada nodo, asigna:
        *   `id`: Un identificador **único y canónico**. Usa el nombre completo o término más representativo y normalizado (e.g., "Microsoft Corp." en lugar de "Microsoft" o "MSFT"). Sé consistente.
        *   `label`: Una etiqueta descriptiva y concisa en formato PascalCase (e.g., "Person", "Organization", "Concept", "Location", "Project", "Product", "Date", "Technology").
        *   `properties`: Un diccionario de propiedades adicionales extraídas **directamente** del texto (e.g., para "Person": {{"title": "CEO"}}, para "Product": {{"version": "1.0"}}). Incluye solo propiedades explícitas. Usa claves en minúscula.

2.  **Identifica Relaciones:**
    *   Extrae las relaciones significativas *entre los nodos que identificaste*.
    *   Para cada relación, asigna:
        *   `source`: El `id` del nodo origen (debe coincidir con un `id` de nodo extraído).
        *   `target`: El `id` del nodo destino (debe coincidir con un `id` de nodo extraído).
        *   `type`: Un tipo de relación claro en formato VERBO_EN_MAYUSCULAS (e.g., "WORKS_AT", "LOCATED_IN", "MENTIONS", "DISCUSSES", "PART_OF", "HAS_FEATURE", "ANNOUNCED_ON", "RELATED_TO").
        *   `properties`: Un diccionario de propiedades de la relación, si las hay (e.g., {{"role": "Developer"}}, {{"date": "2024-01-15"}}).

3.  **Contexto y Precisión:** Basa la extracción *estrictamente* en el texto proporcionado. No inventes nodos, relaciones o propiedades. Si una entidad o relación no es clara, es mejor omitirla.
4.  **Formato de Salida:** Estructura tu respuesta **exactamente** como un objeto JSON que cumpla con el esquema Pydantic proporcionado. Asegúrate de que el JSON sea válido, incluyendo comillas dobles correctas y comas donde sea necesario.

**Texto a Analizar:**
--------------------
{chunk_text}
--------------------

**Esquema JSON Requerido (Sigue esta estructura):**
```json
{{
  "nodes": [
    {{
      "id": "id_canonico_nodo_1",
      "label": "EtiquetaPascalCase",
      "properties": {{ "propiedad_minuscula": "valor" }}
    }},
    ...
  ],
  "relationships": [
    {{
      "source": "id_nodo_origen",
      "target": "id_nodo_destino",
      "type": "TIPO_RELACION_MAYUSCULAS",
      "properties": {{ "prop_rel_minuscula": "valor_rel" }}
    }},
    ...
  ]
}}
{format_instructions}

Salida JSON (Solo el objeto JSON, nada más antes o después):
"""
)

#--- Plantilla para Consulta del Grafo (Usada si load_into_neo4j=True) ---

prompt_graph_query_template = PromptTemplate.from_template(
template="""Eres un asistente experto en grafos de conocimiento Neo4j. Responde preguntas basándote únicamente en el esquema y resumen del grafo proporcionados.

Instrucciones Clave:

    Usa Solo el Esquema/Resumen: No inventes información no presente o sugerida.

    Interpreta el Esquema: Entiende qué tipos de nodos (Labels) y relaciones existen.

    Consulta el Resumen: Obtén una idea del contenido actual (nodos, relaciones).

    Si Falta Información: Indica claramente que no puedes responder con los datos dados (ej: "El esquema no contiene información sobre X..."). No adivines.

    Sé Conciso: Responde directamente a la pregunta.

Esquema y Resumen del Grafo:

Esquema:
{knowledge_graph_schema}
Resumen:
{knowledge_graph_summary}

Pregunta del Usuario: {question}

Respuesta:"""
)
#--- Selección Final de Prompts ---

prompt_chunking = prompt_chunking_semantic_template # Usado internamente por GraphProcessor para Ollama
prompt_extraction = prompt_graph_extraction_template # Usado por GraphProcessor para el LLM principal
prompt_query = prompt_graph_query_template # Para consultas opcionales a Neo4j

print(f"Prompt para chunking (Ollama JSON) definido.")
print(f"Prompt para extracción de grafo JSON (LLM Principal) definido.")
print(f"Prompt para consulta del grafo (Neo4j opcional) definido.")
logger.info("Bloque 6: Plantillas de prompts definidas.")


# In[7]:


# ============================================================
# %% Bloque 7: Definición de la Clase GraphProcessor
# ============================================================
logger.info("Bloque 7: Definiendo la clase GraphProcessor...")

class GraphProcessor:
    """
    Orquesta el procesamiento de archivos para extraer estructuras de grafo en JSON.
    Opcionalmente, carga la estructura extraída a Neo4j.

    Utiliza:
    - LLM Ollama (configurado específicamente) para chunking JSON robusto.
    - LLM Principal (configurable) para extraer la estructura del grafo (Nodos/Relaciones)
      a partir de los chunks y formatearla como JSON según el esquema `ExtractedGraph`.
    - Opcionalmente, interactúa con Neo4j para cargar datos y realizar consultas.
    """
    def __init__(self,
                 main_llm: BaseChatModel,       # LLM principal (para extracción JSON)
                 graph_instance: Optional[Neo4jGraph], # Instancia Neo4jGraph (SOLO si se carga/consulta)
                 chunking_prompt: PromptTemplate, # Prompt semántico para chunking
                 extraction_prompt: PromptTemplate,# Prompt para extraer grafo JSON
                 ollama_chunking_model: str,     # Modelo Ollama *para chunking*
                 ollama_url: str,                # URL Ollama *para chunking*
                 chunk_schema: type[BaseModel] = Chunks,       # Schema Pydantic para chunks
                 graph_schema: type[BaseModel] = ExtractedGraph # Schema Pydantic para grafo extraído
                ):
        """Inicializa el GraphProcessor."""

        if not main_llm: raise ValueError("Se requiere instancia del LLM principal (main_llm).")
        self.llm = main_llm
        logger.info(f"GraphProcessor inicializado con LLM principal tipo: {type(main_llm)}")

        self.graph = graph_instance # Puede ser None
        if self.graph: logger.info("Instancia Neo4jGraph proporcionada (para carga/consulta opcional).")
        else: logger.info("Instancia Neo4jGraph NO proporcionada (solo se generará JSON).")

        if not chunking_prompt: raise ValueError("Se requiere plantilla de prompt para chunking.")
        if not extraction_prompt: raise ValueError("Se requiere plantilla de prompt para extracción de grafo.")

        # --- Configuración Cadena de Chunking (SIEMPRE Ollama JSON) ---
        self.chunking_chain = None
        try:
            logger.info(f"Configurando LLM Ollama para chunking (Modelo: {ollama_chunking_model})...")
            # Añadir timeout a la llamada Ollama
            llm_chunking = ChatOllama(model=ollama_chunking_model, base_url=ollama_url, format="json", temperature=0.0, request_timeout=120.0) # Timeout 2 min
            chunking_parser = PydanticOutputParser(pydantic_object=chunk_schema)
            chunking_prompt_formatted = ChatPromptTemplate.from_messages([
                ("system", "Responde SIEMPRE usando formato JSON válido según el esquema.\n{format_instructions}"),
                ("human", chunking_prompt.template)
            ]).partial(format_instructions=chunking_parser.get_format_instructions())
            # Usar OutputFixingParser con el mismo LLM para robustez
            output_fixing_parser_chunking = OutputFixingParser.from_llm(parser=chunking_parser, llm=llm_chunking)
            self.chunking_chain = chunking_prompt_formatted | llm_chunking | output_fixing_parser_chunking
            logger.info("Cadena de Chunking (Ollama JSON + Fixer) configurada.")
        except Exception as e:
            logger.error(f"Fallo crítico configurando cadena de chunking Ollama: {e}", exc_info=True)
            raise Exception(f"Fallo en configuración de chunking Ollama: {e}") from e

        # --- Configuración Cadena de Extracción de Grafo (LLM Principal + Pydantic) ---
        self.graph_extraction_chain = None
        try:
            logger.info(f"Configurando cadena de extracción de grafo con LLM principal ({type(self.llm)})...")
            graph_parser = PydanticOutputParser(pydantic_object=graph_schema)

            # Crear el prompt completo incluyendo las instrucciones de formato Pydantic
            # Pasar `format_instructions` como variable al template
            self.extraction_prompt_formatted = PromptTemplate(
                template=extraction_prompt.template, # Usar el template original
                input_variables=["chunk_text"], # Variables que el usuario proveerá
                partial_variables={"format_instructions": graph_parser.get_format_instructions()} # Instrucciones Pydantic
            )

            # Crear la cadena: Prompt Formateado -> LLM -> Parser Pydantic
            # Añadir OutputFixingParser si se espera que el LLM falle mucho con JSON
            # graph_fixing_parser = OutputFixingParser.from_llm(parser=graph_parser, llm=self.llm)
            # self.graph_extraction_chain = self.extraction_prompt_formatted | self.llm | graph_fixing_parser
            self.graph_extraction_chain = self.extraction_prompt_formatted | self.llm | graph_parser
            logger.info("Cadena de Extracción de Grafo (LLM Principal + Pydantic Parser) configurada.")

        except Exception as e:
            logger.error(f"Fallo crítico configurando cadena de extracción de grafo: {e}", exc_info=True)
            raise Exception(f"Fallo en configuración de extracción de grafo: {e}") from e


    def _read_file(self, filepath: str) -> Optional[str]:
        """Lee contenido de archivo, probando codificaciones comunes."""
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        logger.debug(f"Intentando leer: {filepath}")
        try:
             # Comprobar si el archivo existe primero
            if not os.path.isfile(filepath):
                logger.error(f"Archivo no encontrado en la ruta especificada: {filepath}")
                return None
            for enc in encodings:
                try:
                    with open(filepath, "r", encoding=enc) as f:
                        # Leer todo el contenido, luego procesar
                        full_content = f.read()
                        # Unir líneas no vacías después de quitar espacios al inicio/fin
                        # y reemplazar múltiples espacios con uno solo
                        content = ' '.join(line.strip() for line in full_content.splitlines() if line.strip())
                        content = ' '.join(content.split()) # Normalizar espacios
                    logger.info(f"Archivo leído: {filepath} (codificación: {enc})")
                    return content
                except UnicodeDecodeError:
                    logger.debug(f"Fallo con codificación {enc} para {filepath}, intentando siguiente...")
                    continue
                except IOError as e: # Capturar errores de permiso, etc. aquí
                    logger.error(f"Error I/O leyendo {filepath} (permisos? disco lleno?): {e}")
                    return None # No seguir intentando si hay error de IO

            # Si salimos del bucle sin éxito en ninguna codificación
            logger.error(f"No se pudo decodificar el archivo {filepath} con las codificaciones probadas: {encodings}")
            return None
        except Exception as e: # Capturar otros errores inesperados
            logger.error(f"Error inesperado accediendo/leyendo archivo {filepath}: {e}", exc_info=True)
            return None


    def _chunk_text_with_ollama_json(self, text: str) -> Optional[List[Chunk]]:
        """Divide texto en chunks usando cadena Ollama JSON."""
        if not self.chunking_chain:
            logger.error("Cadena chunking (Ollama) no inicializada. No se puede dividir.")
            return None
        if not text or not text.strip():
            logger.warning("Texto para chunking vacío o solo espacios.")
            return []

        logger.info("Iniciando división de texto con Ollama (modo JSON)...")
        start_time = time.time()
        try:
            logger.debug(f"Invocando cadena chunking Ollama con texto (len={len(text)} chars)...")
            # El parser (con fixer) debería devolver un objeto Chunks o lanzar excepción
            response: Chunks = self.chunking_chain.invoke({"input": text})
            duration = time.time() - start_time

            if isinstance(response, Chunks) and hasattr(response, 'chunks'):
                num_chunks = len(response.chunks)
                logger.info(f"Chunking Ollama completado ({num_chunks} chunks) en {duration:.2f}s.")
                # Validar IDs secuenciales (opcional)
                for i, chk in enumerate(response.chunks):
                     if chk.chunk_id != i + 1:
                         logger.warning(f"Chunk ID no secuencial detectado: Esperado {i+1}, Obtenido {chk.chunk_id}. Se usará el obtenido.")
                return response.chunks
            else:
                # Esto teóricamente no debería ocurrir si el parser funciona/lanza excepción
                logger.error(f"Chunking Ollama devolvió tipo inesperado tras parseo/fix: {type(response)}. Contenido parcial: {str(response)[:500]}...")
                return None
        except Exception as e:
            duration = time.time() - start_time
            # Loguear el traceback completo
            tb_str = traceback.format_exc()
            logger.error(f"Error DETALLADO durante chunking Ollama (después de {duration:.2f}s): {e}\nTraceback:\n{tb_str}")
            # Podrías querer guardar el texto problemático para depuración
            # try:
            #     with open("error_chunking_input.txt", "w", encoding='utf-8') as f_err:
            #         f_err.write(text)
            #     logger.info("Texto de entrada problemático guardado en error_chunking_input.txt")
            # except Exception as save_e:
            #     logger.error(f"No se pudo guardar el texto de entrada del error: {save_e}")
            return None


    def _extract_graph_from_chunk(self, chunk: Chunk) -> Optional[ExtractedGraph]:
        """Extrae estructura de grafo (JSON) de un chunk usando LLM principal."""
        if not self.graph_extraction_chain:
            logger.error("Cadena de extracción de grafo no inicializada.")
            return None
        if not chunk or not chunk.text or not chunk.text.strip():
             logger.warning(f"Chunk {chunk.chunk_id if chunk else 'N/A'} está vacío. Saltando extracción.")
             return ExtractedGraph(nodes=[], relationships=[]) # Devolver grafo vacío

        logger.info(f"Iniciando extracción de grafo del chunk ID: {chunk.chunk_id} (len={len(chunk.text)} chars)...")
        start_time = time.time()
        try:
            # Invocar la cadena de extracción
            extracted_data: ExtractedGraph = self.graph_extraction_chain.invoke({
                "chunk_text": chunk.text
            })
            duration = time.time() - start_time

            # Validar resultado (Pydantic lo hace, pero podemos añadir más checks)
            if isinstance(extracted_data, ExtractedGraph):
                 node_count = len(extracted_data.nodes)
                 rel_count = len(extracted_data.relationships)
                 logger.info(f"Extracción grafo chunk {chunk.chunk_id} completada ({duration:.2f}s). Nodos: {node_count}, Rels: {rel_count}.")
                 # Validar que las props sean diccionarios (Pydantic debería asegurarlo, pero doble check)
                 for node in extracted_data.nodes: node.properties = node.properties or {}
                 for rel in extracted_data.relationships: rel.properties = rel.properties or {}
                 return extracted_data
            else: # Si el parser devuelve algo inesperado (poco probable con PydanticParser directo)
                 logger.error(f"Extracción grafo chunk {chunk.chunk_id} devolvió tipo inesperado: {type(extracted_data)}")
                 return None

        except ValidationError as e: # Error de validación Pydantic (JSON inválido del LLM)
             duration = time.time() - start_time
             logger.error(f"Error Validación Pydantic (JSON malformado?) en extracción chunk {chunk.chunk_id} ({duration:.2f}s): {e}", exc_info=False)
             # Intentar loguear la salida cruda si se puede (requiere ajuste en Langchain o try/except en parser)
             # logger.error(f"Salida cruda LLM (si disponible): ...")
             return None
        except Exception as e: # Otros errores (API, conexión LLM, timeout, etc.)
            duration = time.time() - start_time
            tb_str = traceback.format_exc()
            logger.error(f"Error DETALLADO en extracción chunk {chunk.chunk_id} ({duration:.2f}s): {e}\nTraceback:\n{tb_str}")
            return None


    def _aggregate_extracted_graphs(self, graph_parts: List[Optional[ExtractedGraph]]) -> Dict[str, Any]:
        """Combina nodos y relaciones de múltiples extracciones, eliminando duplicados y resolviendo conflictos simples."""
        aggregated_nodes: Dict[str, Node] = {} # Clave: node.id
        aggregated_relationships_list: List[Relationship] = [] # Guardar objetos para merge de props

        logger.debug(f"Agregando grafos de {len(graph_parts)} partes extraídas...")
        part_count = 0
        for part in filter(None, graph_parts): # Filtrar Nones y asegurar que es ExtractedGraph
             part_count += 1
             if not isinstance(part, ExtractedGraph):
                  logger.warning(f"Elemento inesperado encontrado durante agregación (tipo: {type(part)}). Omitiendo.")
                  continue

             # Agregar/Actualizar Nodos
             for node in part.nodes:
                 if not node.id or not node.label:
                      logger.warning(f"Nodo inválido omitido en parte {part_count}: ID='{node.id}', Label='{node.label}'")
                      continue
                 # Normalizar ID (ej. quitar espacios extra) puede ser útil aquí
                 node_id_norm = node.id.strip()
                 if not node_id_norm: continue # Saltar si ID queda vacío

                 if node_id_norm in aggregated_nodes:
                     # Fusionar propiedades: las nuevas tienen prioridad
                     existing_node = aggregated_nodes[node_id_norm]
                     merged_props = existing_node.properties.copy()
                     merged_props.update(node.properties or {}) # Asegurar que node.properties no sea None
                     existing_node.properties = merged_props
                     # Etiqueta: mantener la primera vista o implementar lógica de mayoría/prioridad
                     if existing_node.label != node.label:
                          logger.debug(f"Conflicto etiqueta nodo ID '{node_id_norm}': '{existing_node.label}' vs '{node.label}'. Se mantiene '{existing_node.label}'.")
                 else:
                     # Añadir nodo nuevo
                     node.id = node_id_norm # Usar ID normalizado
                     node.properties = node.properties or {} # Asegurar dict
                     aggregated_nodes[node_id_norm] = node

             # Agregar Relaciones (manejo simple de duplicados)
             for rel in part.relationships:
                 # Validar relación básica y normalizar IDs
                 source_id_norm = rel.source.strip() if rel.source else None
                 target_id_norm = rel.target.strip() if rel.target else None
                 rel_type = rel.type.strip() if rel.type else None

                 if not source_id_norm or not target_id_norm or not rel_type:
                      logger.warning(f"Relación inválida omitida en parte {part_count}: {source_id_norm} -[{rel_type}]-> {target_id_norm}")
                      continue
                 # Verificar si nodos existen (importante para consistencia)
                 if source_id_norm not in aggregated_nodes or target_id_norm not in aggregated_nodes:
                      logger.warning(f"Relación omitida: Nodo source ('{source_id_norm}') o target ('{target_id_norm}') no encontrado. Rel: {rel_type}")
                      continue

                 # Buscar si ya existe relación igual (mismo source, target, type)
                 found_existing = False
                 for i, existing_rel in enumerate(aggregated_relationships_list):
                      if existing_rel.source == source_id_norm and existing_rel.target == target_id_norm and existing_rel.type == rel_type:
                          # Fusionar propiedades
                          merged_props = existing_rel.properties.copy()
                          merged_props.update(rel.properties or {})
                          aggregated_relationships_list[i].properties = merged_props
                          found_existing = True
                          logger.debug(f"Relación existente actualizada: {source_id_norm}-[{rel_type}]->{target_id_norm}")
                          break
                 if not found_existing:
                      # Añadir nueva relación usando IDs normalizados
                      rel.source = source_id_norm
                      rel.target = target_id_norm
                      rel.type = rel_type
                      rel.properties = rel.properties or {} # Asegurar dict
                      aggregated_relationships_list.append(rel)

        # Convertir a formato serializable (dict)
        final_structure = {
            "nodes": [node.dict() for node in aggregated_nodes.values()],
            "relationships": [rel.dict() for rel in aggregated_relationships_list]
        }
        logger.info(f"Agregación completada. Nodos únicos: {len(final_structure['nodes'])}, Relaciones únicas: {len(final_structure['relationships'])}.")
        return final_structure


    def _save_json(self, data: Dict[str, Any], filepath: str):
        """Guarda los datos en un archivo JSON con formato legible."""
        logger.info(f"Intentando guardar JSON en: {filepath}")
        try:
            # Asegurar que el directorio existe
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2) # indent=2 para legibilidad
            logger.info(f"Estructura del grafo guardada exitosamente.")
            return True
        except IOError as e:
            logger.error(f"Error de E/S al guardar JSON en {filepath}: {e}")
        except TypeError as e:
            logger.error(f"Error de tipo al serializar JSON para {filepath} (objeto no serializable?): {e}")
        except Exception as e:
            logger.error(f"Error inesperado al guardar JSON en {filepath}: {e}", exc_info=True)
        return False


    def _visualize_json(self, data: Dict[str, Any], max_items: int = 10):
        """Imprime una vista previa del JSON en la consola/notebook."""
        try:
            print("\n--- Visualización Previa JSON Extraído ---")
            if not isinstance(data, dict):
                 print("Error: Datos inválidos para visualizar.")
                 return

            nodes = data.get("nodes", [])
            rels = data.get("relationships", [])

            # Crear una copia para la vista previa
            preview_data = {
                "nodes": nodes[:max_items],
                "relationships": rels[:max_items]
            }

            print(json.dumps(preview_data, indent=2, ensure_ascii=False))

            if len(nodes) > max_items:
                print(f"  (... {len(nodes) - max_items} nodos más ...)")
            if len(rels) > max_items:
                print(f"  (... {len(rels) - max_items} relaciones más ...)")
            print(f"--- (Mostrando hasta {max_items} nodos/relaciones. Total: {len(nodes)} N, {len(rels)} R) ---")

        except Exception as e:
            logger.error(f"Error al visualizar JSON: {e}")
            print("Error al visualizar JSON.")


    def process_file_to_json(self, filepath: str, output_dir: str, print_chunks_flag: bool, visualize_json_flag: bool) -> Optional[Dict[str, Any]]:
        """
        Procesa un archivo: lee, chunkea (Ollama), extrae grafo (LLM principal),
        agrega resultados, guarda JSON y opcionalmente lo visualiza.
        Devuelve la estructura de grafo agregada como diccionario, o None si falla gravemente.
        """
        logger.info(f"--- Iniciando procesamiento del archivo: {filepath} ---")
        file_basename = os.path.basename(filepath)

        # 1. Leer Archivo
        content = self._read_file(filepath)
        if not content:
            logger.error(f"No se pudo leer {file_basename}. Saltando.")
            return None

        # 2. Chunking (Ollama JSON)
        chunks = self._chunk_text_with_ollama_json(content)
        if chunks is None:
            logger.error(f"Fallo en chunking para {file_basename}. Saltando extracción.")
            return None
        if not chunks:
            logger.warning(f"No se generaron chunks para {file_basename}. No se extraerá grafo.")
            empty_graph = {"nodes": [], "relationships": []}
            output_filename = os.path.splitext(file_basename)[0] + ".graph.json"
            output_filepath = os.path.join(output_dir, output_filename)
            self._save_json(empty_graph, output_filepath) # Guardar archivo vacío
            return empty_graph # Indicar que se procesó pero no hubo contenido

        if print_chunks_flag:
            print(f"\n--- Chunks (Ollama) para {file_basename} ({len(chunks)}) ---")
            for chk in chunks[:5]: print(f"ID: {chk.chunk_id}, Texto: '{chk.text[:80].strip()}...'")
            if len(chunks) > 5: print("  (...)")

        # 3. Extracción de Grafo por Chunk (LLM Principal)
        extracted_graph_parts: List[Optional[ExtractedGraph]] = []
        total_chunks = len(chunks)
        logger.info(f"Iniciando extracción de grafo para {total_chunks} chunks de {file_basename}...")
        start_extraction_time = time.time()
        successful_extractions = 0
        failed_extractions = 0

        for i, chunk in enumerate(chunks):
            current_time = time.time()
            if (i + 1) % 20 == 0 or current_time - start_extraction_time > 90: # Log cada 20 chunks o 90 segs
                 elapsed_time = current_time - start_extraction_time
                 logger.info(f"  Progreso extracción {file_basename}: Chunk {i+1}/{total_chunks}... ({elapsed_time:.1f}s desde último log)")
                 start_extraction_time = current_time # Reset timer

            graph_part = self._extract_graph_from_chunk(chunk)
            if graph_part is not None:
                # Solo añadir si tiene nodos o relaciones
                if graph_part.nodes or graph_part.relationships:
                     extracted_graph_parts.append(graph_part)
                     successful_extractions += 1
                else:
                     logger.debug(f"Extracción chunk {chunk.chunk_id} resultó en grafo vacío. Omitiendo.")
                     # Contar como éxito si no hubo error, aunque esté vacío
                     successful_extractions +=1 # O decidir no contarlo
            else: # Hubo un error en la extracción
                failed_extractions += 1
                # El error ya se logueó en _extract_graph_from_chunk

        logger.info(f"Extracción de chunks para {file_basename} completada. Exitosos/Intentados: {successful_extractions}/{total_chunks}, Fallos: {failed_extractions}.")

        # Decidir si continuar si hubo muchos fallos
        # if failed_extractions > total_chunks * 0.5: # Ejemplo: si más del 50% fallaron
        #      logger.error(f"Demasiados fallos ({failed_extractions}) en extracción para {file_basename}. Abortando agregación/guardado.")
        #      return None

        if not extracted_graph_parts:
             logger.error(f"No se pudo extraer ninguna estructura de grafo válida (con nodos/rels) para {file_basename}.")
             empty_graph = {"nodes": [], "relationships": []}
             output_filename = os.path.splitext(file_basename)[0] + ".graph.json"
             output_filepath = os.path.join(output_dir, output_filename)
             self._save_json(empty_graph, output_filepath)
             return None # Indicar fallo global en extracción

        # 4. Agregar Resultados
        logger.info(f"Agregando resultados de {len(extracted_graph_parts)} partes extraídas para {file_basename}...")
        aggregated_graph_data = self._aggregate_extracted_graphs(extracted_graph_parts)

        # 5. Guardar JSON Agregado
        output_filename = os.path.splitext(file_basename)[0] + ".graph.json"
        output_filepath = os.path.join(output_dir, output_filename)
        save_success = self._save_json(aggregated_graph_data, output_filepath)

        # 6. Visualizar JSON (Opcional)
        if visualize_json_flag and save_success:
            self._visualize_json(aggregated_graph_data)

        logger.info(f"--- Procesamiento JSON para archivo {file_basename} completado ---")
        return aggregated_graph_data # Devolver datos agregados

    # (Dentro de la clase GraphProcessor en Bloque 7)

    def _load_graph_data_to_neo4j(self, graph_data: Dict[str, Any], driver: Driver, db_name: Optional[str] = None):
        """
        Carga la estructura de grafo JSON a Neo4j usando MERGE y consultas parametrizadas.
        Utiliza transacciones explícitas y maneja la detección de APOC.
        Devuelve True si la carga fue exitosa (sin errores), False en caso contrario.
        """
        if not graph_data or (not graph_data.get("nodes") and not graph_data.get("relationships")):
             logger.warning("No hay datos de grafo válidos para cargar en Neo4j.")
             return True # Éxito si no hay nada que cargar

        nodes_to_load = graph_data.get("nodes", [])
        rels_to_load = graph_data.get("relationships", [])

        logger.info(f"Iniciando carga a Neo4j: {len(nodes_to_load)} nodos, {len(rels_to_load)} relaciones...")
        start_time = time.time()
        nodes_processed_count = 0 # Contador basado en lotes procesados sin error
        rels_processed_count = 0  # Contador basado en lotes procesados sin error
        errors = 0
        apoc_available = False # Flag para saber qué queries usar

        # --- Inicio Transacción ---
        try:
            with driver.session(database=db_name, default_access_mode="WRITE") as session:

                # --- Detectar APOC ---
                try:
                     session.run("RETURN apoc.version() LIMIT 1").consume()
                     apoc_available = True
                     logger.info("APOC detectado. Se usarán funciones APOC para carga.")
                except Exception:
                     logger.warning("APOC no detectado o no funcional. Se usarán métodos alternativos.")
                     apoc_available = False

                # --- Cargar Nodos ---
                if nodes_to_load:
                    logger.debug(f"Ejecutando carga de {len(nodes_to_load)} nodos...")
                    if apoc_available:
                        node_query = """
                        UNWIND $nodes AS node_batch
                        MERGE (n {id: node_batch.id})
                        ON CREATE SET n = node_batch.properties, n.id = node_batch.id
                        ON MATCH SET n += node_batch.properties
                        WITH n, node_batch.label AS labelStr
                        WHERE labelStr IS NOT NULL AND labelStr <> ''
                        CALL apoc.create.addLabels(n, [labelStr]) YIELD node
                        RETURN count(node) AS processed_count
                        """
                    else: # Sin APOC
                        node_query = """
                        UNWIND $nodes AS node_batch
                        MERGE (n {id: node_batch.id})
                        ON CREATE SET n = node_batch.properties, n.id = node_batch.id, n._label = node_batch.label
                        ON MATCH SET n += node_batch.properties, n._label = node_batch.label
                        RETURN count(n) AS processed_count
                        """
                    try:
                        batch_size = 500
                        for i in range(0, len(nodes_to_load), batch_size):
                            batch = nodes_to_load[i:i+batch_size]
                            logger.debug(f"  Cargando lote de nodos {i+1}-{i+len(batch)}...")
                            # --- CORRECCIÓN APLICADA AQUÍ ---
                            # Ejecutar y consumir para obtener summary y detectar errores
                            summary = session.run(node_query, parameters={"nodes": batch}).consume()
                            # Registrar notificaciones (advertencias) de Neo4j
                            if summary.notifications:
                                logger.warning(f"Notificaciones Neo4j (carga nodos lote {i+1}): {summary.notifications}")
                            # Contar el lote como procesado si no hubo excepción
                            nodes_processed_count += len(batch)
                            # --------------------------------
                        logger.info(f"Lotes de nodos procesados/intentados en Neo4j: {nodes_processed_count}")
                    except Exception as node_e:
                        logger.error(f"Error durante carga de lote de nodos: {node_e}", exc_info=True)
                        errors += 1
                        raise # Relanzar para abortar la transacción completa

                else:
                    logger.info("No hay nodos para cargar.")

                # --- Cargar Relaciones (Solo si no hubo errores en nodos) ---
                if rels_to_load and errors == 0:
                    logger.debug(f"Ejecutando carga de {len(rels_to_load)} relaciones...")
                    if apoc_available:
                        rel_query = """
                        UNWIND $rels AS rel_batch
                        MATCH (source {id: rel_batch.source})
                        MATCH (target {id: rel_batch.target})
                        CALL apoc.create.relationship(source, rel_batch.type, rel_batch.properties, target) YIELD rel
                        RETURN count(rel) AS processed_count
                        """
                    else: # Sin APOC
                        rel_query = """
                        UNWIND $rels AS rel_batch
                        MATCH (source {id: rel_batch.source})
                        MATCH (target {id: rel_batch.target})
                        MERGE (source)-[r:RELATED]->(target)
                        SET r = rel_batch.properties, r.type = rel_batch.type
                        RETURN count(r) AS processed_count
                        """
                    try:
                        batch_size = 500
                        for i in range(0, len(rels_to_load), batch_size):
                             batch = rels_to_load[i:i+batch_size]
                             logger.debug(f"  Cargando lote de relaciones {i+1}-{i+len(batch)}...")
                             # --- CORRECCIÓN APLICADA AQUÍ ---
                             # Ejecutar y consumir
                             summary = session.run(rel_query, parameters={"rels": batch}).consume()
                             if summary.notifications:
                                 logger.warning(f"Notificaciones Neo4j (carga rels lote {i+1}): {summary.notifications}")
                             # Contar lote como procesado si no hubo excepción
                             rels_processed_count += len(batch)
                             # --------------------------------
                        logger.info(f"Lotes de relaciones procesados/intentados en Neo4j: {rels_processed_count}")
                    except Exception as rel_e:
                         logger.error(f"Error durante carga de lote de relaciones: {rel_e}", exc_info=True)
                         errors += 1
                         raise # Abortar transacción
                elif errors > 0:
                     logger.warning("Carga de relaciones omitida debido a errores previos en carga de nodos.")
                else:
                    logger.info("No hay relaciones para cargar.")

            # Fin de la transacción (commit automático si no hubo excepciones)
            duration = time.time() - start_time
            logger.info(f"Carga a Neo4j (transacción) finalizada en {duration:.2f}s. Errores DENTRO de la transacción: {errors}")
            return errors == 0 # Devuelve True si no hubo errores

        except Exception as tx_e: # Capturar error que abortó la transacción
             logger.error(f"Error en la transacción de carga Neo4j (rollback realizado): {tx_e}", exc_info=True)
             return False # La carga falló


    # --- Métodos de Consulta (Solo si Neo4j está cargado) ---
    def query_graph(self, question: str, query_prompt_template: PromptTemplate) -> str:
        """Consulta el grafo Neo4j usando el LLM principal."""
        global driver # Acceder al driver global si está disponible para resumen
        if not self.graph: return "Error: Neo4j no configurado para consultas en este procesador."
        if not query_prompt_template: return "Error: Falta plantilla de consulta."
        if not self.llm: return "Error: LLM principal no inicializado."

        logger.info(f"Recibida consulta al grafo Neo4j: '{question}'")
        try:
            # Obtener esquema y resumen actualizados de Neo4j
            logger.debug("Refrescando esquema Neo4j para consulta...")
            self.graph.refresh_schema()
            schema_context = self.graph.schema
            logger.debug(f"Esquema Neo4j para contexto:\n{schema_context}")

            summary_context = "(Resumen Neo4j no disponible)"
            db_to_query = self.graph.database # Obtener DB del wrapper
            if driver and driver.is_open():
                 logger.debug(f"Intentando obtener resumen con driver directo para DB: {db_to_query}")
                 summary_context = retrieve_graph_summary(driver, db_name=db_to_query)
            else:
                 logger.debug("Driver directo no disponible, intentando resumen simple vía wrapper...")
                 try: # Fallback con consulta simple vía wrapper
                     summary_res = self.graph.query("CALL db.labels() YIELD label RETURN collect(label) as labels")
                     summary_context = f"Tipos de Nodos: {summary_res[0]['labels'] if summary_res else 'N/A'}"
                 except Exception as e_sum: logger.warning(f"Fallo resumen simple vía wrapper: {e_sum}")

            logger.info(f"Invocando LLM principal ({type(self.llm)}) para consulta Neo4j...")
            query_chain = query_prompt_template | self.llm | StrOutputParser()
            start_time = time.time()
            response = query_chain.invoke({
                "question": question,
                "knowledge_graph_schema": schema_context,
                "knowledge_graph_summary": summary_context
            })
            duration = time.time() - start_time
            logger.info(f"Respuesta consulta Neo4j ({type(self.llm)}) recibida en {duration:.2f}s.")
            return response

        except Exception as e:
            logger.error(f"Error durante consulta del grafo Neo4j: {e}", exc_info=True)
            return "Error: Problema al procesar la consulta a Neo4j."


    def interactive_query(self, query_prompt_template: PromptTemplate):
        """Inicia bucle de consulta interactiva contra Neo4j."""
        if not self.graph: print("Error: Neo4j no configurado."); return
        if not self.llm: print("Error: LLM principal no listo."); return
        if not query_prompt_template: print("Error: Falta plantilla de consulta."); return

        print("\n--- Modo Consulta Interactiva (Neo4j) ---")
        print(f"(Usando LLM principal: {llm_type})")
        print("Introduce tu pregunta sobre el grafo Neo4j. Escribe 'exit' o 'quit' para salir.")

        while True:
            try: question = input("\nPregunta Neo4j> ").strip()
            except EOFError: logger.warning("EOF recibido, saliendo."); break
            if question.lower() in ['exit', 'quit']: logger.info("Saliendo modo interactivo."); break
            if not question: continue

            print("Procesando consulta Neo4j...")
            answer = self.query_graph(question, query_prompt_template)
            print("\nRespuesta:") ; print("-" * 10) ; print(answer) ; print("-" * 10)
        print("--- Fin Modo Interactivo ---")


print("Clase GraphProcessor definida.")
logger.info("Bloque 7: Definición de GraphProcessor completada.")


# In[8]:


# ============================================================
# %% Bloque 8: Configuración Opcional de Neo4j
# ============================================================
logger.info("Bloque 8: Configurando conexión Neo4j (si es necesario)...")
print("\n" + "="*50 + "\n=== Bloque 8: Configuración Opcional Neo4j ===\n" + "="*50)

# --- Verificación INICIAL de variables del Bloque 2 (usando globals()) ---
# Comprueba si las variables existen en el scope GLOBAL del kernel
print("Bloque 8: Verificando existencia de variables globales...") # DEBUG PRINT
required_vars = ['load_into_neo4j', 'run_interactive_query', 'neo4j_configured',
                 'NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD', 'NEO4J_DATABASE',
                 'delete_indexes_before_load', 'clear_graph_before_load',
                 'show_neo4j_browser'] # Añadida show_neo4j_browser

# Comprobar usando globals()
missing_vars = [var for var in required_vars if var not in globals()]

# Imprimir estado de variables clave para depuración
try:
    print(f"Bloque 8: Estado GLOBAL de 'load_into_neo4j': {globals().get('load_into_neo4j', 'NO ENCONTRADA')}")
    print(f"Bloque 8: Estado GLOBAL de 'run_interactive_query': {globals().get('run_interactive_query', 'NO ENCONTRADA')}")
except Exception as e_dbg:
    print(f"Bloque 8: Error al verificar variables globales: {e_dbg}")


if missing_vars:
    error_msg = f"Faltan variables GLOBALES de configuración (¿Bloque 2 no ejecutado correctamente en este kernel?): {', '.join(missing_vars)}."
    logger.error(error_msg)
    print(f"ERROR: {error_msg}")
    # Detener la ejecución de esta celda si faltan variables clave
    raise NameError(f"Variables globales requeridas no definidas: {', '.join(missing_vars)}")
else:
    logger.info("Verificación inicial de variables globales de configuración pasada.")
    # --- Asignar variables globales a locales para este bloque ---
    # Esto asegura que el resto del código del bloque use los valores correctos.
    load_into_neo4j = globals()['load_into_neo4j']
    run_interactive_query = globals()['run_interactive_query']
    neo4j_configured = globals()['neo4j_configured']
    NEO4J_URI = globals()['NEO4J_URI']
    NEO4J_USERNAME = globals()['NEO4J_USERNAME']
    NEO4J_PASSWORD = globals()['NEO4J_PASSWORD']
    NEO4J_DATABASE = globals()['NEO4J_DATABASE']
    delete_indexes_before_load = globals()['delete_indexes_before_load']
    clear_graph_before_load = globals()['clear_graph_before_load']
    show_neo4j_browser = globals()['show_neo4j_browser']
    # --------------------------------------------------------------

# --- Variables para este bloque ---
# Estas SÍ son locales a este bloque/celda
driver: Optional[Driver] = None
graph: Optional[Neo4jGraph] = None # Wrapper Langchain

# --- Lógica de conexión ---
# Ahora usa las variables locales definidas justo arriba
if load_into_neo4j or run_interactive_query:
    logger.info("Se requiere conexión Neo4j (para carga o consulta).")
    if not neo4j_configured:
        logger.error("Falta NEO4J_PASSWORD. No se puede conectar a Neo4j.")
        print("ERROR: Falta contraseña Neo4j. La carga/consulta fallará.")
        # Desactivar flags locales para evitar errores posteriores en esta ejecución
        load_into_neo4j = False
        run_interactive_query = False
        show_neo4j_browser = False
    else:
        try:
            # 1. Driver Directo (para limpieza opcional y carga robusta)
            logger.info("Obteniendo driver directo Neo4j...")
            # Usar la función definida en Bloque 5 (debe estar en globals())
            if 'get_neo4j_driver' not in globals(): raise NameError("Función get_neo4j_driver no definida. Ejecuta Bloque 5.")
            driver = get_neo4j_driver(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
            if not driver: raise ConnectionError("Fallo al obtener driver directo Neo4j.")

            db_name_for_log = NEO4J_DATABASE if NEO4J_DATABASE else "default"
            logger.info(f"Driver Neo4j conectado a BD: '{db_name_for_log}'")

            # 2. Wrapper Langchain (SOLO si se va a consultar interactivamente)
            if run_interactive_query: # Usa la variable local
                logger.info(f"Inicializando wrapper Neo4jGraph para BD '{db_name_for_log}' (para consultas)...")
                if 'Neo4jGraph' not in globals(): raise NameError("Clase Neo4jGraph no importada. Ejecuta Bloque 1.")
                graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)
                graph.refresh_schema() # Verificar conexión
                logger.info(f"Wrapper Neo4jGraph conectado a BD '{db_name_for_log}'.")
                print(f"Wrapper Neo4jGraph listo para consultas en BD '{db_name_for_log}'.")
            else:
                 logger.info("Wrapper Neo4jGraph no necesario (run_interactive_query=False).")
                 graph = None # Asegurar que sea None si no se usa

            # 3. Limpieza Opcional (ANTES de cualquier carga, requiere driver)
            if load_into_neo4j: # Usa la variable local
                 # Asegurarse que las funciones de limpieza están definidas
                 if 'delete_all_user_indexes' not in globals() or 'reset_graph_data' not in globals():
                      raise NameError("Funciones de limpieza (delete_all_user_indexes/reset_graph_data) no definidas. Ejecuta Bloque 5.")

                 if delete_indexes_before_load: # Usa la variable local
                     logger.warning(f"'delete_indexes_before_load' activado para BD '{db_name_for_log}'.")
                     print(f"\nIntentando borrar índices usuario en BD '{db_name_for_log}'...")
                     delete_all_user_indexes(driver, db_name=NEO4J_DATABASE)
                     print("-" * 20) # Separador visual
                 if clear_graph_before_load: # Usa la variable local
                     logger.warning(f"'clear_graph_before_load' activado para BD '{db_name_for_log}'.")
                     print(f"\nIntentando borrar TODOS los datos en BD '{db_name_for_log}'...")
                     # La función reset_graph_data ya pide doble confirmación
                     reset_graph_data(driver, db_name=NEO4J_DATABASE)
                     print("-" * 20) # Separador visual

            print(f"Configuración Neo4j completada (Driver {'y Wrapper' if graph else 'solamente'}).")

        except Exception as e:
            logger.error(f"Error crítico durante configuración Neo4j: {e}", exc_info=True)
            print(f"ERROR configurando Neo4j: {e}")
            # Limpieza de driver
            if driver:
                try:
                    driver.close()
                    logger.info("Driver cerrado debido a error en configuración.")
                except Exception as close_e:
                    logger.error(f"Error al intentar cerrar el driver tras error: {close_e}")
            driver = None; graph = None
            # Desactivar flags locales que dependen de Neo4j si falla la conexión
            load_into_neo4j = False
            run_interactive_query = False
            show_neo4j_browser = False
            logger.warning("Se desactivó la carga y consulta a Neo4j debido a error de conexión/configuración.")
else:
    logger.info("No se requiere conexión Neo4j para esta ejecución (load_into_neo4j=False y run_interactive_query=False).")
    print("Conexión Neo4j no necesaria.")

logger.info("Bloque 8: Configuración opcional Neo4j completada.")


# In[9]:


# ============================================================
# %% Bloque 9: Ejecución - Extracción de Grafo a JSON
# ============================================================
logger.info("Bloque 9: Iniciando fase de extracción de grafo a JSON...")
print("\n" + "="*50 + "\n=== Bloque 9: Extracción de Grafo a JSON ===\n" + "="*50)

processor: Optional[GraphProcessor] = None
all_extracted_data: Dict[str, Optional[Dict[str, Any]]] = {} # Guarda datos JSON por archivo de entrada

if not skip_extraction:
    # Verificar dependencias clave
    if not llm:
        logger.error("LLM principal no inicializado. Saltando extracción.")
        print("ERROR: LLM principal no disponible. No se puede extraer.")
    elif not prompt_chunking or not prompt_extraction:
         logger.error("Faltan plantillas de prompt. Saltando extracción.")
         print("ERROR: Faltan prompts necesarios. No se puede extraer.")
    else:
        try:
            logger.info("Instanciando GraphProcessor...")
            # Pasar `graph` (puede ser None si no se carga/consulta Neo4j)
            # El driver directo no es necesario para la extracción JSON
            processor = GraphProcessor(
                main_llm=llm,
                graph_instance=graph, # Pasar instancia Neo4jGraph (o None, solo usado para consultas opcionales)
                chunking_prompt=prompt_chunking,
                extraction_prompt=prompt_extraction,
                ollama_chunking_model=ollama_chunking_model_name,
                ollama_url=OLLAMA_BASE_URL
                # schemas ya definidos por defecto
            )
            logger.info("GraphProcessor instanciado.")
            print("GraphProcessor listo para extracción.")

            # Procesar cada archivo
            start_total_time = time.time()
            num_files = len(input_filepaths)
            processed_files_count = 0
            failed_files_list = []

            for i, filepath in enumerate(input_filepaths):
                file_basename = os.path.basename(filepath)
                logger.info(f"--- Procesando archivo {i+1}/{num_files}: {file_basename} ---")
                print(f"\nProcesando archivo {i+1}/{num_files}: {file_basename}")
                start_file_time = time.time()
                try:
                    extracted_data = processor.process_file_to_json(
                        filepath=filepath,
                        output_dir=output_directory,
                        print_chunks_flag=print_chunks,
                        visualize_json_flag=visualize_json
                    )
                    all_extracted_data[filepath] = extracted_data # Guardar resultado (incluso None o vacío)
                    file_duration = time.time() - start_file_time
                    if extracted_data is not None: # Considerar éxito si devuelve algo
                        processed_files_count += 1
                        # Contar nodos/rels extraídos si no es None
                        node_count = len(extracted_data.get("nodes", [])) if extracted_data else 0
                        rel_count = len(extracted_data.get("relationships", [])) if extracted_data else 0
                        print(f"-> Procesado ({file_duration:.2f}s). Nodos: {node_count}, Rels: {rel_count}. JSON guardado.")
                    else:
                        failed_files_list.append(file_basename)
                        print(f"-> Fallo durante el procesamiento ({file_duration:.2f}s) (ver logs).")
                except KeyboardInterrupt:
                     logger.warning("Procesamiento interrumpido por el usuario.")
                     print("\nInterrupción por teclado detectada. Abortando...")
                     raise # Relanzar para detener completamente
                except Exception as file_e: # Capturar errores inesperados por archivo
                     file_duration = time.time() - start_file_time
                     logger.error(f"Error fatal procesando archivo {filepath} ({file_duration:.2f}s): {file_e}", exc_info=True)
                     print(f"-> ERROR FATAL procesando {file_basename} ({file_duration:.2f}s) (ver logs).")
                     failed_files_list.append(file_basename)
                     all_extracted_data[filepath] = None # Marcar como fallido
                # Pausa opcional entre archivos para evitar rate limits o sobrecarga
                # time.sleep(1)

            end_total_time = time.time()
            total_duration_str = f"{end_total_time - start_total_time:.2f}s"
            logger.info(f"Extracción JSON completada para {num_files} archivos en {total_duration_str}.")
            print(f"\n--- Resumen Extracción JSON ({total_duration_str}) ---")
            print(f"Archivos intentados: {num_files}")
            print(f"Procesados exitosamente (JSON generado): {processed_files_count}")
            if failed_files_list:
                print(f"Archivos con fallos durante procesamiento: {len(failed_files_list)} ({', '.join(failed_files_list)})")

        except Exception as e:
            logger.error(f"Error fatal durante instanciación o bucle de extracción: {e}", exc_info=True)
            print(f"¡ERROR FATAL durante el proceso de extracción!: {e}")
            processor = None # Asegurar que no se use si falló
else:
    logger.info("'skip_extraction' activado. Saltando extracción JSON.")
    print("Extracción JSON omitida.")
    # Instanciar procesador si se necesita para consultas Neo4j posteriores
    if run_interactive_query and load_into_neo4j:
        # Solo instanciar si los componentes necesarios están listos
        if llm and graph and prompt_chunking and prompt_extraction:
             try:
                 logger.info("Instanciando GraphProcessor (skip extraction) para consultas Neo4j...")
                 processor = GraphProcessor(
                     main_llm=llm, graph_instance=graph, chunking_prompt=prompt_chunking,
                     extraction_prompt=prompt_extraction, ollama_chunking_model=ollama_chunking_model_name,
                     ollama_url=OLLAMA_BASE_URL
                 )
                 print("GraphProcessor instanciado (extracción omitida) para consultas.")
             except Exception as e:
                 logger.error(f"Fallo instanciación tardía: {e}"); processor=None
        else:
             logger.warning("Faltan componentes para instanciar GraphProcessor para consultas.")
             print("Advertencia: No se pudo preparar para consultas (faltan llm/graph/prompts).")


logger.info("Bloque 9: Fase de extracción JSON completada (o saltada).")


# In[10]:


# ============================================================
# %% Bloque 10: Ejecución - Carga Opcional a Neo4j
# ============================================================
logger.info("Bloque 10: Iniciando carga opcional a Neo4j...")
print("\n" + "="*50 + "\n=== Bloque 10: Carga Opcional a Neo4j ===\n" + "="*50)

if load_into_neo4j:
    logger.info("'load_into_neo4j' activado.")
    # Necesitamos el driver directo para la carga robusta con transacciones
    if driver: # Verificar si el driver está disponible (creado en Bloque 8)
        print("Intentando cargar datos JSON extraídos a Neo4j...")
        total_files_attempted_load = 0
        files_loaded_successfully = 0
        files_with_load_errors = []
        start_load_time = time.time()

        # Iterar sobre los datos extraídos en el bloque anterior
        if 'all_extracted_data' not in locals():
             logger.warning("Variable 'all_extracted_data' no encontrada. ¿Se saltó la extracción?")
             all_extracted_data = {} # Evitar error

        for filepath, graph_data in all_extracted_data.items():
            file_basename = os.path.basename(filepath)
            # Cargar solo si hay datos válidos extraídos
            if graph_data is not None and isinstance(graph_data, dict) and (graph_data.get("nodes") or graph_data.get("relationships")):
                total_files_attempted_load += 1
                logger.info(f"Cargando datos JSON del archivo: {file_basename} a Neo4j...")
                print(f"Cargando: {file_basename}...")
                try:
                    # Asegurar que processor esté disponible si es necesario (si el método de carga NO fuera estático)
                    # if 'processor' not in locals() or not processor:
                    #     if llm and prompt_chunking and prompt_extraction:
                    #         logger.warning("Instanciando GraphProcessor mínimamente...")
                    #         processor = GraphProcessor(llm, graph, prompt_chunking, prompt_extraction, ollama_chunking_model_name, OLLAMA_BASE_URL)
                    #     else:
                    #         raise RuntimeError("GraphProcessor no instanciado y faltan componentes.")

                    # Llamar al método de carga (pasando el driver)
                    # Asumiendo que _load_graph_data_to_neo4j está en la clase Processor
                    # Si la hiciste independiente, llama a esa función directamente
                    if not processor: raise RuntimeError("GraphProcessor no está instanciado para llamar a _load_graph_data_to_neo4j")
                    success = processor._load_graph_data_to_neo4j(graph_data, driver, NEO4J_DATABASE)

                    if success:
                        files_loaded_successfully += 1
                        print(f"  -> Carga exitosa.")
                    else:
                        files_with_load_errors.append(file_basename)
                        print(f"  -> Error durante la carga (transacción fallida, ver logs).")
                except Exception as load_e:
                     logger.error(f"Error inesperado llamando a carga para {file_basename}: {load_e}", exc_info=True)
                     files_with_load_errors.append(file_basename)
                     print(f"  -> ERROR FATAL durante la carga (ver logs).")
            elif graph_data is None:
                logger.warning(f"Carga omitida para {file_basename} porque la extracción falló.")
            else: # Datos vacíos
                logger.info(f"Carga omitida para {file_basename} (sin nodos/rels extraídos).")

        load_duration = time.time() - start_load_time
        print(f"\n--- Resumen Carga Neo4j ({load_duration:.2f}s) ---")
        print(f"Archivos con datos para intentar cargar: {total_files_attempted_load}")
        print(f"Cargados exitosamente (transacciones completadas): {files_loaded_successfully}")
        if files_with_load_errors:
            print(f"Archivos con errores durante carga (transacción fallida): {len(files_with_load_errors)} ({', '.join(files_with_load_errors)})")
        logger.info(f"Resumen carga Neo4j: Intentados={total_files_attempted_load}, Éxitos={files_loaded_successfully}, Errores={len(files_with_load_errors)}")

        # Mostrar estado final y abrir navegador (solo si el driver sigue siendo válido)
        if driver:
            print("\n--- Estado del Grafo Neo4j Post-Carga ---")
            if 'retrieve_graph_summary' not in locals() or 'print_indexes' not in locals():
                print("Funciones de resumen/índices no definidas (Ejecuta Bloque 5).")
            else:
                print(retrieve_graph_summary(driver, db_name=NEO4J_DATABASE))
                print_indexes(driver, db_name=NEO4J_DATABASE)
            print("-" * 50)

            if show_neo4j_browser:
                logger.info("'show_neo4j_browser' activado post-carga.")
                print("\nIntentando abrir Neo4j Browser...")
                neo4j_browser_url = "http://localhost:7474/browser/"
                try: host = urlparse(NEO4J_URI).hostname
                except Exception: host = None
                if host and host not in ["localhost", "127.0.0.1"]: neo4j_browser_url = f"http://{host}:7474/browser/"
                if 'display_neo4j_browser' in locals():
                    display_neo4j_browser(neo4j_browser_url)
                else: print("Función display_neo4j_browser no definida (Ejecuta Bloque 5).")
        else:
             print("\nNo se puede mostrar estado final de Neo4j (driver no disponible).")

    elif not driver:
         logger.error("Carga a Neo4j omitida: Driver Neo4j no disponible o cerrado previamente.")
         print("ERROR: Carga a Neo4j omitida (Driver Neo4j no listo).")

else:
    logger.info("'load_into_neo4j' desactivado. Saltando carga a Neo4j.")
    print("Carga a Neo4j desactivada.")

logger.info("Bloque 10: Fase de carga opcional a Neo4j completada.")


# In[ ]:


# ============================================================
# %% Bloque 11: Ejecución - Consulta Interactiva (Opcional, si Neo4j cargado)
# ============================================================
logger.info("Bloque 11: Iniciando consulta interactiva opcional...")
print("\n" + "="*50 + "\n=== Bloque 11: Consulta Interactiva Opcional (Neo4j) ===\n" + "="*50)

if run_interactive_query:
    logger.info("'run_interactive_query' activado.")
    # Requiere que Neo4j se haya cargado (implícito si load_into_neo4j=True)
    # Y que el procesador (con wrapper 'graph') esté listo E instanciado (puede no estarlo si skip_extraction=True)
    if load_into_neo4j: # Solo permitir si se cargó
        if processor and graph and prompt_query: # graph (wrapper) solo se crea si run_interactive_query=True en Bloque 8
            logger.info("Iniciando sesión de consulta interactiva Neo4j...")
            try: processor.interactive_query(prompt_query)
            except Exception as e: logger.error(f"Error sesión interactiva: {e}", exc_info=True)
            logger.info("Sesión consulta interactiva Neo4j finalizada.")
        elif not processor: logger.error("Consulta omitida: GraphProcessor no instanciado."); print("Error consulta: GP no instanciado.")
        elif not graph: logger.error("Consulta omitida: Neo4j Graph Wrapper no inicializado (¿run_interactive_query era True en Bloque 8?)."); print("Error consulta: Wrapper Neo4j no listo.")
        else: logger.error("Consulta omitida: Falta prompt de consulta."); print("Error consulta: Prompt falta.")
    else:
         logger.info("Consulta interactiva omitida porque 'load_into_neo4j' es False.")
         print("Consulta interactiva omitida (datos no cargados a Neo4j).")
else:
    logger.info("'run_interactive_query' desactivado.")
    print("Modo de consulta interactiva desactivado.")

logger.info("Bloque 11: Fase de consulta interactiva completada (o saltada).")


# In[ ]:


# ============================================================
# %% Bloque 12: Limpieza Final
# ============================================================
logger.info("Bloque 12: Iniciando limpieza final...")
print("\n" + "="*50 + "\n=== Bloque 12: Limpieza Final ===\n" + "="*50)

# Cerrar driver directo Neo4j si se abrió
if 'driver' in locals() and driver and driver.is_open():
    try:
        driver.close(); logger.info("Conexión driver Neo4j cerrada."); print("Conexión directa Neo4j cerrada.")
    except Exception as e: logger.error(f"Error cerrando driver Neo4j: {e}"); print(f"Error al cerrar conexión Neo4j: {e}")
else:
    logger.info("Driver Neo4j ya cerrado o no inicializado.")

# El objeto `graph` (Neo4jGraph wrapper) maneja su pool internamente.

logger.info("Bloque 12: Limpieza final completada.")
print("\nEjecución del script completada.")
print("="*50)


# In[5]:


# ============================================================
# %% Bloque OPCIONAL: Limpieza Completa de la Base de Datos Neo4j
# ============================================================
# ADVERTENCIA: ¡Este bloque BORRARÁ TODOS los nodos y relaciones
# de la base de datos Neo4j especificada en el Bloque 2!
# Ejecuta esta celda SOLO si estás SEGURO de querer limpiar la BD.

logger.warning("Bloque OPCIONAL de Limpieza Total Neo4j iniciado...")
print("\n" + "="*50 + "\n=== Bloque OPCIONAL: Limpieza Completa Neo4j ===\n" + "="*50)
print("¡¡¡ADVERTENCIA MUY SERIA!!!")

# Verificar si las variables de conexión están definidas (del Bloque 2)
if 'NEO4J_URI' not in locals() or 'NEO4J_USERNAME' not in locals() or 'NEO4J_PASSWORD' not in locals():
    logger.error("Variables de conexión Neo4j (NEO4J_URI, _USERNAME, _PASSWORD) no definidas. Ejecuta el Bloque 2 primero.")
    print("ERROR: Faltan variables de conexión Neo4j. Ejecuta el Bloque 2.")
elif not NEO4J_PASSWORD:
     logger.error("Falta NEO4J_PASSWORD. No se puede conectar para limpiar.")
     print("ERROR: Falta la contraseña de Neo4j (NEO4J_PASSWORD).")
else:
    # Conectarse usando el driver directo
    temp_driver: Optional[Driver] = None
    effective_db_name = NEO4J_DATABASE if NEO4J_DATABASE else "neo4j" # Nombre para mostrar
    print(f"\nSe intentará conectar a: {NEO4J_URI} (Usuario: {NEO4J_USERNAME}, BD: {effective_db_name})")
    print(f"Para BORRAR **TODO** su contenido.")

    confirm = input(f"\nPara proceder con el borrado IRREVERSIBLE de la BD '{effective_db_name}', escribe 'SI QUIERO BORRAR TODO': ")

    if confirm == "SI QUIERO BORRAR TODO":
        logger.warning(f"Confirmación recibida. Intentando borrar TODO en BD '{effective_db_name}'...")
        try:
            logger.info("Obteniendo driver temporal para limpieza...")
            # Usar la función get_neo4j_driver definida en Bloque 5
            # Asegúrate que el Bloque 5 se haya ejecutado o copia la función aquí
            if 'get_neo4j_driver' in locals():
                 temp_driver = get_neo4j_driver(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
            else:
                 logger.error("La función 'get_neo4j_driver' no está definida (ejecuta Bloque 5).")
                 print("ERROR: Función 'get_neo4j_driver' no encontrada.")
                 # Podrías copiar la definición aquí como fallback si es necesario

            if temp_driver:
                logger.info(f"Conectado a Neo4j. Procediendo con borrado en BD '{effective_db_name}'...")
                # Usar la función reset_graph_data definida en Bloque 5
                # Esta función ya incluye una confirmación interna adicional
                if 'reset_graph_data' in locals():
                    # La función reset_graph_data pide una SEGUNDA confirmación
                    reset_success = reset_graph_data(temp_driver, db_name=NEO4J_DATABASE)
                    if reset_success:
                        print(f"\n¡BORRADO COMPLETO de BD '{effective_db_name}' realizado!")
                        logger.info(f"Limpieza completa de BD '{effective_db_name}' finalizada exitosamente.")
                    else:
                        print(f"\nBorrado de BD '{effective_db_name}' falló o fue cancelado en la segunda confirmación.")
                        logger.warning(f"Borrado de BD '{effective_db_name}' falló o fue cancelado.")
                else:
                     logger.error("La función 'reset_graph_data' no está definida (ejecuta Bloque 5).")
                     print("ERROR: Función 'reset_graph_data' no encontrada.")

            else:
                logger.error("No se pudo obtener el driver de Neo4j para la limpieza.")
                print("ERROR: No se pudo conectar a Neo4j para limpiar.")

        except Exception as e:
            logger.error(f"Error inesperado durante el proceso de limpieza: {e}", exc_info=True)
            print(f"ERROR inesperado durante la limpieza: {e}")
        finally:
            # Asegurarse de cerrar el driver temporal
            if temp_driver and temp_driver.is_open():
                try:
                    temp_driver.close()
                    logger.info("Driver temporal de limpieza cerrado.")
                except Exception as close_e:
                    logger.error(f"Error cerrando driver temporal: {close_e}")
    else:
        logger.info("Confirmación para borrado completo NO recibida. Operación cancelada.")
        print("\nBorrado completo CANCELADO.")

logger.info("Bloque OPCIONAL de Limpieza Total Neo4j finalizado.")
print("\n" + "="*50)


# In[ ]:




