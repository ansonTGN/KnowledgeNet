Knowledge Graph Extraction Pipeline 🏗️ <a id="top"></a>

![alt text](https://img.shields.io/badge/License-MIT-yellow.svg)

This Python script provides a flexible pipeline for processing text documents, extracting knowledge graph structures (nodes and relationships) using various Large Language Models (LLMs), saving the results to detailed JSON files, and optionally loading them into a Neo4j graph database.

Designed for modular execution, particularly within Jupyter Notebook environments.

Author: Angel A. Urbina

Table of Contents 📚

Overview

Features ✨

Requirements Checklist ✅

Software Dependencies

Python Libraries

API Keys & Credentials (.env)

Setup Instructions ⚙️

Configuration 🔧

Usage Guide 🚀

Running the Script

Execution Flow

⚠️ Optional Neo4j Cleanup

Output Format 📄

Script Structure (Blocks) 🧱

License 📜

Author ✍️

Overview <a id="overview"></a>

The core goal of this script (KG_AAU_01.py) is to automate the transformation of unstructured text data into structured knowledge graphs. It leverages the power of LLMs for two key tasks:

Semantic Chunking: Breaking down large texts into smaller, meaningful, self-contained units using Ollama's JSON mode for reliability.

Graph Extraction: Identifying entities (nodes) and their connections (relationships) within each chunk using a configurable primary LLM (OpenAI, Google Gemini, Groq, or Ollama).

The extracted graph data is aggregated, saved as JSON, and can be optionally ingested into a Neo4j database for further analysis and querying.

⬆️ Back to Top

Features ✨ <a id="features"></a>

📄 Text File Processing: Reads text content from specified files (.md, .txt, etc.).

🧠 Semantic Chunking: Uses Ollama (with JSON mode enforcement) for robust, context-aware text splitting.

🤖 Multi-LLM Support: Extracts graph structures using:

OpenAI (e.g., GPT-4o-mini, GPT-4)

Google Generative AI (e.g., Gemini 1.5 Flash)

Groq (e.g., Llama3, Mixtral)

Ollama (e.g., Gemma, Llama locally)

📐 Pydantic Schemas: Defines clear data structures (Chunk, Node, Relationship, ExtractedGraph) for reliable input/output parsing.

💾 JSON Output: Saves the aggregated graph structure (nodes & relationships) for each input file into a separate, well-formatted JSON file.

🔗 Optional Neo4j Integration:

Connects to a Neo4j instance.

⚠️ Optional: Clears existing graph data and/or user-defined indexes before loading.

Loads extracted nodes and relationships efficiently using MERGE and batching (APOC support detected automatically).

Provides utility functions for graph inspection (summary, indexes).

Opens Neo4j Browser automatically (optional).

💬 Interactive Query Mode: Allows querying the loaded Neo4j graph using natural language via the configured primary LLM (if data is loaded).

🔧 Configurable Workflow: Easily control which LLM to use, skip steps (extraction, loading), manage Neo4j interactions, and toggle output verbosity via script variables.

🧱 Modular Design: Structured into logical blocks, ideal for execution and modification within Jupyter Notebooks.

📝 Comprehensive Logging: Detailed logging helps track the process and debug issues.

⬆️ Back to Top

Requirements Checklist ✅ <a id="requirements"></a>

Ensure you have the following set up before running the script:

Software Dependencies <a id="software-dependencies"></a>

🐍 Python: Version 3.8 or higher recommended.

🐘 Neo4j Database (Optional): Required only if load_into_neo4j or run_interactive_query is set to True. Download from Neo4j Website or use Docker.

🦙 Ollama (Optional but Recommended): Required for the default semantic chunking (highly recommended for reliability). Also needed if llm_type is set to "ollama". Install from Ollama.ai.

Make sure the Ollama service is running (ollama serve).

Pull the necessary models specified in the configuration (e.g., ollama pull gemma3:27b for the default chunking model).

Python Libraries <a id="python-libraries"></a>

Install the required Python packages. You can run the command provided in the script's first code cell (Block 1):

pip install langchain langchain-openai langchain-google-genai langchain-groq langchain-experimental langchain-neo4j neo4j pydantic python-dotenv ipywidgets openai tiktoken langchain-core langchain-community --upgrade


Alternatively, create a requirements.txt file with the packages listed above and run pip install -r requirements.txt.

API Keys & Credentials (.env) <a id="api-keys--credentials-env"></a>

The script uses a .env file to securely manage sensitive credentials. Create a file named .env in the same directory where you run the script/notebook (or a parent directory).

Expected Variables:

# === Neo4j Credentials (Required if load_into_neo4j=True or run_interactive_query=True) ===
NEO4J_URI="bolt://localhost:7687" # Or your Neo4j instance URI
NEO4J_USERNAME="neo4j"           # Your Neo4j username
NEO4J_PASSWORD="YOUR_NEO4J_PASSWORD" # *** YOUR NEO4J PASSWORD IS REQUIRED ***
# NEO4J_DATABASE="neo4j"         # Optional: Specify a database name (defaults to 'neo4j')

# === LLM API Keys (Required based on chosen 'llm_type') ===
OPENAI_API_KEY="sk-..."          # Required if llm_type = "openai"
GOOGLE_API_KEY="AIza..."         # Required if llm_type = "google"
GROQ_API_KEY="gsk_..."           # Required if llm_type = "groq"

# === Ollama Configuration (Required if using Ollama for chunking or main LLM) ===
OLLAMA_BASE_URL="http://localhost:11434" # Default URL for local Ollama instance
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Dotenv
IGNORE_WHEN_COPYING_END

IMPORTANT:

Fill in YOUR_NEO4J_PASSWORD if you intend to use the Neo4j features.

Provide the API key(s) corresponding to the llm_type you select in the script's configuration (Block 2).

Ensure the .env file is present where the script expects to find it (usually the execution directory).

⬆️ Back to Top

Setup Instructions ⚙️ <a id="setup"></a>

Clone/Download: Get the KG_AAU_01.py script and any associated files (like the datasets folder).

Install Python: Ensure Python 3.8+ is installed.

Install Dependencies: Run the pip install ... command from the Python Libraries section in your terminal/environment.

Setup Neo4j (Optional): If using Neo4j, install it and ensure the database server is running. Note the URI, Username, and Password.

Setup Ollama (Optional): If using Ollama, install it, ensure the service is running, and pull the required models (e.g., ollama pull gemma3:27b). Note the Base URL (usually http://localhost:11434).

Create .env File: Create the .env file in your project's root or execution directory and add t