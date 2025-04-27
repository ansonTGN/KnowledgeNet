Okay, let's revamp the README for a professional GitHub presentation, incorporating more visual elements and structuring it for easy scanning.

---

<h1 align="center">
  <br>
  🧠 Knowledge Graph Extraction Pipeline 🏗️
  <br>
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <!-- Add other badges if applicable (e.g., build status, code coverage) -->
</p>

<p align="center">
  A flexible Python pipeline to extract knowledge graphs (Nodes & Relationships) from text using LLMs (OpenAI, Google, Groq, Ollama), save them as JSON, and optionally load into Neo4j.
</p>

<p align="center">
  <sub>Built with ❤️ by <a href="#author-️">Angel A. Urbina</a></sub>
</p>

<!-- Optional: Add a project logo or banner image here -->
<!-- <p align="center">
  <img src="path/to/your/logo.png" alt="Project Logo" width="300"/>
</p> -->

---

## 🌟 Key Features

*   **📄 Multi-File Processing:** Ingests and processes multiple text documents (`.md`, `.txt`, etc.).
*   **🧩 Semantic Chunking:** Uses Ollama (with enforced JSON output) for reliable, context-aware text splitting before extraction.
*   **🤖 Flexible LLM Integration:** Supports various providers for graph extraction:
    *   OpenAI (GPT models)
    *   Google (Gemini models)
    *   Groq (Llama, Mixtral via fast API)
    *   Ollama (Local models like Gemma, Llama)
*   **💾 Structured JSON Output:** Exports extracted graphs into well-defined JSON files per input document.
*   **🔗 Powerful Neo4j Integration (Optional):**
    *   Connects, optionally cleans (⚠️ **with confirmation**), and loads graph data.
    *   Uses efficient batching and `MERGE` operations (APOC-aware).
    *   Includes utilities for graph inspection (summary, indexes).
*   **💬 Interactive Query Mode:** Chat with your Neo4j graph using natural language via the chosen LLM (after data loading).
*   **🔧 Highly Configurable:** Control LLMs, workflow steps (skip extraction/load), Neo4j options, and verbosity via simple script variables.
*   **🧱 Modular & Jupyter-Friendly:** Code structured in logical blocks, perfect for running cell-by-cell in notebooks.
*   **📝 Robust Logging:** Detailed logs track execution and aid debugging.

---

## 📚 Table of Contents

1.  [🎯 Project Goal](#-project-goal)
2.  [✅ Prerequisites](#-prerequisites)
3.  [⚙️ Installation & Setup](#️-installation--setup)
4.  [🔧 Configuration](#-configuration)
5.  [🚀 How to Run](#-how-to-run)
6.  [🚢 Workflow Steps](#-workflow-steps)
7.  [📄 Output JSON Format](#-output-json-format)
8.  [⚠️ Neo4j Cleanup Warning](#️-neo4j-cleanup-warning)
9.  [🏗️ Script Structure](#️-script-structure)
10. [🤝 Contributing](#-contributing)
11. [📜 License](#-license)
12. [✍️ Author](#️-author)

---

## 🎯 Project Goal <a name="-project-goal"></a>

To provide an automated, configurable, and extensible pipeline for transforming unstructured text into structured knowledge graphs suitable for analysis, querying, and visualization, primarily leveraging LLMs and optionally integrating with Neo4j.

---

## ✅ Prerequisites <a name="-prerequisites"></a>

Before you begin, ensure you have the following:

1.  **Software:**
    *   **Python:** Version 3.8 or higher.
    *   **Neo4j Database (Optional):** Required *only* if using Neo4j features (`load_into_neo4j=True` or `run_interactive_query=True`). Download from [Neo4j Website](https://neo4j.com/download-center/) or use Docker/Aura.
    *   **Ollama (Recommended):** Required for reliable chunking and/or if using Ollama as the primary LLM. Install from [Ollama.ai](https://ollama.ai/). Ensure the service is running and required models are pulled (e.g., `ollama pull gemma3:27b`).

2.  **Python Libraries:** Install the necessary packages:
    ```bash
    pip install langchain langchain-openai langchain-google-genai langchain-groq langchain-experimental langchain-neo4j neo4j pydantic python-dotenv ipywidgets openai tiktoken langchain-core langchain-community --upgrade
    ```
    *(Consider using a virtual environment)*

3.  **API Keys & Credentials (`.env` file):**
    *   Create a file named `.env` in the **root directory** where you run the script.
    *   Populate it with your credentials based on the services you intend to use:

    ```dotenv
    # === Neo4j Credentials (Required if load_into_neo4j=True or run_interactive_query=True) ===
    NEO4J_URI="bolt://localhost:7687"    # Your Neo4j instance URI (e.g., bolt://<IP>:7687 or neo4j+s://<AuraID>.databases.neo4j.io)
    NEO4J_USERNAME="neo4j"               # Your Neo4j username
    NEO4J_PASSWORD="YOUR_NEO4J_PASSWORD" # *** YOUR NEO4J PASSWORD IS REQUIRED ***
    # NEO4J_DATABASE="neo4j"             # Optional: Specify a database name (defaults to 'neo4j')

    # === LLM API Keys (Required based on chosen 'llm_type' in Block 2) ===
    OPENAI_API_KEY="sk-..."              # Required if llm_type = "openai"
    GOOGLE_API_KEY="AIza..."             # Required if llm_type = "google"
    GROQ_API_KEY="gsk_..."               # Required if llm_type = "groq"

    # === Ollama Configuration (Required if using Ollama) ===
    OLLAMA_BASE_URL="http://localhost:11434" # Default URL for local Ollama instance
    ```

    > **IMPORTANT:** The script looks for the `.env` file in the current working directory or its parent directories. Ensure necessary keys/passwords are provided for the features you enable.

---

## ⚙️ Installation & Setup <a name="️-installation--setup"></a>

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt # (Or use the long pip install command above)
    ```
    *(Create a `requirements.txt` file if you prefer)*
3.  **Setup Neo4j (Optional):** Install Neo4j, start the server, and note your URI, username, and password.
4.  **Setup Ollama (Optional):** Install Ollama, run `ollama serve`, and pull models mentioned in the script's configuration (e.g., `ollama pull gemma3:27b`).
5.  **Create `.env` File:** Create the `.env` file in the project root and add your credentials (see [Prerequisites](#-prerequisites)).
6.  **Prepare Data:** Place your input text files (e.g., `.md`) in the `datasets/` folder (or modify `input_filepaths` in the script).

---

## 🔧 Configuration <a name="-configuration"></a>

Adjust the script's behavior by modifying variables within **Block 2 (`KG_AAU_01.py`)**:

*   `input_filepaths`: List of input text files.
*   `output_directory`: Where to save output JSON graphs.
*   `llm_type`: `"openai"`, `"google"`, `"groq"`, or `"ollama"` for main extraction.
*   `*_main_model_name`: Specific model name for the chosen LLM provider.
*   `ollama_chunking_model_name`: Model for semantic chunking (default `gemma3:27b`).
*   `skip_extraction`: `True` to bypass chunking/extraction (e.g., only load).
*   `load_into_neo4j`: `True` to enable Neo4j loading. **Requires Neo4j credentials.**
*   `run_interactive_query`: `True` to enable chat mode **after** loading. Requires `load_into_neo4j=True`.
*   `clear_graph_before_load`: `True` to **DELETE ALL** Neo4j data before loading (⚠️ **prompts for confirmation**).
*   `delete_indexes_before_load`: `True` to drop user indexes before loading (⚠️ **prompts for confirmation**).
*   `show_neo4j_browser`: `True` to auto-open Neo4j Browser post-load.

**Review Block 2 settings carefully before execution!**

---

## 🚀 How to Run <a name="-how-to-run"></a>

1.  **Jupyter Notebook / Lab (Recommended):**
    *   Open `KG_AAU_01.py` directly in Jupyter Lab.
    *   *Or*, create a new `.ipynb` notebook and use `%run KG_AAU_01.py` in a cell.
    *   Execute cells sequentially (Shift+Enter or Run buttons). This allows inspection between steps.

2.  **Standalone Python Script:**
    *   Run from your terminal: `python KG_AAU_01.py`
    *   Interactive confirmation prompts (for Neo4j cleanup) will appear in the terminal.

---

## 🚢 Workflow Steps <a name="-workflow-steps"></a>

The script executes the following main steps (controlled by configuration):

1.  **Setup & Config:** Loads libraries, configs, credentials, and initializes LLMs.
2.  **Neo4j Prep (Optional):** Connects to Neo4j, performs cleanup if configured (data/index deletion **requires explicit confirmation**).
3.  **Extraction:**
    *   Reads input files.
    *   Chunks text semantically using Ollama.
    *   Extracts Nodes/Relationships from chunks using the primary LLM.
    *   Aggregates results across chunks.
    *   Saves the final graph structure to a `.graph.json` file.
    *(Skipped if `skip_extraction=True`)*
4.  **Neo4j Loading (Optional):**
    *   Loads data from the generated JSON file into the configured Neo4j database.
    *   Displays a summary of the graph post-load.
    *   Optionally opens Neo4j Browser.
    *(Skipped if `load_into_neo4j=False`)*
5.  **Interactive Query (Optional):**
    *   Starts a chat interface in the console/notebook.
    *   Allows asking natural language questions about the graph data in Neo4j, answered by the primary LLM.
    *   Type `exit` or `quit` to end the session.
    *(Skipped if `run_interactive_query=False`)*
6.  **Cleanup:** Closes the Neo4j connection.

---

## 📄 Output JSON Format <a name="-output-json-format"></a>

Generated `.graph.json` files follow this structure:

```json
{
  "nodes": [
    {
      "id": "unique_canonical_node_id_1", // Unique identifier (e.g., normalized name)
      "label": "PascalCaseLabel",         // Node type (e.g., Person, Organization)
      "properties": {                     // Additional key-value properties from text
        "property_key_lowercase": "value",
        // ... other properties
      }
    }
    // ... more nodes
  ],
  "relationships": [
    {
      "source": "source_node_id",      // 'id' of the starting node
      "target": "target_node_id",      // 'id' of the ending node
      "type": "RELATIONSHIP_TYPE_VERB", // Type of connection (e.g., WORKS_AT)
      "properties": {                  // Optional properties of the relationship
        "role": "Developer"
        // ... other properties
      }
    }
    // ... more relationships
  ]
}
```

---

## ⚠️ Neo4j Cleanup Warning <a name="️-neo4j-cleanup-warning"></a>

The script includes options (`clear_graph_before_load`, `delete_indexes_before_load`) and a separate **Optional Block (Block 13)** that can **DELETE data or indexes** from your Neo4j database.

*   🛑 These actions are **IRREVERSIBLE**.
*   🛑 You **WILL BE PROMPTED** for explicit confirmation (`SI QUIERO BORRAR TODO` or similar) before any deletion occurs during the standard workflow or via Block 13.
*   🛑 **Use these features with extreme caution and only if you are certain you understand the consequences.**

---

## 🏗️ Script Structure <a name="️-script-structure"></a>

The `KG_AAU_01.py` file is organized into logical blocks:

*   **Blocks 1-3:** Imports, Configuration, Pydantic Models
*   **Block 4:** Primary LLM Initialization
*   **Block 5:** Neo4j Utility Functions
*   **Block 6:** Prompt Templates
*   **Block 7:** `GraphProcessor` Class (Core Logic)
*   **Block 8:** Optional Neo4j Setup & Cleanup Logic
*   **Block 9:** Execution: Extraction to JSON
*   **Block 10:** Execution: Optional Loading to Neo4j
*   **Block 11:** Execution: Optional Interactive Query
*   **Block 12:** Final Cleanup
*   **Block 13 (Optional):** ⚠️ Standalone Neo4j Database Wipe Utility

---

## 🤝 Contributing <a name="-contributing"></a>

Contributions are welcome! If you have suggestions or improvements, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

Please ensure your code adheres to basic Python standards and includes relevant documentation or comments.

---

## 📜 License <a name="-license"></a>

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details, or view the text below:

<details>
<summary>Click to view MIT License</summary>

```
MIT License

Copyright (c) 2024 Angel A. Urbina

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

</details>

---

## ✍️ Author <a name="️-author"></a>

*   **Angel A. Urbina**

---