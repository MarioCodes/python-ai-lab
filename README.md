<div align="center">
  
# Python AI Lab
 
**Hands-on Python examples for building AI apps with Azure, OpenAI, LangChain, RAG, and AI Agents**
 
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Azure](https://img.shields.io/badge/Azure_AI_Foundry-0078D4?logo=microsoft-azure&logoColor=white)](https://azure.microsoft.com/en-us/products/ai-studio)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-GPL-yellow.svg)](LICENSE)
[![Poetry](https://img.shields.io/badge/Poetry-managed-blue?logo=poetry)](https://python-poetry.org/)
 
---
 
A collection of practical, runnable Python examples that walk you through core AI concepts — from your first OpenAI API call all the way to building RAG pipelines with Azure Cosmos DB and deploying AI agents with Microsoft Agent Framework. Each module is self-contained, well-documented, and ready to run with `poetry run`.
 
[Getting Started](#getting-started) · [Modules](#modules) · [Architecture](#architecture) · [Contributing](#contributing)
 
</div>

---
 
## Modules
 
### OpenAI API Basics
 
Direct interaction with the OpenAI chat completions API. Sends the same prompt 5 times to compare response variance — a practical way to understand model temperature and non-determinism.
 
```bash
poetry run call-to-openai
```
 
### LangChain
 
Two progressively complex examples showing how LangChain abstracts LLM interactions.
 
| Script | What it covers | Command |
|--------|---------------|---------|
| `langchain_basics.py` | Messages, system prompts, basic invocation | `poetry run langchain-basics` |
| `langchain_lcel.py` | LangChain Expression Language (LCEL) chains | `poetry run langchain-lcel` |
 
### Vector Databases (Azure Cosmos DB)
 
End-to-end workflow for storing and querying vector embeddings using Azure Cosmos DB as a vector database.
 
1. **Generate embeddings** — uses Azure OpenAI's `text-embedding-3-large` model to create vector representations of text
2. **Store in Cosmos DB** — uploads both the original text and its embedding as a single document
3. **Query with vector search** — performs server-side `VectorDistance` similarity search
```bash
poetry run insert-embeddings    # Step 1-2: embed + store
poetry run query-embeddings     # Step 3: similarity search
```
 
### RAG (Retrieval-Augmented Generation)
 
A full RAG pipeline that lets you ask questions against your own documents, powered by Azure Cosmos DB vector search and LangChain.
 
1. **Setup** — creates a Cosmos DB container configured for vector indexing
2. **Ingest** — reads PDFs, chunks them, generates embeddings, and stores everything in Cosmos DB
3. **Query** — embeds your question, retrieves the most relevant chunks via vector search, and uses GPT-4 with the retrieved context to answer
```bash
poetry run setup-cosmosdb-for-rag   # One-time container setup
poetry run insert-rag-files          # Ingest your documents
poetry run ask-rag-question          # Ask questions against your docs
```
 
### AI Agent Framework (Microsoft Agent Framework)
 
Getting started with Microsoft's Agent Framework for building AI agents on Azure AI Foundry. Includes both synchronous and streaming patterns.
 
```bash
poetry run agent-run          # Non-streaming (full response at once)
poetry run agent-run-stream   # Streaming (token-by-token output)
```
 
---
 
## Architecture
 
```
python-ai-lab/
├── openai_lab/              # Direct OpenAI API usage
├── langchain_lab/           # LangChain basics + LCEL
├── vector_databases/        # Cosmos DB vector embeddings
├── rag/                     # Full RAG pipeline
├── agent_framework/         # Microsoft Agent Framework
├── pyproject.toml           # Poetry config & script aliases
└── python-ai-lab.slnx      # Visual Studio solution (optional)
```
 
---
 
## Getting Started
 
### Prerequisites
 
- **Python 3.13+**
- **Poetry** — [install guide](https://python-poetry.org/docs/#installation)
- **Azure subscription** (for Cosmos DB, Azure OpenAI / AI Foundry modules)
- **OpenAI API key** (for the direct OpenAI module)
### Installation
 
```bash
git clone https://github.com/MarioCodes/python-ai-lab.git
cd python-ai-lab
poetry install
```
 
### Configuration
 
Each module uses environment variables for secrets. Set them before running:
 
```bash
# OpenAI (direct API)
export OPENAI_API_KEY=your-key
 
# Azure AI Foundry (RAG, vector DBs, agents)
export FOUNDRY_URL=https://your-project.services.ai.azure.com/...
export FOUNDRY_KEY=your-key
 
# Azure Cosmos DB
export COSMOSDB_URL=https://your-account.documents.azure.com:443/
export COSMOSDB_KEY=your-key
 
# Agent Framework
export FOUNDRY_POC_URL=https://your-foundry-project.services.ai.azure.com/api/projects/your-project
```
 
Then run any module with the corresponding `poetry run` command listed above.
 
---
 
## Tech Stack
 
| Category | Technology |
|----------|-----------|
| Language | Python 3.13 |
| Package Manager | Poetry |
| LLMs | OpenAI GPT-5, GPT-4, Azure OpenAI |
| Embeddings | `text-embedding-3-large` (Azure) |
| Orchestration | LangChain, LCEL |
| Vector Database | Azure Cosmos DB (NoSQL + vector search) |
| AI Agents | Microsoft Agent Framework |
| Auth | Azure CLI Credential |
 
---
 
## Contributing
 
Contributions are welcome. If you'd like to add a new AI module or improve an existing one, feel free to open a PR.
 
---
 
## License
 
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
 
---
 
<div align="center">
If you find this useful, consider giving it a star — it helps others discover these examples.
</div>
