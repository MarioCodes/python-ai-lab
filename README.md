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
 
A collection of practical, runnable Python examples that walk you through core AI concepts (e.g. OpenAI API calls, RAG with Azure Cosmos DB, AI agents with Microsoft Agent Framework. Each module is self-contained and ready to run with `poetry run`
 
[Modules](#modules) · [Getting Started](#getting-started)
 
</div>

---
 
## Modules
 
### OpenAI API Basics
 
Direct interaction with the OpenAI chat completions API. Sends the same prompt 5 times to compare response variance. A practical way to understand model temperature and non-determinism.
 
~~~ bash
poetry run call-to-openai
~~~
 
### LangChain
 
Two examples showing how LangChain abstracts LLM interactions.

~~~ bash
poetry run langchain-basics # invocation, messages, system prompts
poetry run langchain-lcel # LCEL (LangChain Expression Language) chains
~~~
 
### Vector Databases (Azure Cosmos DB)
 
End-to-end workflow for storing and querying vector embeddings using Azure Cosmos DB as a vector database.
 
1. **Generate embeddings** - uses Azure OpenAI's `text-embedding-3-large` model to create vector representations of text
2. **Store in Cosmos DB** - uploads both the original text and its embedding as a single document
3. **Query with vector search** - performs server-side `VectorDistance` similarity search

~~~ bash
poetry run insert-embeddings    # Step 1-2: embed + store
poetry run query-embeddings     # Step 3: similarity search
~~~
 
### RAG (Retrieval-Augmented Generation)
 
A full RAG pipeline that lets you ask questions against your own documents, powered by Azure Cosmos DB vector search and LangChain.
 
1. **Setup** - creates a Cosmos DB container configured for vector indexing
2. **Ingest** - reads PDFs, chunks them, generates embeddings, and stores everything in Cosmos DB
3. **Query** - embeds your question, retrieves the most relevant chunks via vector search, and uses GPT-4 with the retrieved context to answer

~~~ bash
poetry run setup-cosmosdb-for-rag   # One-time container setup
poetry run insert-rag-files          # Ingest your documents
poetry run ask-rag-question          # Ask questions against your docs
~~~ 
 
### AI Agent Framework (Microsoft Agent Framework)
 
Getting started with Microsoft's Agent Framework for building AI agents on Azure AI Foundry. Includes both synchronous and streaming patterns.
 
~~~ bash
poetry run agent-run          # Non-streaming (full response at once)
poetry run agent-run-stream   # Streaming (token-by-token output)
~~~ 
 
---
 
## Getting Started
 
### Prerequisites
 
- **Python 3.13+**
- **Poetry**
- **Azure subscription**
- **OpenAI API key**
### Installation
 
~~~ bash
git clone https://github.com/MarioCodes/python-ai-lab.git
cd python-ai-lab
poetry install
~~~ 
 
### Configuration
 
Each module uses environment variables for secrets. Set them before running:
 
~~~ bash
# OpenAI (direct API)
export OPENAI_API_KEY=your-key
 
# Azure AI Foundry (RAG, vector DBs, agents) & Agent Framework
export FOUNDRY_URL=https://your-project.services.ai.azure.com/...
export FOUNDRY_KEY=your-key
 
# Azure Cosmos DB
export COSMOSDB_URL=https://your-account.documents.azure.com:443/
export COSMOSDB_KEY=your-key
 
# Agent Framework
export FOUNDRY_POC_URL=https://your-foundry-project.services.ai.azure.com/api/projects/your-project
~~~
 
Then run any module with the corresponding `poetry run` command listed above.
 
---
 
<div align="center">
If you find this useful consider giving it a star
</div>
