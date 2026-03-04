"""We have a text and we want to create its embedding and upload BOTH the og text and the embedding into Cosmos DB
1st - creates embeddings for 'text'
2nd - create an item which holds both text and embedding values and upload it into Cosmos DB

Azure resources needed:
    - Azure Foundry with a deployed 'text-embedding-3-large' model
    - Cosmos DB instance with a database and a container 

Config:
    - review the folder set to hold the knowledge base: "./Rag/knowledge_files/"
    - review the model used for embeddings "text-embedding-3-large" and check it is deployed in your Azure Foundry
    - set FOUNDRY_URL as system var. This is the URL you get when you deploy models in Azure Foundry
    - set FOUNDRY_KEY
    - set COSMOSDB_URL
    - set COSMOSDB_URL
    - set COSMOSDB_KEY
I run this through poetry with 'poetry run x'
"""
from sys import api_version
from openai import AzureOpenAI
from azure.cosmos import CosmosClient
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

def createEmbeddingsClient(endpoint, key):
    return AzureOpenAI(
        api_key=key,
        azure_endpoint=endpoint,
        api_version="2023-05-15",
        azure_deployment="text-embedding-3-large",
    )

def createEmbeddingsBatch(client, texts):
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-large"
    )
    return [item.embedding for item in response.data]

def getCosmosContainer(endpoint, key, db_name, container_name):
    client = CosmosClient(endpoint, key)
    database = client.get_database_client(db_name)
    return database.get_container_client(container_name)

def uploadToCosmosDB(container, item):
    try:
        container.upsert_item(item)
    except Exception as e:
        return f"Error uploading item: {e}"

def requireEnvVar(name):
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(f"{name} is not set")
    return value

# TODO: check and think how to check if documents have already been uploaded to CosmosDB. I need to avoid duplicates and stale information. 
def main():
    foundry_url = requireEnvVar('FOUNDRY_URL') # URL for your Azure Foundry with deployed models. e.g., "https://xxx.openai.azure.com/"
    foundry_key = requireEnvVar('FOUNDRY_KEY')
    cosmosdb_url = requireEnvVar('COSMOSDB_URL') # URL for your Cosmos DB instance. e.g., "https://xxx.documents.azure.com:443/"
    cosmosdb_key = requireEnvVar('COSMOSDB_KEY')

    # load all PDFs we want to use for the RAG
    # TODO: review this loader to check it works
    loader = PyPDFDirectoryLoader("./Rag/knowledge_files/")
    docs = loader.load()
    if not docs:
        print("Error: No documents found in the specified directory. Please add some PDF files to './Rag/knowledge_files/' and try again.")
        return

    # chunk the text of the PDFs into smaller pieces
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # clean all chunks up-front
    cleaned_texts = []
    for chunk in chunks:
        cleaned = chunk.page_content.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        cleaned_texts.append(" ".join(cleaned.split()))

    # create embeddings in batches (single client, fewer API round-trips)
    BATCH_SIZE = 100
    embeddings_client = createEmbeddingsClient(foundry_url, foundry_key)
    all_embeddings = []
    for i in range(0, len(cleaned_texts), BATCH_SIZE):
        batch = cleaned_texts[i:i + BATCH_SIZE]
        print(f"Embedding batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)...")
        all_embeddings.extend(createEmbeddingsBatch(embeddings_client, batch))

    # build items from results
    cosmos_container = getCosmosContainer(cosmosdb_url, cosmosdb_key, db_name="vectorial_ddbb_poc", container_name="container_for_vectors")
    for idx, (cleaned_text, embedding) in enumerate(zip(cleaned_texts, all_embeddings), start=1):
        item = {
            "id": str(idx),
            "original_text": cleaned_text,
            "vectors": embedding
        }
        uploadToCosmosDB(cosmos_container, item=item)
