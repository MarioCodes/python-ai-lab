"""We have a knowledge base which we want to get the vectors for it and upload BOTH the og text and the vectors into Cosmos DB
1st - read files and chunk them using semantic chunking
2nd - clean up the text of the chunks
3rd - create embeddings for the chunks in batches (to reduce API round-trips)
4rth - insert into Cosmos DB the original text, the embeddings and some metadata for each chunk as a single item

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
import os
import configparser
from openai import AzureOpenAI
from azure.cosmos import CosmosClient
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

# Retrieve global configs
config = configparser.ConfigParser()
config.read("settings.ini")

foundry_url = os.environ["FOUNDRY_URL"]
foundry_key = os.environ["FOUNDRY_KEY"]
cosmosdb_url = os.environ["COSMOSDB_URL"]
cosmosdb_key = os.environ["COSMOSDB_KEY"]
# /Retrieve global config

def createEmbeddingsBatch(batchs):
    client = AzureOpenAI(
        azure_endpoint=foundry_url,
        api_key=foundry_key,
        api_version=config["embedding_model"]["api_version"],
        azure_deployment=config["embedding_model"]["model"],
    )
    response = client.embeddings.create(
        input=batchs,
        model=config["embedding_model"]["model"]
    )
    return [item.embedding for item in response.data]

def getCosmosContainer(db_name, container_name):
    client = CosmosClient(cosmosdb_url, cosmosdb_key)
    database = client.get_database_client(db_name)
    return database.get_container_client(container_name)

def uploadToCosmosDB(container, item):
    try:
        container.upsert_item(item)
    except Exception as e:
        return f"Error uploading item: {e}"

def loadDocuments():
    loader = PyPDFDirectoryLoader("./Rag/knowledge_files/")
    docs = loader.load()
    if not docs:
        print("Error: No documents found in the specified directory. Please add some PDF files to './Rag/knowledge_files/' and try again.")
        return

    print(f"\nTOTAL DOCUMENTS LOADED: {len(docs)}\n")
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        print(f"Document: {source} (page {doc.metadata.get('page', '?')+1})")
        print("Raw document's content:")
        print(f"{repr(doc.page_content)}\n")
    return docs

def fixedSizeChunking(docs):
    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = chunk_splitter.split_documents(docs)

def semanticChunking(docs):
    embeddings_model_chunking = AzureOpenAIEmbeddings(
        azure_endpoint=foundry_url,
        api_key=foundry_key,
        api_version=config["embedding_model"]["api_version"],
        azure_deployment=config["embedding_model"]["model"],
    )
    semantic_chunker = SemanticChunker(embeddings_model_chunking, breakpoint_threshold_type="percentile")
    semantic_chunks = semantic_chunker.create_documents(
        [d.page_content for d in docs],
        metadatas=[d.metadata for d in docs]
    )
    for semantic_chunk in semantic_chunks:
        print("\n--- SEMANTIC CHUNK ---")
        print(f"Chunk's length in characters: {len(semantic_chunk.page_content)}")
        print(f"Chunk's' content: \n{repr(semantic_chunk.page_content)}")
    print(f"TOTAL CHUNKS CREATED: {len(semantic_chunks)}")
    return semantic_chunks

def cleanChunks(chunks):
    clean_chunks = []
    chunk_metadata = []
    for chunk in chunks:
        cleaned = chunk.page_content.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            continue

        clean_chunks.append(cleaned)

        source_path = chunk.metadata.get("source", "")
        document_name = os.path.basename(source_path) if source_path else "unknown"
        document_date = datetime.fromtimestamp(os.path.getmtime(source_path)).isoformat() if source_path and os.path.exists(source_path) else "unknown"
        chunk_metadata.append({"document_name": document_name, "document_date": document_date})

    print("\n\nCLEANED CHUNK TEXTS:")
    for clean_chunk in clean_chunks:
        print(f"{repr(clean_chunk)}\n")
    print(f"TOTAL CHUNKS CLEANED: {len(clean_chunks)}")
    return clean_chunks, chunk_metadata

# TODO: check and think how to check if documents have already been uploaded to CosmosDB. I need to avoid duplicates and stale information
def main():
    # load all PDFs we want to use for the RAG - each page as its own document
    docs = loadDocuments()
    if not docs:
        return

    # Chunking
    #   (old) naive char-fixed size chunking
    #       this chunks the text of the PDFs into smaller pieces with an overlap, but has the drawback of potentially cutting sentences in half or topics in multiple chunks
    # chunks = fixedSizeChunking(docs)

    #   (newest) semantic chunking - this approach uses the Azure OpenAI embeddings to create chunks that are semantically meaningful
    #       It can help keep sentences and topics together, but it requires more API calls to create chunks
    chunks = semanticChunking(docs)

    # clean all chunks up-front and extract document metadata
    #   we need to clean all chunks as they come with chars such as '\t' or '\n'. Sometimes chunks can be only empty spaces and we need to clean them or they break the embedding model
    clean_values = cleanChunks(chunks)
    clean_chunks = clean_values[0]
    chunk_metadata = clean_values[1]

    # create embeddings in batches
    BATCH_SIZE = 100
    chunk_embeddings = []
    for i in range(0, len(clean_chunks), BATCH_SIZE):
        batch = clean_chunks[i:i + BATCH_SIZE]
        batch_number = i // BATCH_SIZE + 1
        print(f"Embedding batch {batch_number} ({len(batch)} chunks)...")
        chunk_embeddings.extend(createEmbeddingsBatch(batch))

    # upload embeddings to Cosmos DB
    cosmos_container = getCosmosContainer(db_name="vectorial_ddbb_poc", container_name="container_for_vectors")
    total_chunks = len(clean_chunks)
    for i in range(total_chunks):
        print(f"Uploading chunk {i + 1}/{total_chunks}...")
        item = {
            "id": str(i + 1),
            "original_text": clean_chunks[i],
            "embeddings": chunk_embeddings[i],
            "document_name": chunk_metadata[i]["document_name"],
            "document_date": chunk_metadata[i]["document_date"]
        }
        uploadToCosmosDB(cosmos_container, item=item)
