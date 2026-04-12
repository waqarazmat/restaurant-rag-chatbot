import json
import os
import time
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()  

# 1. API Keys Check
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY") 

if not PINECONE_API_KEY:
    raise ValueError("❌ Please set PINECONE_API_KEY in your .env file or terminal!")
if not COHERE_API_KEY:
    raise ValueError("❌ Please set COHERE_API_KEY in your .env file or terminal!")

# 2. JSON File Load karein
try:
    with open("website_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print("❌ Error: 'website_data.json' file folder me nahi mili. Pehle file yahan copy karein!")
    exit()

# 3. Text Splitter
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# Secondary splitter to break oversized markdown chunks into sentence-aware pieces
size_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
)

all_documents = []

print("✂️ Splitting documents into smart chunks based on headings...")
for page in data:
    url = page.get("url")
    markdown_content = page.get("markdown")

    if "Hello world" in markdown_content and "WordPress" in markdown_content:
        continue

    if markdown_content:
        splits = markdown_splitter.split_text(markdown_content)
        for split in splits:
            split.metadata["url"] = url
            # If a heading-based chunk is still too large, split it further
            if len(split.page_content) > 500:
                sub_splits = size_splitter.create_documents(
                    [split.page_content],
                    metadatas=[split.metadata]
                )
                all_documents.extend(sub_splits)
            else:
                all_documents.append(split)

print(f"Created {len(all_documents)} smart chunks from the website data.")

# 4. Pinecone Database Initialization
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "restaurant-rag"

# ⚠️ CRITICAL: Purane index ko delete kar k Cohere ka (1024 dimension) banana parega
if index_name in pc.list_indexes().names():
    print(f"⚠️ Deleting existing index '{index_name}' for clean start...")
    pc.delete_index(index_name)

print(f"Creating new index: {index_name} with 1024 dimensions...")
pc.create_index(
    name=index_name,
    dimension=1024, # Cohere 'embed-english-v3.0' 1024 dimensions generate karta hai
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index(index_name)

# 5. Cohere Embeddings (100% Stable and Free Trial!)
print("🧠 Initializing Cohere Embeddings...")
embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=COHERE_API_KEY)

# 6. Smart Batch Uploading
print("📤 Uploading vectors to Pinecone cloud in batches...")
vectors_to_upsert = []

for i, doc in enumerate(all_documents):
    text_to_embed = doc.page_content
    vector = embeddings.embed_query(text_to_embed)

    vectors_to_upsert.append({
        "id": f"chunk-{i}",
        "values": vector,
        "metadata": {
            "text": doc.page_content,
            "url": doc.metadata.get("url", ""),
            "header": str(doc.metadata)
        }
    })

    # Stay under Cohere trial limit of 100 calls/min
    time.sleep(0.7)
    if (i + 1) % 10 == 0:
        print(f"  Embedded {i + 1}/{len(all_documents)} chunks...")

# Pura data aik saath push karo
index.upsert(vectors=vectors_to_upsert)
    
print("\n🎉 Mubarak ho! Saara data Cohere embeddings ke zariye Pinecone me store ho gaya hai.")  