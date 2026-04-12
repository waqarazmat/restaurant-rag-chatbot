import json
import os
import time
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()  

# 1. API Keys Check
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY") 

if not PINECONE_API_KEY:
    raise ValueError("❌ Please set PINECONE_API_KEY in your .env file or terminal!")
if not COHERE_API_KEY:
    raise ValueError("❌ Please set COHERE_API_KEY in your .env file or terminal!")

# 2. JSON Menu File Load karein
menu_file_name = "pdf_menu_translated.json"
try:
    with open(menu_file_name, "r", encoding="utf-8") as f:
        menu_data = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: '{menu_file_name}' file folder me nahi mili. Pehle file yahan copy karein!")
    exit()

# 3. Text Preparation
# JSON ke data ko clean text mein convert karte hain taake chunks ban sakein
print(f"📄 Reading and preparing data from {menu_file_name}...")

# Agar JSON list hai to hum har item ko join kar lenge, agar dictionary hai to string bana lenge
if isinstance(menu_data, list):
    full_text = "\n\n".join([json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item) for item in menu_data])
else:
    full_text = json.dumps(menu_data, ensure_ascii=False, indent=2)

# Sentence-aware splitting: chunks respect sentence boundaries, with overlap for context
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
)
chunks = splitter.split_text(full_text)
print(f"Created {len(chunks)} chunks from the translated menu.")

# 4. Pinecone Database Initialization (No Deletion!)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "restaurant-rag"

# Hum index ko delete nahi kar rahe, direct use kar rahe hain
index = pc.Index(index_name)

# 5. Cohere Embeddings (Same model as before)
print("🧠 Initializing Cohere Embeddings...")
embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=COHERE_API_KEY)

# 6. Uploading Menu Vectors
print("📤 Uploading Menu vectors to Pinecone cloud...")
vectors_to_upsert = []

for i, chunk in enumerate(chunks):
    vector = embeddings.embed_query(chunk)

    vectors_to_upsert.append({
        "id": f"menu-chunk-{i}",
        "values": vector,
        "metadata": {
            "text": chunk,
            "url": "pdf_menu_translated",
            "header": "Translated PDF Menu"
        }
    })

    # Stay under Cohere trial limit of 100 calls/min
    time.sleep(0.7)
    if (i + 1) % 10 == 0:
        print(f"  Embedded {i + 1}/{len(chunks)} chunks...")

# Data push karo
index.upsert(vectors=vectors_to_upsert)
    
print("\n🎉 Mubarak ho! Translated Menu bhi Pinecone me successfully add ho gaya hai.")