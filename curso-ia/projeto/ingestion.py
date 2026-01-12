import os
import uuid
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from dotenv import load_dotenv

load_dotenv()

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
COLLECTION_NAME = "financial"
FILE_PATH = "./AAPL_10-K_1A_temp.md"

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Delete collection if it exists
qdrant.delete_collection(COLLECTION_NAME)

# Create collection
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
    },
    sparse_vectors_config={"sparse": models.SparseVectorParams()},
)

# Read and process file
from markdown_it.rules_block.paragraph import paragraph

with open(FILE_PATH, "r", encoding="utf-8") as f:
    content = f.read()

paragraphs = content.split("\n\n")
chunks = [p.strip() for p in paragraphs if len(p.strip()) > 50]

# Generate embeddings and upload to Qdrant
dense_model = TextEmbedding(DENSE_MODEL)
sparse_model = SparseTextEmbedding(SPARSE_MODEL)

points = []
for chunk in chunks:
    dense_embedding = list(dense_model.passage_embed([chunk]))[0].tolist()
    sparse_embedding = list(sparse_model.passage_embed([chunk]))[0].as_object()

    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "dense": dense_embedding,
            "sparse": sparse_embedding,
        },
        payload={"text": chunk, "source": FILE_PATH},
    )
    points.append(point)

qdrant.upload_points(collection_name=COLLECTION_NAME, points=points)

# Example query
query_text = "What are the main financial risks?"
query_dense = list(dense_model.query_embed([query_text]))[0].tolist()
query_sparse = list(sparse_model.query_embed([query_text]))[0].as_object()

results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        {"query": query_dense, "using": "dense", "limit": 10},
        {"query": query_sparse, "using": "sparse", "limit": 10},
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=3,
)

# Print results
for r in results.points:
    print(f"Score: {r.score}")
    print(f"Texto: {r.payload['text'][:100]}...")
    print("-" * 80)
