from sentence_transformers import SentenceTransformer
import faiss, numpy as np


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

snippets = [
    "Ontario RTA section 14 – A provision in a tenancy agreement prohibiting pets is void.",
    "BC Guideline 37 – Late payment fees must be reasonable and cannot exceed $25.",
    "Ontario RTA section 13 – A landlord may accept but cannot demand post‑dated cheques.",
]

embeddings = model.encode(snippets).astype("float32")

faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

query = "My lease says no pets allowed"

q_vec = model.encode([query]).astype("float32")
faiss.normalize_L2(q_vec)
D, I = index.search(q_vec, 2)
print("Distances:", D)
print("Indices:", I)

print("Top match:", snippets[I[0][0]], "with distance", D[0][0])
