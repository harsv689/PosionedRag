import os
import json
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from beir.datasets.data_loader import GenericDataLoader
from src.utils import load_models
import pickle
from sentence_transformers import util as st_util

os.environ['TRANSFORMERS_CACHE'] = '/scratch0/sghosal2/.cache/'
os.environ['HF_HOME'] = '/scratch0/sghosal2/.cache/huggingface/'

# Load BEIR NQ dataset
dataset = "nq"
out_dir = os.path.join(os.getcwd(), "datasets")
data_path = os.path.join(out_dir, dataset)
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

# Load nq.json (Adversarial Dataset)
nq_json_path = "results/adv_targeted_results/nq.json"
with open(nq_json_path, "r") as f:
    adversarial_data = json.load(f)

# Load Contriever Retrieval Model
retrieval_model, _, tokenizer, get_emb = load_models("contriever")
retrieval_model.eval().cuda()

# Load or Compute Corpus Embeddings
embeddings_path = "corpus_embeddings.pkl"
if os.path.exists(embeddings_path):
    print("Loading from saved embeddings")
    with open(embeddings_path, 'rb') as f:
        corpus_embeddings = pickle.load(f)
else:
    corpus_embeddings = {}
    for doc_id, doc in corpus.items():
        doc_text = doc["text"]
        doc_input = tokenizer(doc_text, padding=True, truncation=True, return_tensors="pt")
        doc_input = {key: value.cuda() for key, value in doc_input.items()}
        with torch.no_grad():
            doc_embedding = get_emb(retrieval_model, doc_input)
            corpus_embeddings[doc_id] = doc_embedding.cpu()
    with open(embeddings_path, 'wb') as f:
        pickle.dump(corpus_embeddings, f)

# Process first 20 samples for visualization
num_samples = 100
K = 10

corpus_corpus_sims = []
corpus_adv_sims = []
adv_adv_sims = []

sample_keys = list(adversarial_data.keys()) #[:num_samples]
print(len(sample_keys))
count = 0
for key in sample_keys:
    count +=1
    if count % 10 == 0:
        print(f"{count} samples evaluated")
    data = adversarial_data[key]
    question = data["question"]
    adversarial_texts = data["adv_texts"]

    with torch.no_grad():
        query_input = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
        query_input = {key: value.cuda() for key, value in query_input.items()}
        query_embedding = get_emb(retrieval_model, query_input)

    adv_embeddings = []
    for text in adversarial_texts:
        adv_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        adv_input = {key: value.cuda() for key, value in adv_input.items()}
        with torch.no_grad():
            adv_emb = get_emb(retrieval_model, adv_input)
            adv_embeddings.append(adv_emb.cpu())

    combined_embeddings = list(corpus_embeddings.values()) + adv_embeddings
    embeddings_matrix = torch.cat(combined_embeddings).cuda()

    scores = st_util.cos_sim(query_embedding, embeddings_matrix)[0]
    top_k_indices = torch.topk(scores, K).indices.cpu().numpy()

    top_k_embeddings = [combined_embeddings[idx] for idx in top_k_indices]
    top_k_matrix = torch.cat(top_k_embeddings).cuda()

    similarity_matrix = st_util.cos_sim(top_k_matrix, top_k_matrix).cpu().numpy()

    # Categorize similarities
    for i in range(K):
        for j in range(i + 1, K):
            if top_k_indices[i] < len(corpus_embeddings) and top_k_indices[j] < len(corpus_embeddings):
                corpus_corpus_sims.append(similarity_matrix[i, j])
            elif top_k_indices[i] >= len(corpus_embeddings) and top_k_indices[j] >= len(corpus_embeddings):
                adv_adv_sims.append(similarity_matrix[i, j])
            else:
                corpus_adv_sims.append(similarity_matrix[i, j])

# Plot similarity distributions
plt.figure(figsize=(12, 8))
sns.histplot(corpus_corpus_sims, color="green", label="Corpus-Corpus", kde=True, stat="density", linewidth=0)
sns.histplot(corpus_adv_sims, color="yellow", label="Corpus-Adversarial", kde=True, stat="density", linewidth=0)
sns.histplot(adv_adv_sims, color="blue", label="Adversarial-Adversarial", kde=True, stat="density", linewidth=0)

plt.title("Distribution of Similarities")
plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("dist_sim.png", dpi=100, bbox_inches="tight")
