import os
import json
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from beir.datasets.data_loader import GenericDataLoader
from src.utils import load_beir_datasets, load_models
from sentence_transformers import util as st_util

# Step 1: Load BEIR NQ dataset
dataset = "nq"
out_dir = os.path.join(os.getcwd(), "datasets")
data_path = os.path.join(out_dir, dataset)

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

# Step 2: Load nq.json (Adversarial Dataset)
nq_json_path = "results/adv_targeted_results/nq.json"
with open(nq_json_path, "r") as f:
    adversarial_data = json.load(f)

# Step 3: Load Contriever Retrieval Model
retrieval_model, _, tokenizer, get_emb = load_models("contriever")  # ✅ Use Contriever
retrieval_model.eval()
retrieval_model.to("cuda")

# Step 4: Retrieve Top K Ground Truth Passages and Compute Embeddings
K = 3  # Number of top retrieved documents
similarity_results = []

for key, data in adversarial_data.items():
    question = data["question"]
    adversarial_texts = data["adv_texts"]

    # Encode the query using Contriever
    with torch.no_grad():
        query_input = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
        query_input = {key: value.cuda() for key, value in query_input.items()}
        query_embedding = get_emb(retrieval_model, query_input)

    # Retrieve top-K documents (similarity search)
    doc_embeddings = {}
    for doc_id, doc in corpus.items():
        doc_text = doc["text"]
        doc_input = tokenizer(doc_text, padding=True, truncation=True, return_tensors="pt")
        doc_input = {key: value.cuda() for key, value in doc_input.items()}
        with torch.no_grad():
            doc_embedding = get_emb(retrieval_model, doc_input)
            doc_embeddings[doc_id] = doc_embedding

    # Sort by similarity to query embedding
    sorted_docs = sorted(doc_embeddings.items(), key=lambda x: torch.cosine_similarity(x[1], query_embedding).item(), reverse=True)
    top_k_texts = [corpus[doc_id]["text"] for doc_id, _ in sorted_docs[:K]]

    # Compute embeddings for ground truth (top-K)
    ground_truth_embeddings = [doc_embeddings[doc_id] for doc_id, _ in sorted_docs[:K]]

    # Compute embeddings for adversarial texts
    adv_embeddings = []
    for text in adversarial_texts:
        adv_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        adv_input = {key: value.cuda() for key, value in adv_input.items()}
        with torch.no_grad():
            adv_embedding = get_emb(retrieval_model, adv_input)
            adv_embeddings.append(adv_embedding)

    # Compute cosine similarity matrix
    similarity_matrix = torch.zeros((len(ground_truth_embeddings), len(adv_embeddings)))

    for i, gt_emb in enumerate(ground_truth_embeddings):
        for j, adv_emb in enumerate(adv_embeddings):
            similarity_matrix[i, j] = torch.cosine_similarity(gt_emb, adv_emb).item()

    # Convert to NumPy array for visualization
    similarity_results.append(similarity_matrix.cpu().numpy())

# Step 5: Plot Heatmap for Similarity Scores
fig, axes = plt.subplots(nrows=1, ncols=len(similarity_results), figsize=(15, 5))

for idx, sim_matrix in enumerate(similarity_results[:3]):  # Plot first 3 examples
    ax = axes[idx]
    sns.heatmap(sim_matrix, annot=True, cmap="coolwarm", ax=ax, cbar=True)
    ax.set_title(f"Question {idx+1}")
    ax.set_xlabel("Adversarial Texts")
    ax.set_ylabel("Ground Truth Passages")

plt.tight_layout()
plt.show()