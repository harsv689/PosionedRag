import argparse
import os
import json
import torch
import numpy as np
from typing import List, Dict
from src.utils import load_beir_datasets, load_models, load_json
from src.utils import setup_seeds

def parse_args():
    parser = argparse.ArgumentParser(description='Embedding Similarity Analysis')
    
    # Retriever and Dataset Arguments
    parser.add_argument("--eval_model_code", type=str, default="contriever")
    parser.add_argument('--eval_dataset', type=str, default="nq")
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--top_k', type=int, default=5)
    
    # Technical Arguments
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    
    # Results and Paths
    parser.add_argument("--orig_beir_results", type=str, default=None)
    
    return parser.parse_args()

def compute_similarities(query_emb: torch.Tensor, 
                         candidate_embs: torch.Tensor, 
                         similarity_type: str = 'cos_sim') -> np.ndarray:
    """
    Compute similarities between query embedding and candidate embeddings.
    
    Args:
        query_emb (torch.Tensor): Query embedding
        candidate_embs (torch.Tensor): Candidate embeddings
        similarity_type (str): Type of similarity computation
    
    Returns:
        np.ndarray: Similarity scores
    """
    if similarity_type == 'dot':
        similarities = torch.mm(query_emb, candidate_embs.T).cpu().numpy()
    elif similarity_type == 'cos_sim':
        similarities = torch.cosine_similarity(query_emb, candidate_embs).cpu().numpy()
    
    return similarities

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup device and seed
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'
    setup_seeds(args.seed)
    
    # Load datasets and incorrect answers
    corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)
    incorrect_answers = load_json(f'results/adv_targeted_results/{args.eval_dataset}.json')
    incorrect_answers = list(incorrect_answers.values())
    
    # Load retrieval models
    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
    model.eval()
    model.to(device)
    c_model.eval()
    c_model.to(device)
    
    # Load BEIR results
    if args.orig_beir_results is None:
        args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}.json"
    
    with open(args.orig_beir_results, 'r') as f:
        results = json.load(f)
    
    # Analysis containers
    similarity_analysis = {}
    
    # Iterate through incorrect answers
    for idx, incorrect_answer in enumerate(incorrect_answers):
        # Get query and its ID
        query = incorrect_answer['question']
        query_id = incorrect_answer['id']
        
        # Get top-k retrieved passages
        top_k_indices = list(results[query_id].keys())[:args.top_k]
        top_k_passages = [corpus[idx]['text'] for idx in top_k_indices]
        
        # Adversarial text to compare
        adv_texts = [incorrect_answer['answer']]  # You might want to modify this
        
        # Tokenize and embed query
        query_input = tokenizer(query, padding=True, truncation=True, return_tensors="pt")
        query_input = {key: value.cuda() for key, value in query_input.items()}
        
        with torch.no_grad():
            query_emb = get_emb(model, query_input)
            
            # Tokenize and embed top-k passages
            top_k_inputs = tokenizer(top_k_passages, padding=True, truncation=True, return_tensors="pt")
            top_k_inputs = {key: value.cuda() for key, value in top_k_inputs.items()}
            top_k_embs = get_emb(model, top_k_inputs)
            
            # Tokenize and embed adversarial texts
            adv_inputs = tokenizer(adv_texts, padding=True, truncation=True, return_tensors="pt")
            adv_inputs = {key: value.cuda() for key, value in adv_inputs.items()}
            adv_embs = get_emb(model, adv_inputs)
        
        # Compute similarities
        top_k_similarities = compute_similarities(query_emb, top_k_embs)
        adv_similarities = compute_similarities(query_emb, adv_embs)
        
        # Store analysis results
        similarity_analysis[query_id] = {
            'query': query,
            'top_k_passages': top_k_passages,
            'top_k_similarities': top_k_similarities.tolist(),
            'adv_text': adv_texts,
            'adv_similarities': adv_similarities.tolist(),
            'mean_top_k_similarity': float(np.mean(top_k_similarities)),
            'mean_adv_similarity': float(np.mean(adv_similarities))
        }
        
        print(f"Analysis for query {idx+1}/{len(incorrect_answers)}")
        print(f"Mean Top-K Similarity: {similarity_analysis[query_id]['mean_top_k_similarity']}")
        print(f"Mean Adversarial Similarity: {similarity_analysis[query_id]['mean_adv_similarity']}\n")
    
    # Save analysis results
    output_path = f'results/embedding_analysis/{args.eval_dataset}_embedding_similarities.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(similarity_analysis, f, indent=2)
    
    print(f"Embedding similarity analysis complete. Results saved to {output_path}")

if __name__ == '__main__':
    main()