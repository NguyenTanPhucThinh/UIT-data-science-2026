from typing import List, Set, Union
import math

def calculate_recall(retrieved_ids: List[str], gold_ids: Union[List[str], Set[str]], k: int = 5) -> float:
    """
    Calculates Recall@K. Returns 1.0 if any gold document is found within the top K.
    """
    top_k = retrieved_ids[:k]
    gold_set = set(gold_ids)
    
    for doc_id in top_k:
        if doc_id in gold_set:
            return 1.0
    return 0.0

def calculate_mrr(retrieved_ids: List[str], gold_ids: Union[List[str], Set[str]], k: int = 5) -> float:
    """
    Calculates Mean Reciprocal Rank (MRR@K).
    """
    top_k = retrieved_ids[:k]
    gold_set = set(gold_ids)
    
    for rank, doc_id in enumerate(top_k, 1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0

def calculate_ndcg(retrieved_ids: List[str], gold_ids: Union[List[str], Set[str]], k: int = 5) -> float:
    """
    Calculates NDCG@K (Normalized Discounted Cumulative Gain).
    Assumes binary relevance (1 if in gold_ids, 0 otherwise).
    """
    top_k = retrieved_ids[:k]
    gold_set = set(gold_ids)
    
    dcg = 0.0
    for i, doc_id in enumerate(top_k, 1):
        if doc_id in gold_set:
            dcg += 1.0 / math.log2(i + 1)
            
    # For binary relevance and 1 true target per question (mostly), IDCG is 1.0
    # (Because the best possible ranking has the gold doc at index 1 -> 1/log2(2) = 1)
    # If there are multiple gold docs, IDCG is sum of 1/log2(i+1) for i=1..min(len(gold_set), k)
    idcg = 0.0
    for i in range(1, min(len(gold_set), k) + 1):
        idcg += 1.0 / math.log2(i + 1)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg
