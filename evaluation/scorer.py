from sentence_transformers import SentenceTransformer, util

# Load the model lazily
_similarity_model = None

def get_similarity_model():
    global _similarity_model
    if _similarity_model is None:
        _similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _similarity_model

def calculate_robustness(clean_accuracy: float, attacked_accuracy: float) -> tuple[float, float]:
    """
    Computes Robustness Score and Drop Rate.
    Returns: (robustness_score_percentage, drop_rate_percentage)
    """
    if clean_accuracy == 0:
        return 0.0, 0.0
        
    drop_rate = clean_accuracy - attacked_accuracy
    robustness = (attacked_accuracy / clean_accuracy) * 100.0
    
    return robustness, drop_rate

def evaluate_custom_prompt(original_response: str, attacked_response: str) -> float:
    """
    Calculates the semantic consistency score between original and attacked responses.
    """
    if not original_response or not attacked_response:
        return 0.0
        
    model = get_similarity_model()
    embeddings1 = model.encode(original_response, convert_to_tensor=True)
    embeddings2 = model.encode(attacked_response, convert_to_tensor=True)
    
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    return float(cosine_scores[0][0])
