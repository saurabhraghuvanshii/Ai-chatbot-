from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from typing import List, Dict
import numpy as np

class JenkinsNLPModel:
    def __init__(self):
        # Initialize with a suitable pre-trained model for embeddings
        self.model_name = "sentence-transformers/all-mpnet-base-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Initialize text generation model
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=200
        )
        
    def encode_text(self, text: str) -> np.ndarray:
        """Convert text to vector embeddings"""
        # Tokenize and encode text
        inputs = self.tokenizer(text, return_tensors="pt", 
                              truncation=True, max_length=512,
                              padding=True)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.numpy()

    def compute_similarity(self, query_embedding: np.ndarray, 
                         doc_embeddings: List[np.ndarray]) -> List[float]:
        """Compute similarity scores between query and documents"""
        # Convert to unit vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = [doc / np.linalg.norm(doc) for doc in doc_embeddings]
        
        # Compute cosine similarity
        similarities = [np.dot(query_norm, doc.T)[0][0] for doc in doc_norms]
        return similarities

    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate response based on query and relevant context"""
        # Prepare prompt with context
        prompt = f"""
        Context: {context[0][:1000]}...
        
        Question: {query}
        
        Answer: Let me help you with that.
        """
        
        # Generate response
        response = self.generator(prompt)[0]['generated_text']
        
        return response

def main():
    # Test the model
    model = JenkinsNLPModel()
    test_text = "How to create a Jenkins pipeline?"
    embeddings = model.encode_text(test_text)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Test response generation
    context = ["Jenkins Pipeline is a suite of plugins that supports implementing and integrating continuous delivery pipelines into Jenkins. A pipeline is a sequence of steps that tells Jenkins what to do."]
    response = model.generate_response(test_text, context)
    print(f"Generated response: {response}")

if __name__ == "__main__":
    main()