from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

class VectorDB:
    def __init__(self):
        # Force the model to load on the CPU
        device = torch.device("cpu")
        self.model = SentenceTransformer("all-mpnet-base-v2", device='cpu')
        # self.model = self.model.to(device)
        self.embeddings = []
        self.text_chunks = []

    def store_embeddings(self, text):
        # Split text into chunks
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        self.text_chunks.extend(chunks)
        # Generate embeddings
        self.embeddings.extend(self.model.encode(chunks, convert_to_tensor=True).cpu().numpy())

    def query(self, question):
        question_embedding = self.model.encode([question], convert_to_tensor=True).cpu().numpy()
        similarities = cosine_similarity(question_embedding, self.embeddings)
        top_indices = np.argsort(similarities[0])[::-1][:5]
        return [self.text_chunks[i] for i in top_indices]
