from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import pickle
import nltk
import numpy as np

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
class VectorDB:
    def __init__(self, embedding_dimension=768, index_file="vector_db.faiss", data_file="vector_data.pkl"):
        self.embedding_dimension = embedding_dimension
        self.index_file = index_file
        self.data_file = data_file
        self.model = SentenceTransformer("all-mpnet-base-v2", device='gpu' if torch.cuda.is_available() else 'cpu')
        self.embeddings = None
        self.text_chunks = None
        self.index = None
        self._load()

    def _load(self):
        try:
            self.index = faiss.read_index(self.index_file)
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
                self.text_chunks = data['text_chunks']
        except (FileNotFoundError, EOFError,Exception):
            # Initialize a new FAISS index if the file does not exist or is corrupted
            self.embeddings = []
            self.text_chunks = []
            self.index = faiss.IndexFlatL2(self.embedding_dimension)  # L2 distance for faiss-cpu
            self._save()  # Save the newly created index and data file

    def _save(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.data_file, 'wb') as f:
            pickle.dump({'embeddings': self.embeddings, 'text_chunks': self.text_chunks}, f)

    def semantic_chunking(self, text, chunk_size=500, overlap=50):
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += (sentence + " ")
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
                elif len(sentence) > chunk_size:
                    sub_chunks = [sentence[i:i+chunk_size] for i in range(0, len(sentence), chunk_size)]
                    chunks.extend(sub_chunks)
                    current_chunk = ""
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Add overlap
        final_chunks = []
        if overlap > 0 and len(chunks) > 1:
            for i in range(len(chunks)):
                start = max(0, i - 1)
                end = min(len(chunks), i + 2)
                final_chunks.append(" ".join(chunks[start:end]))
        else:
            final_chunks = chunks

        return final_chunks

    def store_embeddings(self, text):
        chunks = self.semantic_chunking(text)
        new_embeddings = self.model.encode(chunks, convert_to_tensor=True).cpu().numpy()
        new_embeddings = new_embeddings.reshape(-1, self.embedding_dimension)  # Ensure correct dimensions
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.concatenate([self.embeddings, new_embeddings], axis=0)  # Concatenate along rows
        self.text_chunks.extend(chunks)
        self.index.add(new_embeddings.astype('float32'))  # Ensure embeddings are float32 for faiss
        self._save()

    def query(self, question, top_k=5):
        question_embedding = self.model.encode([question], convert_to_tensor=True).cpu().numpy().astype('float32')  # Ensure float32
        D, I = self.index.search(question_embedding, top_k)
        return [self.text_chunks[i] for i in I[0] if i < len(self.text_chunks)]
