import os
from typing import List, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from .text_preprocessing import preprocess_text
from datetime import datetime
import glob
import json

class TextIndexer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', dimension: int = 384, index_dir: str = 'indexes'):
        """
        Initialize the indexer with a specific model and dimension.
        
        Args:
            model_name: Name of the Sentence-BERT model to use (default: 'all-MiniLM-L6-v2')
            dimension: Dimension of the embeddings (384 for all-MiniLM-L6-v2)
            index_dir: Directory to store index files
        """
        self.model_name = model_name
        self.dimension = dimension
        self.index_dir = index_dir
        self.index = faiss.IndexFlatL2(dimension)
        self.current_index_file = None
        
        # Create index directory if it doesn't exist
        os.makedirs(index_dir, exist_ok=True)
        
        # Initialize SentenceTransformer model
        self.model = SentenceTransformer(model_name)
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
    def get_latest_index_file(self, db_name: str = None) -> str:
        """
        Get the path to the latest index file for a specific database.
        
        Args:
            db_name: Name of the database (optional)
            
        Returns:
            Path to the latest index file or None if no index exists
        """
        pattern = f"{self.index_dir}/*.index"
        if db_name:
            pattern = f"{self.index_dir}/{db_name}_*.index"
            
        index_files = glob.glob(pattern)
        if not index_files:
            return None
            
        # Sort by modification time, newest first
        latest_index = max(index_files, key=os.path.getmtime)
        return latest_index
        
    def get_index_metadata(self, index_file: str) -> Dict:
        """
        Get metadata for an index file.
        
        Args:
            index_file: Path to the index file
            
        Returns:
            Dictionary containing index metadata
        """
        metadata_file = index_file.replace('.index', '.meta.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {}
        
    def save_index_metadata(self, index_file: str, metadata: Dict):
        """
        Save metadata for an index file.
        
        Args:
            index_file: Path to the index file
            metadata: Dictionary containing metadata to save
        """
        metadata_file = index_file.replace('.index', '.meta.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array of embeddings
        """
        # Preprocess the text
        processed_text = preprocess_text(text)
        
        # Generate embeddings using SentenceTransformer
        with torch.no_grad():
            embedding = self.model.encode(processed_text, convert_to_numpy=True)
        
        # Ensure embedding is 2D (FAISS expects [n_samples, dimension])
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        return embedding
        
    def add_texts(self, texts: List[str], db_name: str = None) -> List[int]:
        """
        Add multiple texts to the index.
        
        Args:
            texts: List of texts to add
            db_name: Name of the database (optional)
            
        Returns:
            List of indices where the texts were added
        """
        indices = []
        for text in texts:
            embedding = self.get_embedding(text)
            index_id = self.index.ntotal
            self.index.add(embedding)
            indices.append(index_id)
            
        # Update the existing index file if it exists, otherwise create a new one
        if indices:
            if not self.current_index_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if db_name:
                    self.current_index_file = f"{self.index_dir}/{db_name}_{timestamp}.index"
                else:
                    self.current_index_file = f"{self.index_dir}/index_{timestamp}.index"
            
            # Save to the current index file
            self.save_index(self.current_index_file)
            
            # Update metadata
            metadata = {
                "last_updated": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "created_at": self.get_index_metadata(self.current_index_file).get('created_at', datetime.now().strftime("%Y%m%d_%H%M%S")),
                "db_name": db_name,
                "total_documents": self.index.ntotal,
                "dimension": self.dimension,
                "model_name": self.model_name
            }
            self.save_index_metadata(self.current_index_file, metadata)
            
        return indices
        
    def search(self, query: str, k: int = 5, db_name: str = None) -> Dict[str, Any]:
        """
        Search for similar texts.
        
        Args:
            query: Text to search for
            k: Number of results to return
            db_name: Name of the database to search in (optional)
            
        Returns:
            Dictionary containing distances and indices
        """
        # If db_name provided, ensure we're using the latest index for that db
        if db_name and (not self.current_index_file or db_name not in self.current_index_file):
            latest_index = self.get_latest_index_file(db_name)
            if latest_index and latest_index != self.current_index_file:
                self.load_index(latest_index)
        
        query_embedding = self.get_embedding(query)
        distances, indices = self.index.search(query_embedding, k)
        
        return {
            'distances': distances[0].tolist(),
            'indices': indices[0].tolist()
        }
        
    def save_index(self, path: str = None):
        """
        Save the FAISS index to disk.
        
        Args:
            path: Path where to save the index. If None, uses current_index_file
        """
        save_path = path or self.current_index_file
        if not save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{self.index_dir}/index_{timestamp}.index"
            
        faiss.write_index(self.index, save_path)
        self.current_index_file = save_path
        
    def load_index(self, path: str):
        """
        Load the FAISS index from disk.
        
        Args:
            path: Path to the index file
        """
        if os.path.exists(path):
            self.index = faiss.read_index(path)
            self.current_index_file = path
        else:
            raise FileNotFoundError(f"No index file found at {path}")
            
    def clear_index(self):
        """Clear the current index and initialize a new one."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.current_index_file = None