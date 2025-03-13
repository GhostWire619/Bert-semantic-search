import os
from typing import List, Dict, Any
import pymongo
from datetime import datetime
from .indexer import TextIndexer  # Assuming TextIndexer is in indexer.py

class DatabaseIndexer:
    def __init__(self, mongodb_uri: str, db_name: str, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the database indexer.
        
        Args:
            mongodb_uri: MongoDB connection URI
            db_name: Name of the database
            model_name: Name of the Sentence-BERT model to use (default: 'all-MiniLM-L6-v2')
        """
        self.client = pymongo.MongoClient(mongodb_uri)
        self.db = self.client[db_name]
        self.indexer = TextIndexer(model_name=model_name)  # Pass the model name explicitly
        
    def create_index_for_collection(self, collection_name: str, text_field: str = 'text') -> str:
        """
        Create a FAISS index for a specific collection.
        
        Args:
            collection_name: Name of the collection to index
            text_field: Name of the field containing text to index
            
        Returns:
            Path to the created index file
        """
        collection = self.db[collection_name]
        
        # Get all documents from the collection
        documents = list(collection.find())
        
        if not documents:
            raise ValueError(f"No documents found in collection {collection_name}")
            
        # Extract texts from documents
        texts = [doc[text_field] for doc in documents if text_field in doc]
        
        if not texts:
            raise ValueError(f"No documents found with field {text_field} in collection {collection_name}")
            
        # Create index name with timestamp and database name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        index_name = f"indexes/{collection_name}_{timestamp}.index"  # Adjusted to match TextIndexer's default dir
        
        # Add texts to index
        print(f"Creating index for {len(texts)} documents...")
        indices = self.indexer.add_texts(texts, db_name=collection_name)  # Pass db_name for metadata
        
        # Save the index
        self.indexer.save_index(index_name)
        print(f"Index saved as {index_name}")
        
        # Store index metadata in MongoDB
        metadata = {
            "collection_name": collection_name,
            "text_field": text_field,
            "index_file": index_name,
            "document_count": len(texts),
            "created_at": datetime.now(),
            "indices": indices  # Store the mapping of document indices
        }
        
        # Create or get metadata collection
        metadata_collection = self.db[f"{collection_name}_index_metadata"]
        metadata_collection.insert_one(metadata)
        
        return index_name
        
    def get_collection_indices(self, collection_name: str) -> List[Dict[str, Any]]:
        """
        Get all indices created for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            List of index metadata
        """
        metadata_collection = self.db[f"{collection_name}_index_metadata"]
        return list(metadata_collection.find().sort("created_at", -1))
        
    def delete_index(self, collection_name: str, index_file: str):
        """
        Delete an index and its metadata.
        
        Args:
            collection_name: Name of the collection
            index_file: Name of the index file to delete
        """
        # Delete the index file
        if os.path.exists(index_file):
            os.remove(index_file)
            print(f"Deleted index file: {index_file}")
            
        # Delete metadata from MongoDB
        metadata_collection = self.db[f"{collection_name}_index_metadata"]
        result = metadata_collection.delete_one({"index_file": index_file})
        if result.deleted_count > 0:
            print(f"Deleted index metadata for {index_file}")
            
    def reindex_collection(self, collection_name: str, text_field: str = 'text') -> str:
        """
        Reindex a collection by creating a new index and deleting old ones.
        
        Args:
            collection_name: Name of the collection to reindex
            text_field: Name of the field containing text to index
            
        Returns:
            Path to the new index file
        """
        # Get existing indices
        existing_indices = self.get_collection_indices(collection_name)
        
        # Delete existing indices
        for index in existing_indices:
            self.delete_index(collection_name, index['index_file'])
            
        # Create new index
        return self.create_index_for_collection(collection_name, text_field)
        
    def close(self):
        """Close the MongoDB connection."""
        self.client.close()