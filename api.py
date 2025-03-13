from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from utils.database_indexer import DatabaseIndexer
from utils.indexer import TextIndexer
from contextlib import asynccontextmanager
import pymongo.errors
import os
import logging
from dotenv import load_dotenv
import certifi
import json

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB configuration from .env
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "my_database"
os.environ["KMP_DUPLICATE_LIB_OK"] = os.getenv("KMP_DUPLICATE_LIB_OK", "TRUE")

if not MONGODB_URI:
    logger.error("MONGODB_URI not found in .env file")
    raise Exception("MONGODB_URI environment variable is required")

try:
    db_indexer = DatabaseIndexer(mongodb_uri=MONGODB_URI, db_name=DB_NAME)
    db_indexer.client.admin.command('ping')
    logger.info("Successfully connected to MongoDB")
except pymongo.errors.ConnectionError as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise Exception(f"Database connection failed: {str(e)}")
except pymongo.errors.ConfigurationError as e:
    logger.error(f"MongoDB configuration error: {str(e)}")
    raise Exception(f"Database configuration failed: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    try:
        db_indexer.close()
        logger.info("MongoDB connection closed successfully")
    except Exception as e:
        logger.error(f"Error closing MongoDB connection: {str(e)}")

app = FastAPI(
    title="Database Indexing API",
    description="API for indexing, adding, and searching text in a MongoDB database",
    version="1.0.0",
    lifespan=lifespan
)

# Existing Pydantic models (unchanged)
class IndexRequest(BaseModel):
    collection_name: str
    text_field: str = "text"

class AddTextRequest(BaseModel):
    collection_name: str
    text: str
    metadata: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    collection_name: str
    query: str
    k: int = 5

class IndexResponse(BaseModel):
    index_file: str
    document_count: int
    message: str

class AddTextResponse(BaseModel):
    document_id: str
    message: str

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    message: str

# Updated response model for bulk addition
class AddBulkTextResponse(BaseModel):
    document_ids: List[str]
    count: int
    message: str

# Modified route to handle a list of strings
@app.post("/add-bulk-texts", response_model=AddBulkTextResponse)
async def add_bulk_texts(collection_name: str, file: UploadFile = File(...)):
    """
    Add a large batch of texts from a JSON file to a MongoDB collection without indexing.
    Expects a JSON file with a list of strings (e.g., ["text1", "text2", ...]).
    Each string is stored as a document with a 'text' field.
    """
    try:
        # Validate collection name
        if not collection_name or not collection_name.strip():
            logger.warning("Invalid collection name provided")
            raise HTTPException(status_code=400, detail="Collection name cannot be empty")

        # Validate file type
        if not file.filename.endswith('.json'):
            logger.warning(f"Invalid file type uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="Only JSON files are supported")

        logger.info(f"Processing bulk text upload for collection: {collection_name}")
        
        # Read and parse JSON file
        content = await file.read()
        try:
            texts = json.loads(content.decode('utf-8'))
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")

        # Validate that it's a list of strings
        if not isinstance(texts, list):
            logger.warning("JSON must contain a list")
            raise HTTPException(status_code=400, detail="JSON must be a list")

        if not texts:
            logger.warning("Empty text list provided")
            raise HTTPException(status_code=400, detail="No texts provided")

        # Prepare documents for insertion
        prepared_docs = []
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                logger.warning(f"Item at index {i} is not a string")
                raise HTTPException(status_code=400, detail=f"Item at index {i} must be a string")
            
            if not text.strip():
                logger.warning(f"Empty text at index {i}")
                raise HTTPException(status_code=400, detail=f"Text at index {i} cannot be empty")
            
            prepared_docs.append({"text": text})

        # Bulk insert into MongoDB
        collection = db_indexer.db[collection_name]
        result = collection.insert_many(prepared_docs)
        document_ids = [str(id) for id in result.inserted_ids]

        logger.info(f"Successfully added {len(document_ids)} documents to {collection_name}")
        return AddBulkTextResponse(
            document_ids=document_ids,
            count=len(document_ids),
            message=f"Successfully added {len(document_ids)} documents to {collection_name}"
        )

    except pymongo.errors.PyMongoError as e:
        logger.error(f"MongoDB error during bulk text addition: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during bulk text addition: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    
#halaaaaa
    



@app.post("/index", response_model=IndexResponse)
async def index_collection(request: IndexRequest):
    try:
        if not request.collection_name or not request.collection_name.strip():
            logger.warning("Invalid collection name provided")
            raise HTTPException(status_code=400, detail="Collection name cannot be empty")
        
        if not request.text_field or not request.text_field.strip():
            logger.warning("Invalid text field provided")
            raise HTTPException(status_code=400, detail="Text field cannot be empty")

        logger.info(f"Starting indexing for collection: {request.collection_name}")
        index_file = db_indexer.create_index_for_collection(
            collection_name=request.collection_name,
            text_field=request.text_field
        )
        
        metadata = db_indexer.get_collection_indices(request.collection_name)
        if not metadata:
            logger.error(f"No metadata found after indexing {request.collection_name}")
            raise HTTPException(status_code=500, detail="Failed to retrieve index metadata")
        
        logger.info(f"Successfully indexed {request.collection_name} with {metadata[0]['document_count']} documents")
        return IndexResponse(
            index_file=index_file,
            document_count=metadata[0]["document_count"],
            message=f"Successfully indexed collection {request.collection_name}"
        )
    
    except ValueError as e:
        logger.error(f"ValueError during indexing: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except pymongo.errors.PyMongoError as e:
        logger.error(f"MongoDB error during indexing: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")
    except FileNotFoundError as e:
        logger.error(f"File system error during indexing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Index file error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during indexing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/add-text", response_model=AddTextResponse)
async def add_text_to_collection(request: AddTextRequest):
    try:
        if not request.collection_name or not request.collection_name.strip():
            logger.warning("Invalid collection name provided")
            raise HTTPException(status_code=400, detail="Collection name cannot be empty")
        
        if not request.text or not request.text.strip():
            logger.warning("Empty text provided")
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        logger.info(f"Adding text to collection: {request.collection_name}")
        collection = db_indexer.db[request.collection_name]
        document = {"text": request.text}
        if request.metadata:
            document.update(request.metadata)
        
        result = collection.insert_one(document)
        document_id = str(result.inserted_id)
        
        logger.info(f"Reindexing collection {request.collection_name} after adding document {document_id}")
        db_indexer.reindex_collection(
            collection_name=request.collection_name,
            text_field="text"
        )
        
        logger.info(f"Text added and index updated for {request.collection_name}")
        return AddTextResponse(
            document_id=document_id,
            message=f"Text added to {request.collection_name} and index updated"
        )
    
    except pymongo.errors.PyMongoError as e:
        logger.error(f"MongoDB error during text addition: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")
    except FileNotFoundError as e:
        logger.error(f"File system error during reindexing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Index file error: {str(e)}")
    except ValueError as e:
        logger.error(f"ValueError during reindexing: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during text addition: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_collection(request: SearchRequest):
    try:
        if not request.collection_name or not request.collection_name.strip():
            logger.warning("Invalid collection name provided")
            raise HTTPException(status_code=400, detail="Collection name cannot be empty")
        
        if not request.query or not request.query.strip():
            logger.warning("Empty query provided")
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if request.k <= 0:
            logger.warning(f"Invalid k value: {request.k}")
            raise HTTPException(status_code=400, detail="k must be a positive integer")

        logger.info(f"Searching collection: {request.collection_name} with query: '{request.query}'")
        indices = db_indexer.get_collection_indices(request.collection_name)
        if not indices:
            logger.warning(f"No index found for {request.collection_name}")
            raise HTTPException(status_code=404, detail=f"No index found for collection {request.collection_name}")
        
        latest_index = indices[0]["index_file"]
        if not os.path.exists(latest_index):
            logger.error(f"Index file {latest_index} not found on disk")
            raise HTTPException(status_code=500, detail=f"Index file {latest_index} missing")

        text_indexer = TextIndexer()
        text_indexer.load_index(latest_index)
        search_results = text_indexer.search(
            query=request.query,
            k=request.k,
            db_name=request.collection_name
        )
        
        collection = db_indexer.db[request.collection_name]
        documents = list(collection.find())
        if not documents:
            logger.warning(f"No documents found in {request.collection_name}")
            raise HTTPException(status_code=404, detail=f"No documents found in collection {request.collection_name}")

        results = []
        for idx, distance in zip(search_results["indices"], search_results["distances"]):
            if idx < len(documents):
                doc = documents[idx]
                results.append({
                    "document_id": str(doc["_id"]),
                    "text": doc.get("text", ""),
                    "distance": distance,
                    "metadata": {k: v for k, v in doc.items() if k not in ["_id", "text"]}
                })
            else:
                logger.warning(f"Index {idx} out of range for {request.collection_name}")
        
        logger.info(f"Search completed, found {len(results)} results")
        return SearchResponse(
            results=results,
            message=f"Found {len(results)} similar texts"
        )
    
    except pymongo.errors.PyMongoError as e:
        logger.error(f"MongoDB error during search: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")
    except FileNotFoundError as e:
        logger.error(f"File system error during search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Index file error: {str(e)}")
    except ValueError as e:
        logger.error(f"ValueError during search: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)