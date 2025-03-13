# Database Indexing API

A RESTful API built with FastAPI for indexing, adding, and searching text data in a MongoDB database using FAISS and SentenceTransformers.

## Overview

This API provides endpoints to:

- Index a MongoDB collection for fast similarity search.
- Add text to a collection and update its index.
- Search for similar texts in an indexed collection.

It leverages `DatabaseIndexer` and `TextIndexer` from the `utils` module to manage text embeddings and vector search.

## Prerequisites

- Python 3.8+
- MongoDB instance running (e.g., `mongodb://localhost:27017/`)
- Required Python packages (see [Installation](#installation))

## Installation

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set Up a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Use the provided `requirements.txt` to install dependencies without caching:

   ```bash
   pip install -r requirements.txt --no-cache-dir
   ```

   Example `requirements.txt`:

   ```
   fastapi
   uvicorn
   pymongo
   sentence-transformers
   faiss-cpu
   torch
   pydantic
   ```

4. **Configure MongoDB**:

   - Ensure MongoDB is running locally or update `MONGODB_URI` in `api.py` to point to your instance.
   - Default: `mongodb://localhost:27017/` with database `my_database`.

5. **Run the API**:
   ```bash
   uvicorn api:app --reload --host 0.0.0.0 --port 8000
   ```
   - `--reload`: Enables auto-reload for development.
   - Access at `http://localhost:8000`.

## API Endpoints

### 1. Index a Collection

- **Endpoint**: `POST /index`
- **Request Body**:
  ```json
  {
    "collection_name": "articles",
    "text_field": "content"
  }
  ```
- **Response**:
  ```json
  {
    "index_file": "indexes/articles_20250313_123456.index",
    "document_count": 100,
    "message": "Successfully indexed collection articles"
  }
  ```
- **Purpose**: Creates a FAISS index for the specified collection and text field.

### 2. Add Text to a Collection

- **Endpoint**: `POST /add-text`
- **Request Body**:
  ```json
  {
    "collection_name": "articles",
    "text": "This is a sample text",
    "metadata": { "author": "John" }
  }
  ```
- **Response**:
  ```json
  {
    "document_id": "623f1a2b3c4d5e6f7890abcd",
    "message": "Text added to articles and index updated"
  }
  ```
- **Purpose**: Adds text to a collection and reindexes it.

### 3. Search a Collection

- **Endpoint**: `POST /search`
- **Request Body**:
  ```json
  {
    "collection_name": "articles",
    "query": "sample text",
    "k": 3
  }
  ```
- **Response**:
  ```json
  {
    "results": [
      {
        "document_id": "623f1a2b3c4d5e6f7890abcd",
        "text": "This is a sample text",
        "distance": 0.123,
        "metadata": { "author": "John" }
      }
    ],
    "message": "Found 3 similar texts"
  }
  ```
- **Purpose**: Searches for the top `k` similar texts in the indexed collection.

## Best Practices

### Setup

- **Use a Virtual Environment**: Isolate dependencies to avoid conflicts with other projects.
- **Environment Variables**: Store sensitive configs (e.g., `MONGODB_URI`) in a `.env` file and load them using `python-dotenv`:
  ```python
  from dotenv import load_dotenv
  import os
  load_dotenv()
  MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
  ```
- **Version Control**: Pin dependency versions in `requirements.txt` (e.g., `fastapi==0.95.1`) for reproducibility.

### Usage

- **Test with Swagger UI**: Access `http://localhost:8000/docs` for an interactive API explorer.
- **Batch Operations**: For large datasets, consider batching text additions instead of reindexing after every `/add-text` call to improve performance.
- **Error Handling**: Check response status codes (e.g., 400 for bad requests, 500 for server errors) and handle accordingly in your client application.
- **Rate Limiting**: Add rate limiting (e.g., with `slowapi`) in production to prevent abuse:
  ```bash
  pip install slowapi
  ```

### Performance

- **Incremental Indexing**: Modify `/add-text` to update the index incrementally (append to FAISS) rather than reindexing the entire collection.
- **GPU Support**: Use `faiss-gpu` instead of `faiss-cpu` if you have a compatible GPU for faster indexing and search.
- **Async Operations**: For large indexing tasks, run them in the background using a task queue (e.g., Celery) to avoid blocking the API:
  ```bash
  pip install celery
  ```

### Security

- **Authentication**: Implement OAuth2 or API key authentication for production use:
  ```python
  from fastapi.security import OAuth2PasswordBearer
  oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
  ```
- **Input Validation**: Rely on Pydantic models to enforce strict input validation and avoid injection attacks.
- **HTTPS**: Deploy with a reverse proxy (e.g., Nginx) and SSL in production.

### Maintenance

- **Logging**: Add logging to track API usage and errors:
  ```python
  import logging
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger(__name__)
  ```
- **Monitoring**: Integrate with tools like Prometheus and Grafana for performance monitoring.
- **Backup Indexes**: Regularly back up the `indexes/` directory to prevent data loss.

## Troubleshooting

- **MongoDB Connection Errors**: Verify the URI and ensure the MongoDB server is running.
- **Index Not Found**: Ensure the collection has been indexed before searching.
- **Performance Issues**: Check memory usage for large collections; consider sharding MongoDB or optimizing FAISS parameters.

## Development

- **Reload Mode**: Use `--reload` with `uvicorn` during development for live updates.
- **Unit Tests**: Write tests with `pytest` to verify endpoint behavior:
  ```bash
  pip install pytest httpx
  ```

## License

MIT License (or specify your preferred license).

---

### Notes

- This README assumes a basic setup but includes scalable best practices for production use.
- You can customize sections (e.g., add specific deployment instructions) based on your needs.
- Save this as `README.md` in your project root for easy access.

Let me know if you'd like to expand any section or tailor it further!
