## Document Search and Retrieval System

### Overview

This project aims to develop a document search and retrieval system using various text vectorization techniques and semantic search methods. The system can ingest large databases of documents, preprocess them, generate embeddings, and efficiently retrieve relevant documents based on user queries.

### Features
- Ingesting Documents: Read and decode database files, sample data if necessary, clean and preprocess subtitle documents.
- BERT-based SentenceTransformers: Generate embeddings to encode semantic information for a semantic search engine.
- Document Chunking: Divide large documents into smaller chunks to manage embedding and mitigate information loss.
- Storage: Store embeddings in a ChromaDB database for efficient retrieval.
- Retrieving Documents: Accept user search queries, preprocess them, generate query embeddings, calculate similarity scores using cosine distance, and return relevant documents.
