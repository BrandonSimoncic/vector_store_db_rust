# vector-store-db
A simple Rust crate for building and managing a vector database for semantic search. 

**Currently a work in progress**

If you attempt to install this, please add your own *tokenizer.json* file for whatever model you intend to use.



## Overview
This project provides tools to create and manage embeddings-based vector stores for documents. It supports:

- Document storage with embeddings
- Querying based on semantic similarity
- Metadata filtering for more precise searches
- PDF file processing
- Text chunking and embedding generation

## Features
- Document Storage: Store text chunks along with their embeddings and metadata.
- Semantic Search: Find similar documents using vector embeddings.
- Metadata Filtering: Filter search results based on document metadata.
- PDF Processing: Extract text from PDF files for indexing.
- Text Chunking: Split long texts into manageable chunks before embedding.