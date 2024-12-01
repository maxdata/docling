#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/DS4SD/docling/blob/main/docs/examples/hybrid_rag_qdrant
# .ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Hybrid RAG with Qdrant

# ## Overview

# This example demonstrates using Docling with [Qdrant](https://qdrant.tech/) to perform a hybrid search across your documents using dense and sparse vectors.
# 
# We'll chunk the documents using Docling before adding them to a Qdrant collection. By limiting the length of the chunks, we can preserve the meaning in each vector embedding.

# ## Setup

# - 👉 Qdrant client uses [FastEmbed](https://github.com/qdrant/fastembed) to generate vector embeddings. You can install the `fastembed-gpu` package if you've got the hardware to support it.



get_ipython().run_line_magic('pip', 'install --no-warn-conflicts -q qdrant-client docling docling-core fastembed')


# Let's import all the classes we'll be working with.



from docling_core.transforms.chunker import HierarchicalChunker
from qdrant_client import QdrantClient

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter


# - For Docling, we'll set the  allowed formats to HTML since we'll only be working with webpages in this tutorial.
# - If we set a sparse model, Qdrant client will fuse the dense and sparse results using RRF. [Reference](https://qdrant.tech/documentation/tutorials/hybrid-search-fastembed/).



COLLECTION_NAME = "docling"

doc_converter = DocumentConverter(allowed_formats=[InputFormat.HTML])
client = QdrantClient(location=":memory:")
# The :memory: mode is a Python imitation of Qdrant's APIs for prototyping and CI.
# For production deployments, use the Docker image: docker run -p 6333:6333 qdrant/qdrant
# client = QdrantClient(location="http://localhost:6333")

client.set_model("sentence-transformers/all-MiniLM-L6-v2")
client.set_sparse_model("Qdrant/bm25")


# We can now download and chunk the document using Docling. For demonstration, we'll use an article about chunking strategies :)



result = doc_converter.convert(
    "https://www.sagacify.com/news/a-guide-to-chunking-strategies-for-retrieval-augmented-generation-rag"
)
documents, metadatas = [], []
for chunk in HierarchicalChunker().chunk(result.document):
    documents.append(chunk.text)
    metadatas.append(chunk.meta.export_json_dict())


# Let's now upload the documents to Qdrant.
# 
# - The `add()` method batches the documents and uses FastEmbed to generate vector embeddings on our machine.



client.add(COLLECTION_NAME, documents=documents, metadata=metadatas, batch_size=64)


# ## Query Documents



points = client.query(COLLECTION_NAME, query_text="Can I split documents?", limit=10)

print("<=== Retrieved documents ===>")
for point in points:
    print(point.document)

