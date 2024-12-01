#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/DS4SD/docling/blob/main/docs/examples/rag_llamaindex.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # RAG with LlamaIndex 🦙

# ## Overview

# This example leverages the official [LlamaIndex Docling extension](../../integrations/llamaindex/).
# 
# Presented extensions `DoclingReader` and `DoclingNodeParser` enable you to:
# - use various document types in your LLM applications with ease and speed, and
# - leverage Docling's rich format for advanced, document-native grounding.

# ## Setup

# - 👉 For best conversion speed, use GPU acceleration whenever available; e.g. if running on Colab, use GPU-enabled runtime.
# - Notebook uses HuggingFace's Inference API; for increased LLM quota, token can be provided via env var `HF_TOKEN`.
# - Requirements can be installed as shown below (`--no-warn-conflicts` meant for Colab's pre-populated Python env; feel free to remove for stricter usage):



get_ipython().run_line_magic('pip', 'install -q --progress-bar off --no-warn-conflicts llama-index-core llama-index-readers-docling llama-index-node-parser-docling llama-index-embeddings-huggingface llama-index-llms-huggingface-api llama-index-vector-stores-milvus llama-index-readers-file python-dotenv')




import os
from pathlib import Path
from tempfile import mkdtemp
from warnings import filterwarnings

from dotenv import load_dotenv


def _get_env_from_colab_or_os(key):
    try:
        from google.colab import userdata

        try:
            return userdata.get(key)
        except userdata.SecretNotFoundError:
            pass
    except ImportError:
        pass
    return os.getenv(key)


load_dotenv()

filterwarnings(action="ignore", category=UserWarning, module="pydantic")
filterwarnings(action="ignore", category=FutureWarning, module="easyocr")
# https://github.com/huggingface/transformers/issues/5486:
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# We can now define the main parameters:



from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

EMBED_MODEL = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
MILVUS_URI = str(Path(mkdtemp()) / "docling.db")
GEN_MODEL = HuggingFaceInferenceAPI(
    token=_get_env_from_colab_or_os("HF_TOKEN"),
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
)
SOURCE = "https://arxiv.org/pdf/2408.09869"  # Docling Technical Report
QUERY = "Which are the main AI models in Docling?"

embed_dim = len(EMBED_MODEL.get_text_embedding("hi"))


# ## Using Markdown export

# To create a simple RAG pipeline, we can:
# - define a `DoclingReader`, which by default exports to Markdown, and
# - use a standard node parser for these Markdown-based docs, e.g. a `MarkdownNodeParser`



from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.vector_stores.milvus import MilvusVectorStore

reader = DoclingReader()
node_parser = MarkdownNodeParser()

vector_store = MilvusVectorStore(
    uri=str(Path(mkdtemp()) / "docling.db"),  # or set as needed
    dim=embed_dim,
    overwrite=True,
)
index = VectorStoreIndex.from_documents(
    documents=reader.load_data(SOURCE),
    transformations=[node_parser],
    storage_context=StorageContext.from_defaults(vector_store=vector_store),
    embed_model=EMBED_MODEL,
)
result = index.as_query_engine(llm=GEN_MODEL).query(QUERY)
print(f"Q: {QUERY}\nA: {result.response.strip()}\n\nSources:")
display([(n.text, n.metadata) for n in result.source_nodes])


# ## Using Docling format

# To leverage Docling's rich native format, we:
# - create a `DoclingReader` with JSON export type, and
# - employ a `DoclingNodeParser` in order to appropriately parse that Docling format.
# 
# Notice how the sources now also contain document-level grounding (e.g. page number or bounding box information):



from llama_index.node_parser.docling import DoclingNodeParser

reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
node_parser = DoclingNodeParser()

vector_store = MilvusVectorStore(
    uri=str(Path(mkdtemp()) / "docling.db"),  # or set as needed
    dim=embed_dim,
    overwrite=True,
)
index = VectorStoreIndex.from_documents(
    documents=reader.load_data(SOURCE),
    transformations=[node_parser],
    storage_context=StorageContext.from_defaults(vector_store=vector_store),
    embed_model=EMBED_MODEL,
)
result = index.as_query_engine(llm=GEN_MODEL).query(QUERY)
print(f"Q: {QUERY}\nA: {result.response.strip()}\n\nSources:")
display([(n.text, n.metadata) for n in result.source_nodes])


# ## With Simple Directory Reader

# To demonstrate this usage pattern, we first set up a test document directory.



from pathlib import Path
from tempfile import mkdtemp

import requests

tmp_dir_path = Path(mkdtemp())
r = requests.get(SOURCE)
with open(tmp_dir_path / f"{Path(SOURCE).name}.pdf", "wb") as out_file:
    out_file.write(r.content)


# Using the `reader` and `node_parser` definitions from any of the above variants, usage with `SimpleDirectoryReader` then looks as follows:



from llama_index.core import SimpleDirectoryReader

dir_reader = SimpleDirectoryReader(
    input_dir=tmp_dir_path,
    file_extractor={".pdf": reader},
)

vector_store = MilvusVectorStore(
    uri=str(Path(mkdtemp()) / "docling.db"),  # or set as needed
    dim=embed_dim,
    overwrite=True,
)
index = VectorStoreIndex.from_documents(
    documents=dir_reader.load_data(SOURCE),
    transformations=[node_parser],
    storage_context=StorageContext.from_defaults(vector_store=vector_store),
    embed_model=EMBED_MODEL,
)
result = index.as_query_engine(llm=GEN_MODEL).query(QUERY)
print(f"Q: {QUERY}\nA: {result.response.strip()}\n\nSources:")
display([(n.text, n.metadata) for n in result.source_nodes])






