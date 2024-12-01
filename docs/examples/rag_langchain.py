#!/usr/bin/env python
# coding: utf-8

# # RAG with LangChain 🦜🔗



# requirements for this example:
get_ipython().run_line_magic('pip', 'install -qq docling docling-core python-dotenv langchain-text-splitters langchain-huggingface langchain-milvus')




import os

from dotenv import load_dotenv

load_dotenv()


# ## Setup

# ### Loader and splitter

# Below we set up:
# - a `Loader` which will be used to create LangChain documents, and
# - a splitter, which will be used to split these documents



from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument

from docling.document_converter import DocumentConverter

class DoclingPDFLoader(BaseLoader):

    def __init__(self, file_path: str | list[str]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter()

    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            dl_doc = self._converter.convert(source).document
            text = dl_doc.export_to_markdown()
            yield LCDocument(page_content=text)




FILE_PATH = "https://raw.githubusercontent.com/DS4SD/docling/main/tests/data/2206.01062.pdf"  # DocLayNet paper




from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = DoclingPDFLoader(file_path=FILE_PATH)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


# We now used the above-defined objects to get the document splits:



docs = loader.load()
splits = text_splitter.split_documents(docs)


# ### Embeddings



from langchain_huggingface.embeddings import HuggingFaceEmbeddings

HF_EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL_ID)


# ### Vector store



from tempfile import TemporaryDirectory

from langchain_milvus import Milvus

MILVUS_URI = os.environ.get(
    "MILVUS_URI", f"{(tmp_dir := TemporaryDirectory()).name}/milvus_demo.db"
)

vectorstore = Milvus.from_documents(
    splits,
    embeddings,
    connection_args={"uri": MILVUS_URI},
    drop_old=True,
)


# ### LLM



from langchain_huggingface import HuggingFaceEndpoint

HF_API_KEY = os.environ.get("HF_API_KEY")
HF_LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

llm = HuggingFaceEndpoint(
    repo_id=HF_LLM_MODEL_ID,
    huggingfacehub_api_token=HF_API_KEY,
)


# ## RAG



from typing import Iterable

from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs: Iterable[LCDocument]):
    return "\n\n".join(doc.page_content for doc in docs)


retriever = vectorstore.as_retriever()

prompt = PromptTemplate.from_template(
    "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {question}\nAnswer:\n"
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)




rag_chain.invoke("How many pages were human annotated for DocLayNet?")

