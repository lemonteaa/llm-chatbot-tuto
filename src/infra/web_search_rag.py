# Sample RAG chatbot

from transformers import AutoTokenizer

import json
import re

from duckduckgo_search import DDGS

from llama_index.readers.web import SimpleWebPageReader

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.node_parser import TokenTextSplitter

def extract_action(text):
    cap_grp = re.search("```(json|js|javascript)?([\s\S]*)```", text).group(2)
    return json.loads(cap_grp)

def web_search(q):
    results = DDGS().text(q + " filetype:html", max_results=10)
    return results


#def gen_docs(search_results):
#    docs = SimpleWebPageReader(html_to_text=True).load_data([result['href'] for result in search_results])
#    for result, doc in zip(search_results, docs):
#        doc.metadata = { 'title': result['title'], 'href': result['href'] }
#    return docs

def gen_docs(search_results):
    #docs = SimpleWebPageReader(html_to_text=True).load_data([result['href'] for result in search_results])
    docs = []
    for result in search_results:
        try:
            doc = SimpleWebPageReader(html_to_text=True).load_data([ result['href'] ])[0]
            doc.metadata = { 'title': result['title'], 'href': result['href'] }
            docs.append(doc)
        except BaseException as e:
            print(e)
    return docs

def query_crawl(docs, query, top_k, embed_model, chroma_client):
    chroma_collection = chroma_client.create_collection("temp")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, embed_model=embed_model)
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return nodes

