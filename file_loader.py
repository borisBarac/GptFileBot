from typing import Tuple

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

from langchain.agents.agent_toolkits import (
    VectorStoreToolkit,
    VectorStoreInfo
)

def create_vectorstore(file: str) -> Tuple[VectorStoreToolkit, Chroma]:
    '''Creates langchain vectorStore in ChromaDB form a PDF file'''
    embeddings = OpenAIEmbeddings()

    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    store = Chroma.from_documents(pages, embeddings, collection_name='{}_collection'.format(file))

    vectorstore_info = VectorStoreInfo(
        name="annual_report",
        description="a banking annual report as a pdf",
        vectorstore=store
    )

    # Convert the document store into a langchain toolkit
    return (VectorStoreToolkit(vectorstore_info=vectorstore_info), store)

def is_Pdf(file):
    try:
        PdfReader(file)
    except PdfReadError:
        print("Error: invalid PDF file")
        return False
    else:
        return True