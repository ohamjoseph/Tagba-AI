
from langchain_community.vectorstores import FAISS
from .utils import *
from loguru import logger

LEXIQUE_VECTOR = LEXIQUE_VECTOR
TEXT_VECTOR = TEXT_VECTOR



@logger.catch(level='ERROR')
def retrived_lexque(lexique):

    logger.info("Loading lexique vector")

    faiss_index = FAISS.load_local(LEXIQUE_VECTOR,
                                   embeddings,
                                   allow_dangerous_deserialization=True)

    results = faiss_index.similarity_search(lexique, k=2)  # k est le nombre de résultats
    lexiques = [str(result.page_content) for result in results]
    return "\n".join(lexiques)



@logger.catch(level='ERROR')
def retrived_text(text):

    logger.info("Loading text vector")

    faiss_index = FAISS.load_local(TEXT_VECTOR,
                                      embeddings,
                                      allow_dangerous_deserialization=True)


    results = faiss_index.similarity_search(text, k=5)  # k est le nombre de résultats
    lexiques = [str(result.page_content) for result in results]

    return "\n".join(lexiques)



