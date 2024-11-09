import pandas as pd
from langchain_community.vectorstores import FAISS
import utils
from utils import embeddings
from loguru import logger


@logger.catch()
def create_vector_index(paths, lexique=True):
    faiss_index = None
    for path in paths:
        if lexique:
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(path, sep="\t")

        df.dropna(inplace=True)

        if lexique : df = df.assign(Français=df['Français'].str.split(', ')).explode('Français')

        df['Français'] = df['Français'].apply(utils.nettoyer_texte)
        df['Tagba'] = df['Tagba'].apply(utils.nettoyer_texte)

        # Préparer les textes à indexer
        texts = df['Tagba'] + " signifie" + df['Français']
        texts_list = texts.tolist()

        logger.info("create vectors")

        if faiss_index is None:
            # Créer l'index FAISS avec LangChain Community
            faiss_index = FAISS.from_texts(texts_list, embeddings)
        else:
            new_embeddings = embeddings.embed(texts_list)
            faiss_index.add_texts(texts_list, new_embeddings)

    return faiss_index

@logger.catch()
def save_vectors(path, faiss_index=None):
    logger.info("save vectors")
    faiss_index.save_local(path)
    logger.info("vectors saved")


if __name__ == "__main__":
    LEXIQUE_FILES = ["../data/tagba_francais.csv"]
    LEXIQUE_VECTOR = "/home/hamed/Documents/Perso/Python/BootCamp/Tagba AI/vector/lexique"
    TEXTE_FILES = ["../data/phrase_francais_tagba.csv"]
    TEXT_VECTOR = "/home/hamed/Documents/Perso/Python/BootCamp/Tagba AI/vector/traduction"


    # logger.info("Lexique")
    # index = create_vector_index(paths=LEXIQUE_FILES)
    # save_vectors(path=LEXIQUE_VECTOR, faiss_index=index)

    logger.info("Text")
    index = create_vector_index(paths=TEXTE_FILES, lexique=False)
    save_vectors(path=TEXT_VECTOR, faiss_index=index)

    logger.info("Ok...")