import string

from langchain_ollama import OllamaEmbeddings


LEXIQUE_FILES = ["../data/tagba_francais.csv"]
LEXIQUE_VECTOR = "/home/hamed/Documents/Perso/Python/BootCamp/Tagba AI/vector/lexique"
TEXTE_FILES = ["../data/phrase_francais_tagba.csv"]
TEXT_VECTOR = "/home/hamed/Documents/Perso/Python/BootCamp/Tagba AI/vector/traduction"

embeddings = OllamaEmbeddings(
    model= "llama3.2",
)

def nettoyer_texte(texte):
    # Convertir en minuscules
    texte = texte.lower()
    # Supprimer la ponctuation
    texte_sans_ponctuation = texte.translate(str.maketrans('', '', string.punctuation))
    return texte_sans_ponctuation