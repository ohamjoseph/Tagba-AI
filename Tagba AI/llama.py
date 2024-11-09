from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from .vector_loader import retrived_lexque, retrived_text
from .utils import *




# template_str = """
#     Tu es un assistant dont le rôle est faire la traduction entre du Tagba au français ou du français au Tagba
#     en te basant sur le dictionnaire du tagba vers le français suivant.
#     '''
#     {lexique}
#     '''
#     et sur les traduction ci-dessous:
#     '''
#         {text}
#     '''
#
#     Assure toi de toujours donnée une traduction dans la langue demandé
#     Affiche uniquement la traduction correspondant.
#
#     Tu traduis la phrase dans la langue mentionée:
#     phrase : {phrase}
#     langue : {langue}
#     """


template_str = """
    Tu es un assistant spécialisé dans la traduction entre le Tagba et le français.
    Ton rôle est de fournir des traductions précises en utilisant les ressources suivantes :

    1. Dictionnaire Tagba-Français :
        {lexique}

    2. Traductions de référence :
        {text}

    Instructions :
    - Traduis toujours dans la langue demandée, sans afficher d'informations additionnelles.
    - Présente uniquement la traduction correspondante en suivant la langue spécifiée.

    Détails de la demande :
    - Phrase à traduire : {phrase}
    - Langue cible : {langue}
"""

# template_str = """
#     You are an assistant whose role is to translate between Tagba and French or from French to Tagba
#     based on the following dictionary:
#     '''
#         {lexique}
#     '''
#     You translate the sentence into the mentioned language:
#     sentence : {phrase}
#     language : {langue}
#
#     Provide only the translation
#     """



# template_str = """
#     Tu es un assistant dont le rôle est faire la traduction entre du Tagba au français ou du français au Tagba
#     en te basant sur le dictionnaire suivant:
#     '''
#         bǎ -> ballon
#         bâd -> empêcher
#         bádáféw -> jamais
#         bà̰.jí.ŋɛ -> vin de palme.
#         bàlɛ́ -> sans.
#         bán -> finir
#         bât -> fabriquer.
#         bǎt -> causérie.
#         bàt.bɩ̀-> N. banane.
#         bə̂ 	-> pois de terre.
#         bə́d -> esclave.
#         bə̌d -> bile.
#         bə̀t -> couler.
#         bɛt.mɛ́->. être fort.
#         bí 	->.
#         bɩ́	->. petit.
#         bɩd -> ramasser.
#         bɩd -> respecter, obéir.
#         bɩd -> puits.
#         bɩ̀.dà -> finalement.
#         bɩ̀lɩ̀ 	-> être épais, être lourd.
#         bɩ̂t -> surprendre.
#         bǒb -> muet.
#         bòg.nàt.té 	-> gecko.
#         bɔ̌t -> restreindre.
#         bǔd -> araignée.
#         bǔn -> terrasse aérienne
#         cá 	-> être rapide.
#         cá 	-> Auxiliaire de prédication.
#         cǎ 	->  N. enfant.
#         cǎ̰ 	-> marché.
#         cá.câd -> truie.
#         càd -> aplatir, écarter.
#         câd -> porc, cochon.
#         cǎd -> haricot.
#         cǎk -> chercher.
#         cǎn -> jour.
#         cǎ̰t -> panthère.
#         cà.wɛ̌-> odeur forte.
#         cə̌d -> faire sècher.
#         cə̀t -> décortiquer.
#         cé 	-> faire.
#         cě -> noix.
#         cě -> frémir.
#         cê -> oeuf.
#         cê -> refuser.
#         cê -> corps.
#         cèd -> chanter.
#         cêd -> femme.
#
#         dù.jàn 	-> mal de poitrine.
#         dǔn -> poitrine.
#     '''
#
#     Tu traduit la phrase dans la langue mention:
#     phrase : {phrase}
#     langue : {langue}
#
#
#     Donne uniquement la traduction
#
#     """

prompt = PromptTemplate.from_template(template_str)

# print(prompt)
model = ChatOllama(
        model="llama3.2", temperature=0
    )

phrase = "maison"

#print(prompt.format(phrase=phrase, langue="tagba", lexique=retrived_lexque(phrase)))

# query = {
#     "phrase": phrase,
#     "langue": "tagba",
#     "lexique": retrived_lexque(phrase),
# }

parser = StrOutputParser()

from operator import itemgetter

chain = (
        {
            "phrase": itemgetter("phrase"),
            "langue": itemgetter("langue"),
            "lexique": itemgetter("lexique"),
            "text": itemgetter("text"),
        }
        | prompt
        | model
        | parser

    )


def generate_prompte(inputs, langue):
    inputs = nettoyer_texte(inputs)
    lexiques = inputs.split(" ")
    ret_lexique = [ retrived_lexque(lexique) for lexique in lexiques ]
    ret_lexique = "\n".join(ret_lexique)

    ret_text = retrived_text(ret_lexique)

    query = {
        "phrase": inputs,
        "langue": langue,
        "lexique": ret_lexique,
        "text": ret_text,
    }

    return chain.invoke(query)


#generate_prompte("le tô est de la nourriture.", langue="tagba")


