import wikipedia

# Configurer la langue en français
wikipedia.set_lang("fr")

# Titre de l'article (par exemple "Argentine", "ChatGPT", etc.)
titre = "Intelligence_artificielle"

# Récupération du texte de l'article
article = wikipedia.page(titre)
texte_brut = article.content

# Affichage ou enregistrement
print(texte_brut)

with open("wikipediaIA.txt", "w", encoding="utf-8") as f:
    f.write(texte_brut)
