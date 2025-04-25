with open("wikipediaIA.txt", "r", encoding="utf-8") as f:
    texte = f.read()

texte = texte.replace(".", ". <EOS> ")

with open("wikipediaIA.txt", "w", encoding="utf-8") as f:
    f.write(texte)
