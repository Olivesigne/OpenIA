with open("wikipediaIA.txt", "r", encoding="utf-8") as f:
    texte = f.read()

chars = 0

for char in texte:
    chars += 1
    if chars < 25:
        print(char)

print(f"Il y a {chars} caractères dans le texte")
