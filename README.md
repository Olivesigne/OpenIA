# Projet IA - OpenIA

Ce projet est un système d'intelligence artificielle qui a pour objectif de pouvoir discuter naturellement et se rapprocher d'un modèle transformers (comme ChatGPT, Le Chat de Mistral ...). Il a été développé avec Python 3.10.10.

---

## 📁 Arborescence du projet

```
/
├── README.md                        # Ce fichier
├── models                           # Modèles sauvegardés
├── requirements.txt                 # Dépendances Python
└── src
    ├── datasets
    │   └── ...                      # Jeux de données
    └── scripts
    ├── test
    │   └── Test.py                  # Script de test du modèle
    ├── train
    │   └── Train.py                 # Script d'entraînement du modèle
    └── utils
        ├── EOS.py                   # Gestion des fins de phrase
        ├── NbCaracteres.py          # Utilitaire de comptage de caractères
        └── WikipediaData.py         # Prétraitement des données Wikipédia
```

---

## 🔧 Installation

1. Clone le dépôt :
   ```bash
   git clone https://github.com/Olivesigne/OpenIA.git
   cd OpenIA
   ```
2. Installe Python 3.10.10 : \
   [Installer Python 3.10.10](https://www.python.org/downloads/release/python-31010/)


3. Crée un environnement virtuel (optionnel mais recommandé) :
   ```bash
   python -m venv venv
   ./venv/Scripts/activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

4. Installe les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚖️ Licence

Ce projet est sous licence **MIT**. Voir le fichier `LICENSE` pour plus de détails.

---

## 📜 Conditions supplémentaires

En complément de la licence MIT, les conditions suivantes s’appliquent :

- ✅ **Utilisation privée autorisée** : Vous pouvez utiliser ce projet librement à titre personnel ou pour l'apprentissage.
- 🔧 **Modification autorisée** : Vous êtes autorisé à modifier le code pour vos besoins personnels.
- 🚫 **Interdiction de redistribution de versions modifiées** : Toute redistribution publique du projet, en partie ou en totalité, incluant des versions modifiées, est **interdite sans autorisation écrite explicite** de l’auteur.
- 📛 **Pas de fork public** : Ne créez pas de forks publics ou dérivés accessibles publiquement.
- 📩 Pour toute demande spéciale (usage commercial, publication d’une version modifiée), contactez Olivesigne via GitHub ou Discord.

---

## 🤝 Contribuer

Les contributions sont bienvenues uniquement après discussion préalable. Pour proposer une amélioration :
1. Ouvre une issue ou un post dans le salon [#openia](https://discord.gg/FkyqS4NB3S) sur le serveur discord pour en discuter.
2. Si approuvé, crée une branche et une pull request.

---

## 📬 Contact

Créé par **Olivesigne**  
GitHub : [@Olivesigne](https://github.com/Olivesigne)\
Discord : [@le.de](https://discord.com/channels/@me)

---

Merci d'utiliser ce projet de manière respectueuse et responsable ! 🙏
