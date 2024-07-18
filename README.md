# LLM Chat PDF

Cette application est un chatbot alimenté par un modèle de langage (LLM) utilisant Streamlit, LangChain et OpenAI. Elle permet de poser des questions sur le contenu d'un document PDF et d'obtenir des réponses pertinentes.

## Fonctionnalités

- Téléchargement et lecture de documents PDF.
- Extraction et traitement de texte à partir des PDF.
- Division du texte en morceaux gérables pour le traitement.
- Utilisation des embeddings OpenAI pour la recherche de similarités.
- Stockage et chargement des embeddings en utilisant FAISS.
- Interface utilisateur simple et intuitive.

## Installation

### Prérequis

- Python 3.7 ou supérieur
- [OpenAI API Key](https://platform.openai.com/account/api-keys)
- Compte GitHub (si possible)

### Étapes

1. Clonez le dépôt :

   ```bash
   git clone https://github.com/<votre_nom_utilisateur>/<nom_du_dépôt>.git
   cd <nom_du_dépôt>

# Créez un environnement virtuel et activez-le :

- python -m venv env
- source env/bin/activate  # Sur Windows: .\env\Scripts\activate

# Installez les dépendances :

- pip install -r requirements.txt

# Créez un fichier .env et ajoutez votre clé API OpenAI :

OPENAI_API_KEY=your_openai_api_key

# Exécutez l'application Streamlit :
- streamlit run "le nom de votre fichier python" exemple : streamlit run app.py
