
import streamlit as st
import altair as alt
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import openai
import os
from dotenv import load_dotenv
from pathlib import Path
import tiktoken
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI

# Load environment variables from .env
load_dotenv()

# OpenAI API key 
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("La clé API OpenAI n'est pas définie. Veuillez la définir dans le fichier .env ou comme variable d'environnement.")

# Pour definir le chemin absolu (Path) 
absolute_directory_path = Path("C:/llm_rag/doc")

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        margin-bottom: 20px;    
    }
    .main-text-input>div>input {
        border-radius: 5px;
        border: 1px solid blue;
        padding: 10px;
        width: 100%;
        
    }
          
   .social-links {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    .social-links a {
        margin: 0 10px;
    }
    .social-links img {
        width: 30px;
        height: 30px;
    }
    </style>
    """, unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.title("LLM Chat PDF ")
    st.markdown(
        """
        ## A propos
        C'est une application RAG réalisée en utilisant ces téchnologies : 
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models) LLM model
        """)
    
    #Pour l'ajout d'espace
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.write(" ### Réalisé par  [Bassirou CISSE](https://github.com/cisse17) ")
    st.write("Etudiant en école d'ingénieure informatique")
    
     # Ajout GitHub and LinkedIn links 
    st.markdown("""
        <div class="social-links">
            <a href="https://github.com/cisse17?tab=repositories" target="_blank">
                <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo">
            </a>
            <a href="https://www.linkedin.com/in/bassirou-mbacké-cissé-683529263" target="_blank">
                <img src="https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg" alt="LinkedIn logo">
            </a>
        </div>
        """, unsafe_allow_html=True)


# main fonction
def main():
    st.markdown("<h1 class='main-header'>Chat avec votre document PDF</h1>", unsafe_allow_html=True)
    pdf = st.file_uploader('Joindre votre document PDF', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Generate file name for FAISS index
        file_name = pdf.name[:-4].replace(" ", "_")
        faiss_index_file = absolute_directory_path / f"{file_name}.faiss"

        # Créer le répertoire s'il n'existe pas
        absolute_directory_path.mkdir(parents=True, exist_ok=True)

        if faiss_index_file.exists():
            try:
                db = FAISS.load_local(str(faiss_index_file), OpenAIEmbeddings(openai_api_key=openai_api_key), allow_dangerous_deserialization=True)
                #st.write("Embeddings chargés à partir du disque.")
            except Exception as e:
                st.write(f"Échec du chargement des intégrations à partir du disque: {e}")
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            db = FAISS.from_texts(chunks, embedding=embeddings)
            db.save_local(str(faiss_index_file))
            #st.write("Calcul des intégrations terminé et enregistré sur le disque.")

    # Accept user questions/query
    query = st.text_input("Posez une question à propos de votre document :", placeholder="Votre question ici..." )
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    if query:
        docs = db.similarity_search(query=query, k=3) # recherche de similarité 
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        st.write(response)

if __name__ == "__main__":
    main()








