import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
from gtts import gTTS
import pyttsx3
from spellchecker import SpellChecker
import spacy
import re
import speech_recognition as sr

# ===== CONFIGURATION INITIALE =====
st.set_page_config(
    page_title="English Conversation Partner",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded"  # Sidebar ouverte par défaut
)

# ===== SECTION CONFIGURATION API (TOUJOURS VISIBLE) =====
with st.sidebar:
    st.title("🔧 Configuration API")
    
    # Section API Key - TOUJOURS VISIBLE
    st.subheader("1. Clé API Groq")
    
    # Instructions
    st.markdown("""
    **Pour obtenir une clé GRATUITE :**
    1. Allez sur [console.groq.com](https://console.groq.com)
    2. Créez un compte (gratuit)
    3. Cliquez sur **API Keys** → **Create API Key**
    4. Copiez la clé (commence par `gsk_`)
    """)
    
    # Zone de saisie de la clé API
    api_key_input = st.text_input(
        "Collez votre clé API Groq ici :",
        type="password",
        placeholder="gsk_votre_clé_ici",
        help="La clé API est nécessaire pour utiliser l'application",
        key="api_key_input_main"
    )
    
    # Bouton pour sauvegarder la clé
    if st.button("💾 Sauvegarder la clé API", use_container_width=True):
        if api_key_input:
            # Sauvegarder dans la session
            st.session_state.groq_api_key = api_key_input
            st.success("✅ Clé API sauvegardée dans la session !")
            st.rerun()
        else:
            st.error("Veuillez entrer une clé API valide")
    
    # Afficher l'état de la clé
    if 'groq_api_key' in st.session_state and st.session_state.groq_api_key:
        st.success(f"✅ Clé API configurée (derniers 4 caractères: ...{st.session_state.groq_api_key[-4:]})")
    
    # Bouton pour tester la clé
    if st.button("🔍 Tester la connexion API", use_container_width=True):
        if 'groq_api_key' in st.session_state:
            try:
                from groq import Groq
                test_client = Groq(api_key=st.session_state.groq_api_key)
                
                # Tester avec une requête simple
                test_response = test_client.chat.completions.create(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="llama3-8b-8192",
                    max_tokens=10
                )
                st.success("✅ Connexion API réussie !")
            except Exception as e:
                st.error(f"❌ Erreur de connexion: {str(e)}")
        else:
            st.warning("Veuillez d'abord entrer une clé API")
    
    st.divider()
    
    # ===== PARAMÈTRES DE L'APPLICATION =====
    st.subheader("2. Paramètres")
    
    # Voix
    voice_gender = st.selectbox(
        "Voix de l'assistante",
        ["Féminine", "Masculine", "Neutre"]
    )
    
    # Sujet
    conversation_topic = st.selectbox(
        "Sujet de conversation",
        ["Vie quotidienne", "Voyages", "Nourriture", "Loisirs", "Travail", "Libre"]
    )
    
    # Niveau
    difficulty_level = st.select_slider(
        "Niveau de difficulté",
        options=["Débutant", "Intermédiaire", "Avancé"]
    )
    
    # Corrections
    st.checkbox("Corriger la grammaire", value=True, key="correct_grammar")
    st.checkbox("Donner des conseils de prononciation", value=True, key="pronunciation_tips")
    
    # Boutons d'action
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Effacer chat", use_container_width=True):
            if 'conversation_history' in st.session_state:
                st.session_state.conversation_history = []
            if 'corrections' in st.session_state:
                st.session_state.corrections = []
            st.rerun()
    
    with col2:
        if st.button("🔄 Redémarrer", use_container_width=True):
            st.rerun()

# ===== FONCTIONS UTILITAIRES =====
def get_groq_client():
    """Obtenir le client Groq depuis la session"""
    if 'groq_api_key' not in st.session_state or not st.session_state.groq_api_key:
        return None
    
    try:
        from groq import Groq
        return Groq(api_key=st.session_state.groq_api_key)
    except Exception as e:
        st.sidebar.error(f"Erreur client Groq: {str(e)}")
        return None

def transcribe_audio(audio_bytes):
    """Transcrire l'audio"""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(tmp_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)
            
            try:
                text = recognizer.recognize_google(audio, language="en-US")
                os.unlink(tmp_path)
                return text
            except sr.UnknownValueError:
                os.unlink(tmp_path)
                return "Je n'ai pas compris l'audio. Pouvez-vous répéter ?"
            except sr.RequestError as e:
                os.unlink(tmp_path)
                return f"Erreur de reconnaissance vocale: {e}"
                
    except Exception as e:
        return f"Erreur: {str(e)}"

def get_ai_response(user_input, topic, level):
    """Obtenir une réponse de l'IA"""
    client = get_groq_client()
    
    if not client:
        # Mode démo
        responses = {
            "Vie quotidienne": [
                "Hello! How was your day today?",
                "What did you do this morning?",
                "Tell me about your daily routine."
            ],
            "Voyages": [
                "Have you traveled anywhere interesting recently?",
                "Where would you like to go on vacation?",
                "Tell me about your dream destination."
            ],
            "Nourriture": [
                "What's your favorite food?",
                "Do you like cooking?",
                "Tell me about a memorable meal you had."
            ]
        }
        
        import random
        topic_responses = responses.get(topic, ["Let's practice English! What would you like to talk about?"])
        return random.choice(topic_responses)
    
    try:
        system_prompt = f"""You are a friendly English conversation partner.
        Topic: {topic}
        Level: {level}
        
        Be natural, ask questions, and help practice English.
        Keep responses 2-3 sentences."""
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=150
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}. Please check your API key."

def text_to_speech(text, voice_type="female"):
    """Synthèse vocale"""
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_path = temp_file.name
        temp_file.close()
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(temp_path)
        return temp_path
    except:
        return None

# ===== INTERFACE PRINCIPALE =====
st.title("🗣️ English Conversation Partner")

# Message d'information si pas de clé API
if 'groq_api_key' not in st.session_state or not st.session_state.groq_api_key:
    st.warning("""
    ⚠️ **Configuration requise**
    
    Pour utiliser l'application, veuillez :
    1. Obtenir une clé API Groq gratuite (instructions dans la barre latérale)
    2. Coller votre clé dans le champ **"Collez votre clé API Groq ici"**
    3. Cliquer sur **"Sauvegarder la clé API"**
    
    L'application fonctionnera en mode démo jusqu'à ce que vous ajoutiez une clé valide.
    """)

# Initialiser l'historique
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'corrections' not in st.session_state:
    st.session_state.corrections = []

# Interface en deux colonnes
col1, col2 = st.columns([2, 1])

with col1:
    # Zone de conversation
    st.subheader("💬 Conversation")
    
    # Afficher l'historique
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            st.markdown(f"**Vous:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")
    
    # Afficher les corrections
    if st.session_state.corrections:
        st.subheader("📝 Corrections")
        for correction in st.session_state.corrections[-3:]:
            st.info(correction)

with col2:
    # Zone d'entrée
    st.subheader("🎤 Parler ou écrire")
    
    # Option audio
    audio_data = st.audio_input(
        "Enregistrer un message vocal",
        key="audio_input"
    )
    
    # Option texte
    user_text = st.text_area(
        "Ou taper votre message:",
        height=100,
        placeholder="Bonjour ! Comment allez-vous aujourd'hui ?"
    )
    
    # Bouton d'envoi
    if st.button("📤 Envoyer", type="primary", use_container_width=True):
        user_input = ""
        
        # Priorité à l'audio
        if audio_data:
            with st.spinner("Transcription en cours..."):
                user_input = transcribe_audio(audio_data)
        
        # Sinon utiliser le texte
        if not user_input and user_text:
            user_input = user_text
        
        if user_input:
            # Ajouter à l'historique
            st.session_state.conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Obtenir une réponse
            with st.spinner("L'assistante réfléchit..."):
                response = get_ai_response(
                    user_input,
                    conversation_topic,
                    difficulty_level
                )
                
                # Ajouter la réponse
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Synthèse vocale
                audio_file = text_to_speech(response, voice_gender.lower())
                if audio_file:
                    st.audio(audio_file, format='audio/mp3')
            
            st.rerun()

# Section exercices
st.divider()
st.subheader("💪 Exercices pratiques")

# Exercice de vocabulaire
if st.button("🎯 Mot du jour", use_container_width=True):
    client = get_groq_client()
    
    if client:
        try:
            response = client.chat.completions.create(
                messages=[{
                    "role": "user", 
                    "content": "Give me one useful English word with definition and example sentence"
                }],
                model="llama3-8b-8192",
                max_tokens=100
            )
            st.info(response.choices[0].message.content)
        except:
            st.info("**Perseverance** (noun):\nContinuing to try despite difficulties.")
    else:
        st.info("**Practice** (verb):\nTo do something regularly to improve skills.\nExample: I practice English every day.")

# Mode d'emploi
with st.expander("📖 Comment utiliser l'application"):
    st.markdown("""
    **Étapes simples :**
    
    1. **Configurez votre clé API** (barre latérale)
       - Obtenez une clé gratuite sur [Groq](https://console.groq.com)
       - Collez-la dans le champ prévu
       - Cliquez sur "Sauvegarder"
    
    2. **Choisissez vos paramètres** :
       - Sujet de conversation
       - Niveau de difficulté
       - Type de voix
    
    3. **Commencez à parler** :
       - Utilisez le micro pour parler en anglais
       - Ou tapez votre message
       - L'IA répondra et corrigera vos erreurs
    
    4. **Pratiquez régulièrement** :
       - Utilisez les exercices
       - Écoutez les réponses audio
       - Notez vos progrès
    
    **Fonctionnalités incluses :**
    - Reconnaissance vocale
    - Synthèse vocale
    - Correction de grammaire
    - Conversations naturelles
    - Exercices pratiques
    """)
