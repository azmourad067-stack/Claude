# Main.py - Application complète en un seul fichier

import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import tempfile
from gtts import gTTS
import pyttsx3
from spellchecker import SpellChecker
import spacy
import re

# Charger les variables d'environnement
load_dotenv()

# Initialiser OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Charger le modèle SpaCy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # Télécharger si non présent
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Configuration de la page
st.set_page_config(
    page_title="English Conversation Partner",
    page_icon="🗣️",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .conversation-box {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        min-height: 300px;
        max-height: 400px;
        overflow-y: auto;
    }
    .user-message {
        background-color: #DBEAFE;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: right;
    }
    .ai-message {
        background-color: #E5E7EB;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .correction {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 8px;
        margin: 5px 0;
        font-size: 0.9rem;
    }
    .stAudio {
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">🗣️ English Conversation Partner</h1>', unsafe_allow_html=True)

# Initialiser l'état de la session
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'corrections' not in st.session_state:
    st.session_state.corrections = []

# Fonctions utilitaires dans le même fichier
def transcribe_audio_streamlit(audio_bytes):
    """Transcrire l'audio avec Whisper"""
    try:
        # Sauvegarder temporairement
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        # Transcrire avec Whisper
        with open(tmp_path, "rb") as audio:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio
            )
        
        # Nettoyer
        os.unlink(tmp_path)
        return transcript.text
        
    except Exception as e:
        st.error(f"Erreur de transcription: {str(e)}")
        return None

def text_to_speech_simple(text, voice_type="female"):
    """Convertir texte en parole"""
    try:
        # Créer un fichier temporaire
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_path = temp_file.name
        temp_file.close()
        
        # Utiliser gTTS (en ligne, meilleure qualité)
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(temp_path)
        return temp_path
        
    except Exception as e:
        # Fallback local
        try:
            engine = pyttsx3.init()
            
            # Configurer la voix
            voices = engine.getProperty('voices')
            if voice_type == "female" and len(voices) > 1:
                engine.setProperty('voice', voices[1].id)
            elif voice_type == "male" and len(voices) > 0:
                engine.setProperty('voice', voices[0].id)
            
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            
            engine.save_to_file(text, temp_path)
            engine.runAndWait()
            return temp_path
            
        except Exception as e2:
            st.error(f"Erreur TTS: {str(e2)}")
            return None

def check_grammar_simple(text):
    """Vérifier la grammaire"""
    corrections = []
    
    # Vérificateur d'orthographe
    spell = SpellChecker(language='en')
    
    # Erreurs courantes
    common_errors = {
        r'\bi (am|was)\b': 'I',
        r'your welcome': "you're welcome",
        r'could of': 'could have',
        r'would of': 'would have',
        r'should of': 'should have',
        r'\bme (and|&)\b': '... and I',
    }
    
    for pattern, correction in common_errors.items():
        if re.search(pattern, text, re.IGNORECASE):
            corrections.append(f"Erreur courante : utilisez '{correction}'")
    
    # Vérifier l'orthographe
    words = text.split()
    misspelled = spell.unknown(words)
    
    if misspelled:
        for word in misspelled:
            correction = spell.correction(word)
            if correction and correction != word:
                corrections.append(f"Orthographe : '{word}' → '{correction}'")
    
    if corrections:
        return "💡 Suggestions :\n" + "\n".join(f"- {c}" for c in corrections[:3])
    return None

def get_ai_response_simple(user_input, context, level, topic):
    """Obtenir une réponse de l'IA"""
    system_prompt = f"""Tu es une amie anglaise qui aide à pratiquer l'anglais.
    Niveau de l'étudiant : {level}
    Sujet de conversation : {topic}
    
    Sois amicale, naturelle et encourageante.
    Pose des questions pour continuer la conversation.
    Utilise un vocabulaire adapté au niveau.
    Sois positive et supportive !
    
    Contexte : {context}
    
    Message de l'étudiant : {user_input}
    
    Réponds naturellement comme une amie (2-3 phrases)."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content
        
    except Exception as e:
        # Réponses de secours
        import random
        fallback = [
            "That's interesting! Tell me more about that.",
            "I understand. What happened next?",
            "That sounds great! How did you feel about it?",
            "Thanks for sharing. What are your thoughts on this?",
        ]
        return random.choice(fallback)

# Barre latérale pour les paramètres
with st.sidebar:
    st.header("⚙️ Paramètres")
    
    # Voix
    st.subheader("Voix")
    voice_gender = st.selectbox(
        "Voix de l'assistante",
        ["Féminine", "Masculine", "Neutre"]
    )
    
    # Sujet de conversation
    st.subheader("Conversation")
    conversation_topic = st.selectbox(
        "Sujet du jour",
        ["Vie quotidienne", "Voyages", "Nourriture", "Loisirs", "Travail", "Libre"]
    )
    
    difficulty_level = st.select_slider(
        "Niveau",
        options=["Débutant", "Intermédiaire", "Avancé"]
    )
    
    # Corrections
    st.subheader("Corrections")
    correct_grammar = st.checkbox("Corriger la grammaire", value=True)
    
    if st.button("🧹 Effacer la conversation", type="secondary"):
        st.session_state.conversation_history = []
        st.session_state.corrections = []
        st.rerun()

# Interface principale
col1, col2 = st.columns([2, 1])

with col1:
    # Afficher la conversation
    st.subheader("💬 Conversation")
    
    conversation_container = st.container()
    with conversation_container:
        for message in st.session_state.conversation_history:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>Vous :</strong> {message["content"]}</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-message"><strong>Assistante :</strong> {message["content"]}</div>', 
                          unsafe_allow_html=True)
    
    # Afficher les corrections
    if st.session_state.corrections:
        st.subheader("📝 Corrections")
        for correction in st.session_state.corrections[-3:]:
            st.markdown(f'<div class="correction">{correction}</div>', unsafe_allow_html=True)

with col2:
    # Entrée vocale
    st.subheader("🎤 Parlez maintenant")
    
    # Enregistrement audio
    audio_data = st.audio_input(
        "Cliquez pour enregistrer",
        sample_rate=16000,
        help="Cliquez pour commencer, cliquez à nouveau pour arrêter"
    )
    
    # Ou texte
    text_input = st.text_area(
        "Ou tapez votre message :",
        height=100,
        placeholder="Bonjour ! Comment ça va aujourd'hui ?"
    )
    
    # Bouton d'envoi
    if st.button("💬 Envoyer le message", type="primary", use_container_width=True):
        user_input = ""
        
        # Traiter l'audio
        if audio_data:
            with st.spinner("Écoute en cours..."):
                user_input = transcribe_audio_streamlit(audio_data)
                if user_input:
                    st.success("Transcription réussie !")
        
        # Sinon utiliser le texte
        if not user_input and text_input:
            user_input = text_input
        
        if user_input:
            # Ajouter à l'historique
            st.session_state.conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Vérifier la grammaire
            if correct_grammar:
                correction = check_grammar_simple(user_input)
                if correction:
                    st.session_state.corrections.append(correction)
            
            # Obtenir une réponse
            with st.spinner("Réflexion en cours..."):
                # Contexte
                context = "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in st.session_state.conversation_history[-4:]
                ])
                
                # Réponse IA
                ai_response = get_ai_response_simple(
                    user_input,
                    context,
                    difficulty_level,
                    conversation_topic
                )
                
                # Ajouter à l'historique
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": ai_response
                })
                
                # Synthèse vocale
                with st.spinner("Préparation de la réponse vocale..."):
                    audio_file = text_to_speech_simple(
                        ai_response,
                        voice_gender.lower()
                    )
                    
                    # Jouer l'audio
                    if audio_file:
                        st.audio(audio_file, format='audio/mp3')
                        
                        # Téléchargement optionnel
                        with open(audio_file, "rb") as f:
                            audio_bytes = f.read()
                        
                        st.download_button(
                            label="📥 Télécharger l'audio",
                            data=audio_bytes,
                            file_name="reponse_anglaise.mp3",
                            mime="audio/mp3"
                        )
            
            st.rerun()

# Exercices de pratique
st.divider()
st.subheader("💪 Exercices pratiques")

tab1, tab2, tab3 = st.tabs(["Vocabulaire", "Grammaire", "Prononciation"])

with tab1:
    if st.button("🎯 Nouveau mot du jour"):
        with st.spinner("Recherche d'un mot intéressant..."):
            word_response = get_ai_response_simple(
                "Donne-moi un mot utile en anglais avec sa définition et un exemple de phrase",
                "",
                difficulty_level,
                "Vocabulaire"
            )
            st.info(word_response)

with tab2:
    grammar_point = st.selectbox(
        "Point de grammaire",
        ["Present Tense", "Past Tense", "Future Tense", "Conditionals", "Prepositions"],
        key="grammar_select"
    )
    if st.button(f"📚 Pratiquer {grammar_point}"):
        with st.spinner("Préparation de l'exercice..."):
            exercise = get_ai_response_simple(
                f"Crée un court exercice pour pratiquer {grammar_point} avec 3 questions",
                "",
                difficulty_level,
                "Grammaire"
            )
            st.info(exercise)

with tab3:
    if st.button("👅 Virelangue pour la prononciation"):
        with st.spinner("Recherche d'un virelangue..."):
            tongue_twister = get_ai_response_simple(
                "Donne-moi un virelangue anglais adapté à mon niveau pour pratiquer la prononciation",
                "",
                difficulty_level,
                "Prononciation"
            )
            st.info(tongue_twister)
            
            # Dire lentement
            slow_audio = text_to_speech_simple(
                f"Say this slowly: {tongue_twister}",
                voice_gender.lower()
            )
            if slow_audio:
                st.audio(slow_audio, format='audio/mp3')
