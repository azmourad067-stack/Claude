import streamlit as st
import os
import tempfile
import re
from dotenv import load_dotenv
from gtts import gTTS
from spellchecker import SpellChecker
import speech_recognition as sr

# ===== CONFIGURATION INITIALE =====
st.set_page_config(
    page_title="English Conversation Partner",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== SIDEBAR - CONFIGURATION =====
with st.sidebar:
    st.title("🔧 Configuration")
    
    # Section API Key
    st.subheader("1. Clé API Groq")
    
    st.markdown("""
    **Pour obtenir une clé GRATUITE :**
    1. [console.groq.com](https://console.groq.com)
    2. Créez un compte gratuit
    3. **API Keys** → **Create API Key**
    4. Copiez la clé (`gsk_...`)
    """)
    
    # Input pour la clé API
    api_key_input = st.text_input(
        "Collez votre clé API Groq :",
        type="password",
        placeholder="gsk_votre_clé_ici",
        key="api_key_input"
    )
    
    if st.button("💾 Sauvegarder la clé", use_container_width=True):
        if api_key_input:
            st.session_state.groq_api_key = api_key_input
            st.success("✅ Clé sauvegardée !")
            st.rerun()
        else:
            st.error("❌ Veuillez entrer une clé valide")
    
    # Afficher état de la clé
    if 'groq_api_key' in st.session_state:
        st.success(f"✅ Clé configurée (...{st.session_state.groq_api_key[-4:]})")
    
    st.divider()
    
    # Paramètres de conversation
    st.subheader("2. Paramètres de conversation")
    
    # Options de langue pour gTTS
    voice_options = {
        "Anglais (US)": "en",
        "Anglais (UK)": "en-uk",
        "Anglais (Australie)": "en-au"
    }
    
    selected_voice = st.selectbox(
        "Accent anglais :",
        list(voice_options.keys()),
        index=0
    )
    
    conversation_topic = st.selectbox(
        "Sujet :",
        ["Daily Life", "Travel", "Food", "Hobbies", "Work", "Movies", "Sports", "Free Talk"]
    )
    
    difficulty_level = st.select_slider(
        "Niveau :",
        options=["Beginner", "Intermediate", "Advanced"],
        value="Intermediate"
    )
    
    # Options de correction
    st.checkbox("Corriger ma grammaire", value=True, key="correct_grammar")
    st.checkbox("Parler lentement", value=False, key="slow_speech")
    
    # Boutons d'action
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Effacer", use_container_width=True):
            for key in ['conversation_history', 'corrections']:
                if key in st.session_state:
                    st.session_state[key] = []
            st.rerun()
    
    with col2:
        if st.button("🔄 Actualiser", use_container_width=True):
            st.rerun()

# ===== FONCTIONS PRINCIPALES =====
def get_groq_client():
    """Obtenir le client Groq"""
    if 'groq_api_key' not in st.session_state:
        return None
    
    try:
        from groq import Groq
        return Groq(api_key=st.session_state.groq_api_key)
    except:
        return None

def transcribe_audio(audio_bytes):
    """Transcrire audio en texte"""
    try:
        # Sauvegarder temporairement
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        # Utiliser speech_recognition
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(tmp_path) as source:
            # Réduire le bruit
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio = recognizer.record(source)
            
            # Essayer Google Web Speech API (gratuit)
            try:
                text = recognizer.recognize_google(audio, language="en-US")
                return text
            except sr.UnknownValueError:
                return "Sorry, I couldn't understand the audio."
            except sr.RequestError:
                return "Speech service unavailable. Please type instead."
                
    except Exception as e:
        return f"Error: {str(e)[:50]}"

def text_to_speech_simple(text, lang='en', slow=False):
    """Synthèse vocale simple avec gTTS"""
    try:
        # Créer fichier temporaire
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_path = temp_file.name
        temp_file.close()
        
        # Générer audio avec gTTS
        tts = gTTS(text=text, lang=lang, slow=slow)
        tts.save(temp_path)
        
        return temp_path
    except Exception as e:
        st.error(f"Voice generation failed: {str(e)}")
        return None

def check_grammar(text):
    """Vérification simple de grammaire"""
    corrections = []
    
    # Initialiser correcteur d'orthographe
    try:
        spell = SpellChecker(language='en')
        
        # Vérifier l'orthographe
        words = text.split()
        misspelled = spell.unknown(words)
        
        for word in misspelled:
            suggestion = spell.correction(word)
            if suggestion and suggestion != word:
                corrections.append(f"Spelling: '{word}' → '{suggestion}'")
    except:
        pass  # Ignorer si spellchecker ne fonctionne pas
    
    # Vérifier erreurs courantes
    common_errors = {
        r'\bi (am|was)\b': 'I',
        r'your welcome': "you're welcome",
        r'could of': 'could have',
    }
    
    for pattern, correction in common_errors.items():
        if re.search(pattern, text, re.IGNORECASE):
            corrections.append(f"Grammar: Use '{correction}'")
    
    if corrections:
        return "💡 Suggestions:\n" + "\n".join(f"- {c}" for c in corrections[:3])
    return None

def get_ai_response(user_input, topic, level):
    """Obtenir réponse de Groq"""
    client = get_groq_client()
    
    # Mode démo si pas de client
    if not client:
        demo_responses = [
            "Hello! I'm here to help you practice English. How are you today?",
            "Nice to meet you! What would you like to talk about?",
            "Let's practice English together! Tell me about your day.",
            "I'm excited to help you improve your English! What's on your mind?"
        ]
        import random
        return random.choice(demo_responses)
    
    try:
        # Construire le prompt
        system_prompt = f"""You are a friendly English conversation partner.
        Topic: {topic}
        Student level: {level}
        
        Respond naturally (2-3 sentences).
        Ask follow-up questions.
        Be encouraging and helpful."""
        
        # Appeler Groq API
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
        return f"Let's continue our conversation! What else would you like to talk about?"

# ===== INTERFACE PRINCIPALE =====
st.title("🗣️ English Conversation Partner")

# Message si pas de clé API
if 'groq_api_key' not in st.session_state:
    st.info("""
    👋 **Bienvenue !**
    
    Pour commencer :
    1. Obtenez une clé API gratuite sur [Groq](https://console.groq.com)
    2. Collez-la dans la barre latérale à gauche
    3. Cliquez sur "Sauvegarder la clé"
    
    L'application fonctionne en mode démo sans clé API.
    """)

# Initialiser l'état
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'corrections' not in st.session_state:
    st.session_state.corrections = []

# Interface en deux colonnes
col1, col2 = st.columns([2, 1])

with col1:
    # Zone de conversation
    st.subheader("💬 Conversation")
    
    # Afficher messages
    if not st.session_state.conversation_history:
        st.info("💭 Start by saying hello! Use the microphone or type below.")
    
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style="background-color: #E3F2FD; padding: 12px; border-radius: 10px; margin: 8px 0; text-align: right;">
                <strong>You:</strong> {msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #F3F4F6; padding: 12px; border-radius: 10px; margin: 8px 0;">
                <strong>Assistant:</strong> {msg['content']}
            </div>
            """, unsafe_allow_html=True)
    
    # Afficher corrections
    if st.session_state.corrections:
        st.subheader("📝 Corrections")
        for correction in st.session_state.corrections[-2:]:
            st.warning(correction)

with col2:
    # Zone d'entrée
    st.subheader("🎤 Your Turn")
    
    # Option 1: Audio
    audio_data = st.audio_input(
        "Speak in English",
        key="audio_input"
    )
    
    # Option 2: Texte
    user_text = st.text_area(
        "Or type your message:",
        height=120,
        placeholder="Hello! How are you today?",
        label_visibility="collapsed"
    )
    
    # Bouton d'envoi
    if st.button("🚀 Send Message", type="primary", use_container_width=True):
        user_input = ""
        
        # Priorité à l'audio
        if audio_data:
            with st.spinner("🎤 Listening..."):
                user_input = transcribe_audio(audio_data)
                if user_input and "Error" not in user_input:
                    st.success("✅ Transcribed!")
        
        # Sinon texte
        if not user_input and user_text:
            user_input = user_text
        
        if user_input:
            # Ajouter à l'historique
            st.session_state.conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Vérifier grammaire si activé
            if st.session_state.get('correct_grammar', True):
                correction = check_grammar(user_input)
                if correction:
                    st.session_state.corrections.append(correction)
            
            # Obtenir réponse
            with st.spinner("💭 Thinking..."):
                response = get_ai_response(
                    user_input,
                    conversation_topic,
                    difficulty_level
                )
                
                # Ajouter réponse
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Générer audio
                lang_code = voice_options[selected_voice]
                slow = st.session_state.get('slow_speech', False)
                
                audio_file = text_to_speech_simple(
                    response,
                    lang=lang_code,
                    slow=slow
                )
                
                # Jouer audio
                if audio_file:
                    st.audio(audio_file, format='audio/mp3')
                    
                    # Option téléchargement
                    with open(audio_file, "rb") as f:
                        audio_bytes = f.read()
                    
                    st.download_button(
                        "📥 Download Audio",
                        data=audio_bytes,
                        file_name="english_response.mp3",
                        mime="audio/mp3",
                        use_container_width=True
                    )
            
            st.rerun()

# Section exercices
st.divider()
st.subheader("💪 Practice Exercises")

# Cards d'exercices
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎯 Daily Phrase", use_container_width=True):
        response = get_ai_response(
            "Give me one useful English phrase with explanation",
            "Vocabulary",
            difficulty_level
        )
        st.info(response)

with col2:
    if st.button("📚 Grammar Quiz", use_container_width=True):
        response = get_ai_response(
            "Create a short grammar quiz with 2 questions",
            "Grammar",
            difficulty_level
        )
        st.info(response)

with col3:
    if st.button("🗣️ Pronunciation", use_container_width=True):
        response = get_ai_response(
            "Give me a sentence to practice pronunciation",
            "Pronunciation",
            difficulty_level
        )
        st.info(response)
        
        # Dire lentement
        if response:
            audio_file = text_to_speech_simple(
                f"Repeat after me: {response}",
                lang='en',
                slow=True
            )
            if audio_file:
                st.audio(audio_file, format='audio/mp3')

# Pied de page
st.divider()
st.caption("⚡ Powered by Groq AI • 🆓 Free to use • 🗣️ Practice English daily")
