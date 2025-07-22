import streamlit as st
import os
import tempfile
from pathlib import Path
import time
from datetime import datetime
import re
from typing import List, Dict
import PyPDF2
import io

# Simple dependencies only
try:
    from gtts import gTTS
    from pydub import AudioSegment
    from pydub.generators import Sine, WhiteNoise
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    # Create dummy classes for type hints when imports fail
    class AudioSegment:
        pass

# Set page config
st.set_page_config(
    page_title="🎧 Simple Audiobook Generator",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SimpleAudiobookGenerator:
    """Simple audiobook generator using basic dependencies"""
    
    def __init__(self):
        self.emotion_patterns = {
            'joy': ['happy', 'joy', 'delight', 'wonderful', 'beautiful', 'love', 'smile', 'laugh'],
            'sadness': ['sad', 'cry', 'tear', 'sorrow', 'grief', 'mourn', 'melancholy'],
            'anger': ['angry', 'rage', 'fury', 'mad', 'furious', 'irritated'],
            'fear': ['afraid', 'fear', 'scared', 'terror', 'horror', 'frightened'],
            'surprise': ['surprise', 'amazed', 'astonished', 'shocked', 'unexpected'],
            'love': ['love', 'beloved', 'dear', 'sweetheart', 'romance', 'affection']
        }
        
        self.emotion_settings = {
            'joy': {'lang': 'en', 'slow': False, 'tld': 'com.au'},
            'sadness': {'lang': 'en', 'slow': True, 'tld': 'com'},
            'anger': {'lang': 'en', 'slow': False, 'tld': 'com'},
            'fear': {'lang': 'en', 'slow': True, 'tld': 'com'},
            'surprise': {'lang': 'en', 'slow': False, 'tld': 'ca'},
            'love': {'lang': 'en', 'slow': True, 'tld': 'co.uk'},
            'neutral': {'lang': 'en', 'slow': False, 'tld': 'com'}
        }
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    def simple_emotion_detection(self, text: str) -> str:
        """Simple pattern-based emotion detection"""
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, patterns in self.emotion_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        if emotion_scores:
            return max(emotion_scores, key=emotion_scores.get)
        return 'neutral'
    
    def split_into_segments(self, text: str) -> List[str]:
        """Split text into readable segments"""
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split by sentences
        sentences = re.split(r'[.!?]+', text)
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_segment) + len(sentence) > 200 and current_segment:
                segments.append(current_segment.strip())
                current_segment = sentence
            else:
                current_segment += " " + sentence if current_segment else sentence
        
        if current_segment.strip():
            segments.append(current_segment.strip())
            
        return segments
    
    def generate_background_sound(self, emotion: str, duration_ms: int) -> 'AudioSegment':
        """Generate simple background sounds"""
        try:
            if emotion == 'sadness':
                # Soft rain sound
                rain = WhiteNoise().to_audio_segment(duration=duration_ms)
                return rain.low_pass_filter(800).apply_gain(-25)
            elif emotion == 'joy':
                # Happy melody
                melody = Sine(440).to_audio_segment(duration=duration_ms//2)
                melody += Sine(550).to_audio_segment(duration=duration_ms//2)
                return melody.apply_gain(-30)
            elif emotion == 'fear':
                # Tension drone
                tension = Sine(150).to_audio_segment(duration=duration_ms)
                return tension.apply_gain(-35)
            else:
                # Gentle ambient
                ambient = Sine(330).to_audio_segment(duration=duration_ms)
                return ambient.apply_gain(-35)
        except:
            return AudioSegment.silent(duration=duration_ms)
    
    def text_to_speech(self, text: str, emotion: str) -> 'AudioSegment':
        """Convert text to speech with emotion"""
        try:
            settings = self.emotion_settings.get(emotion, self.emotion_settings['neutral'])
            tts = gTTS(text=text, **settings)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                audio = AudioSegment.from_mp3(tmp_file.name)
                os.unlink(tmp_file.name)
            
            # Apply emotional effects
            if emotion == 'joy':
                audio = audio.speedup(playback_speed=1.1)
            elif emotion == 'sadness':
                audio = audio.speedup(playback_speed=0.9)
            elif emotion == 'fear':
                audio = audio.speedup(playback_speed=1.15)
            
            return audio
        except Exception as e:
            st.error(f"TTS Error: {e}")
            return AudioSegment.silent(duration=2000)
    
    def create_audiobook(self, text: str, progress_callback=None):
        """Create audiobook from text"""
        segments = self.split_into_segments(text)
        
        final_audio = AudioSegment.empty()
        emotion_counts = {}
        
        for i, segment in enumerate(segments):
            if progress_callback:
                progress_callback(i + 1, len(segments), f"Processing segment {i+1}...")
            
            # Detect emotion
            emotion = self.simple_emotion_detection(segment)
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Generate speech
            speech = self.text_to_speech(segment, emotion)
            
            # Add background sound
            bg_sound = self.generate_background_sound(emotion, len(speech))
            mixed = speech.overlay(bg_sound)
            
            # Add pause and combine
            mixed += AudioSegment.silent(duration=500)
            
            if len(final_audio) > 0:
                final_audio = final_audio.append(mixed, crossfade=300)
            else:
                final_audio = mixed
        
        return final_audio.normalize(), emotion_counts

# Initialize
if 'generator' not in st.session_state:
    st.session_state.generator = SimpleAudiobookGenerator()

# UI Header
st.markdown('<h1 class="main-header">🎧 Simple Audiobook Generator</h1>', unsafe_allow_html=True)

if not HAS_AUDIO:
    st.error("📦 Missing audio dependencies! Please install:")
    st.code("pip install gtts pydub soundfile PyPDF2")
    st.stop()

# Sidebar
st.sidebar.header("📁 Upload & Settings")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=['pdf'],
    help="Upload a PDF book to convert to audiobook"
)

if uploaded_file:
    st.sidebar.success(f"📄 File loaded: {uploaded_file.name}")

# Main interface
if uploaded_file is not None:
    # Extract text
    with st.spinner("📖 Extracting text from PDF..."):
        text = st.session_state.generator.extract_text_from_pdf(uploaded_file)
    
    if text:
        st.success(f"✅ Extracted {len(text)} characters from PDF")
        
        # Show preview
        with st.expander("📝 Text Preview"):
            st.text_area("Extracted Text", text[:1000] + "..." if len(text) > 1000 else text, height=200)
        
        # Generate audiobook button
        if st.button("🎵 Generate Audiobook", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total, message):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(f"{message} ({current}/{total})")
            
            try:
                with st.spinner("🎧 Creating your audiobook..."):
                    audio, emotion_counts = st.session_state.generator.create_audiobook(
                        text, progress_callback=update_progress
                    )
                
                # Show results
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.success("🎉 Audiobook generated successfully!")
                
                # Emotion breakdown
                st.subheader("🎭 Emotion Analysis")
                total_segments = sum(emotion_counts.values())
                for emotion, count in emotion_counts.items():
                    percentage = (count / total_segments) * 100
                    st.write(f"**{emotion.title()}**: {count} segments ({percentage:.1f}%)")
                
                # Audio info
                duration = len(audio) / 1000
                st.write(f"🕐 **Duration**: {duration/60:.1f} minutes")
                st.write(f"📊 **Segments**: {total_segments}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Export audio
                with st.spinner("💾 Preparing download..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        audio.export(tmp_file.name, format="wav")
                        
                        with open(tmp_file.name, 'rb') as f:
                            audio_bytes = f.read()
                        
                        os.unlink(tmp_file.name)
                
                # Download button
                filename = f"audiobook_{uploaded_file.name.replace('.pdf', '')}.wav"
                st.download_button(
                    label="📥 Download Audiobook (WAV)",
                    data=audio_bytes,
                    file_name=filename,
                    mime="audio/wav"
                )
                
                # Audio player
                st.subheader("🎧 Preview")
                st.audio(audio_bytes, format='audio/wav')
                
            except Exception as e:
                st.error(f"❌ Error generating audiobook: {e}")
    else:
        st.error("❌ Could not extract text from PDF")

else:
    # Welcome screen
    st.markdown("""
    <div class="feature-box">
    <h3>🌟 Features</h3>
    <ul>
    <li>📚 PDF text extraction</li>
    <li>🧠 Simple emotion detection</li>
    <li>🎤 Multiple voice styles (different accents/speeds)</li>
    <li>🎵 Adaptive background sounds</li>
    <li>🎧 Professional audio mixing</li>
    <li>💾 High-quality WAV export</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("👆 Upload a PDF file to get started!") 