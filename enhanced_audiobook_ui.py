import streamlit as st
import os
import tempfile
import re
import subprocess
import wave
import struct
import math
from typing import List, Dict
import PyPDF2
import io

# Ultra-simple dependencies only
try:
    from gtts import gTTS
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

# Set page config
st.set_page_config(
    page_title="🎧 Enhanced Audiobook Generator",
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

class SoundEffectGenerator:
    """Generate simple sound effects using pure Python"""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
    
    def generate_sine_wave(self, frequency, duration, amplitude=0.3):
        """Generate a sine wave"""
        samples = int(self.sample_rate * duration)
        wave_data = []
        for i in range(samples):
            value = amplitude * math.sin(2 * math.pi * frequency * i / self.sample_rate)
            wave_data.append(int(value * 32767))
        return wave_data
    
    def generate_noise(self, duration, amplitude=0.1):
        """Generate white noise"""
        import random
        samples = int(self.sample_rate * duration)
        wave_data = []
        for i in range(samples):
            value = amplitude * (random.random() * 2 - 1)
            wave_data.append(int(value * 32767))
        return wave_data
    
    def create_background_music(self, emotion: str, duration: float) -> bytes:
        """Create background music based on emotion"""
        wave_data = []
        
        if emotion == 'sadness':
            # Soft rain sound (filtered noise)
            rain = self.generate_noise(duration, 0.15)
            # Add low-frequency component for depth
            low_tone = self.generate_sine_wave(80, duration, 0.1)
            wave_data = [min(32767, max(-32767, r + l)) for r, l in zip(rain, low_tone)]
            
        elif emotion == 'joy':
            # Happy melody with harmonics
            base = self.generate_sine_wave(440, duration, 0.2)  # A4
            harmony1 = self.generate_sine_wave(550, duration, 0.15)  # C#5
            harmony2 = self.generate_sine_wave(660, duration, 0.1)   # E5
            wave_data = [min(32767, max(-32767, b + h1 + h2)) for b, h1, h2 in zip(base, harmony1, harmony2)]
            
        elif emotion == 'fear':
            # Low tension drone with tremolo
            base_freq = 100
            tremolo_freq = 6  # Hz
            base = self.generate_sine_wave(base_freq, duration, 0.2)
            tremolo = self.generate_sine_wave(tremolo_freq, duration, 0.1)
            wave_data = [min(32767, max(-32767, int(b * (1 + t/32767)))) for b, t in zip(base, tremolo)]
            
        elif emotion == 'love':
            # Warm pad with soft harmonics
            base = self.generate_sine_wave(220, duration, 0.2)  # A3
            harmony = self.generate_sine_wave(330, duration, 0.15)  # E4
            wave_data = [min(32767, max(-32767, b + h)) for b, h in zip(base, harmony)]
            
        elif emotion == 'anger':
            # Rhythmic percussive element
            beat_duration = 0.1
            silence_duration = 0.4
            total_beats = int(duration / (beat_duration + silence_duration))
            
            for beat in range(total_beats):
                # Sharp attack
                beat_samples = self.generate_noise(beat_duration, 0.3)
                silence_samples = [0] * int(self.sample_rate * silence_duration)
                wave_data.extend(beat_samples + silence_samples)
            
        else:  # neutral, surprise
            # Gentle ambient
            ambient = self.generate_sine_wave(330, duration, 0.15)
            wave_data = ambient
        
        return self.wave_data_to_bytes(wave_data)
    
    def create_contextual_sound(self, text: str, duration: float) -> bytes:
        """Create contextual sound effects based on text content"""
        text_lower = text.lower()
        
        # Sound effect keywords
        if any(word in text_lower for word in ['door', 'creak', 'hinge']):
            # Door creak: low-frequency sweep
            return self.door_creak_effect(duration)
        elif any(word in text_lower for word in ['footstep', 'walk', 'step']):
            # Footsteps: rhythmic low thumps
            return self.footstep_effect(duration)
        elif any(word in text_lower for word in ['wind', 'breeze', 'gust']):
            # Wind: filtered noise with frequency modulation
            return self.wind_effect(duration)
        elif any(word in text_lower for word in ['fire', 'flame', 'crackling']):
            # Fire: crackling noise
            return self.fire_effect(duration)
        elif any(word in text_lower for word in ['water', 'river', 'stream', 'ocean']):
            # Water: flowing sound
            return self.water_effect(duration)
        elif any(word in text_lower for word in ['thunder', 'storm', 'lightning']):
            # Thunder: low rumble
            return self.thunder_effect(duration)
        else:
            # No specific sound, return silence
            return self.wave_data_to_bytes([0] * int(self.sample_rate * duration))
    
    def door_creak_effect(self, duration: float) -> bytes:
        """Generate door creak sound"""
        # Frequency sweep from 200Hz to 150Hz
        samples = int(self.sample_rate * duration)
        wave_data = []
        for i in range(samples):
            progress = i / samples
            freq = 200 - (50 * progress)  # Sweep down
            value = 0.3 * math.sin(2 * math.pi * freq * i / self.sample_rate)
            # Add some noise for realism
            noise = 0.1 * (2 * (i % 7) / 7 - 1)  # Simple pseudo-noise
            wave_data.append(int((value + noise) * 32767))
        return self.wave_data_to_bytes(wave_data)
    
    def footstep_effect(self, duration: float) -> bytes:
        """Generate footstep sound"""
        step_duration = 0.15
        step_interval = 0.6
        steps = int(duration / step_interval)
        
        wave_data = []
        for step in range(steps):
            # Each step: quick low-frequency thump
            step_samples = int(self.sample_rate * step_duration)
            for i in range(step_samples):
                # Exponential decay
                decay = math.exp(-10 * i / step_samples)
                value = 0.4 * decay * math.sin(2 * math.pi * 80 * i / self.sample_rate)
                wave_data.append(int(value * 32767))
            
            # Silence between steps
            silence_samples = int(self.sample_rate * (step_interval - step_duration))
            wave_data.extend([0] * silence_samples)
        
        return self.wave_data_to_bytes(wave_data)
    
    def wind_effect(self, duration: float) -> bytes:
        """Generate wind sound"""
        # Filtered noise with slow frequency modulation
        noise = self.generate_noise(duration, 0.2)
        modulation = self.generate_sine_wave(0.5, duration, 0.3)  # Slow modulation
        
        wave_data = [min(32767, max(-32767, int(n * (1 + m/32767)))) for n, m in zip(noise, modulation)]
        return self.wave_data_to_bytes(wave_data)
    
    def fire_effect(self, duration: float) -> bytes:
        """Generate fire crackling sound"""
        # High-frequency noise with random bursts
        import random
        samples = int(self.sample_rate * duration)
        wave_data = []
        
        for i in range(samples):
            # Base crackling (high-freq noise)
            base = 0.15 * (random.random() * 2 - 1)
            
            # Random pops
            if random.random() < 0.001:  # 0.1% chance of pop
                pop = 0.3 * math.sin(2 * math.pi * 2000 * i / self.sample_rate)
                base += pop
            
            wave_data.append(int(base * 32767))
        
        return self.wave_data_to_bytes(wave_data)
    
    def water_effect(self, duration: float) -> bytes:
        """Generate water flowing sound"""
        # Multiple sine waves with slow modulation
        base1 = self.generate_sine_wave(150, duration, 0.1)
        base2 = self.generate_sine_wave(200, duration, 0.08)
        base3 = self.generate_sine_wave(250, duration, 0.06)
        noise = self.generate_noise(duration, 0.05)
        
        wave_data = [min(32767, max(-32767, b1 + b2 + b3 + n)) for b1, b2, b3, n in zip(base1, base2, base3, noise)]
        return self.wave_data_to_bytes(wave_data)
    
    def thunder_effect(self, duration: float) -> bytes:
        """Generate thunder rumble"""
        # Very low frequency with noise
        rumble = self.generate_sine_wave(40, duration, 0.3)
        noise = self.generate_noise(duration, 0.1)
        
        wave_data = [min(32767, max(-32767, r + n)) for r, n in zip(rumble, noise)]
        return self.wave_data_to_bytes(wave_data)
    
    def wave_data_to_bytes(self, wave_data) -> bytes:
        """Convert wave data to WAV bytes"""
        # Create a simple WAV file in memory
        buffer = io.BytesIO()
        
        # WAV header
        buffer.write(b'RIFF')
        buffer.write(struct.pack('<I', 36 + len(wave_data) * 2))  # File size
        buffer.write(b'WAVE')
        buffer.write(b'fmt ')
        buffer.write(struct.pack('<I', 16))  # Subchunk1 size
        buffer.write(struct.pack('<H', 1))   # Audio format (PCM)
        buffer.write(struct.pack('<H', 1))   # Num channels (mono)
        buffer.write(struct.pack('<I', self.sample_rate))  # Sample rate
        buffer.write(struct.pack('<I', self.sample_rate * 2))  # Byte rate
        buffer.write(struct.pack('<H', 2))   # Block align
        buffer.write(struct.pack('<H', 16))  # Bits per sample
        buffer.write(b'data')
        buffer.write(struct.pack('<I', len(wave_data) * 2))  # Data size
        
        # Audio data
        for sample in wave_data:
            buffer.write(struct.pack('<h', sample))
        
        buffer.seek(0)
        return buffer.read()

class EnhancedAudiobookGenerator:
    """Enhanced audiobook generator with background music and sound effects"""
    
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
            'joy': {'lang': 'en', 'slow': False, 'tld': 'com.au'},      # Australian - cheerful
            'sadness': {'lang': 'en', 'slow': True, 'tld': 'com'},     # US slow - melancholic
            'anger': {'lang': 'en', 'slow': False, 'tld': 'com'},      # US fast - assertive
            'fear': {'lang': 'en', 'slow': True, 'tld': 'com'},        # US slow - nervous
            'surprise': {'lang': 'en', 'slow': False, 'tld': 'ca'},    # Canadian - animated
            'love': {'lang': 'en', 'slow': True, 'tld': 'co.uk'},      # British - warm
            'neutral': {'lang': 'en', 'slow': False, 'tld': 'com'}     # US standard
        }
        
        self.sound_generator = SoundEffectGenerator()
    
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
    
    def text_to_speech(self, text: str, emotion: str) -> bytes:
        """Convert text to speech with emotion using gTTS"""
        try:
            settings = self.emotion_settings.get(emotion, self.emotion_settings['neutral'])
            tts = gTTS(text=text, **settings)
            
            # Save to memory buffer
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            return audio_buffer.read()
            
        except Exception as e:
            st.error(f"TTS Error: {e}")
            return b""
    
    def create_enhanced_segment(self, text: str, emotion: str, duration: float) -> Dict:
        """Create enhanced audio segment with background music and effects"""
        # Generate speech
        speech_audio = self.text_to_speech(text, emotion)
        
        # Generate background music based on emotion
        background_music = self.sound_generator.create_background_music(emotion, duration)
        
        # Generate contextual sound effects
        sound_effects = self.sound_generator.create_contextual_sound(text, duration)
        
        return {
            'text': text,
            'emotion': emotion,
            'speech_audio': speech_audio,
            'background_music': background_music,
            'sound_effects': sound_effects,
            'voice_style': self.emotion_settings[emotion]['tld']
        }
    
    def create_audiobook_segments(self, text: str, progress_callback=None):
        """Create enhanced audiobook segments with background music and effects"""
        segments = self.split_into_segments(text)
        
        segment_data = []
        emotion_counts = {}
        sound_effects_used = []
        
        for i, segment in enumerate(segments):
            if progress_callback:
                progress_callback(i + 1, len(segments), f"Processing segment {i+1}...")
            
            # Detect emotion
            emotion = self.simple_emotion_detection(segment)
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Estimate duration (rough: 150 words per minute for TTS)
            word_count = len(segment.split())
            estimated_duration = word_count / 150 * 60  # Convert to seconds
            
            # Create enhanced segment
            enhanced_segment = self.create_enhanced_segment(segment, emotion, estimated_duration)
            
            # Check for sound effects
            if any(keyword in segment.lower() for keywords in [
                ['door', 'creak'], ['footstep', 'walk'], ['wind', 'breeze'], 
                ['fire', 'flame'], ['water', 'river'], ['thunder', 'storm']
            ] for keyword in keywords):
                sound_effects_used.append(f"Segment {i+1}")
            
            segment_data.append(enhanced_segment)
        
        return segment_data, emotion_counts, sound_effects_used

# Initialize
if 'generator' not in st.session_state:
    st.session_state.generator = EnhancedAudiobookGenerator()

# UI Header
st.markdown('<h1 class="main-header">🎧 Enhanced Audiobook Generator</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem;">✨ With Emotional Background Music & Contextual Sound Effects ✨</p>', unsafe_allow_html=True)

if not HAS_TTS:
    st.error("📦 Missing gTTS! Please install:")
    st.code("pip install gtts PyPDF2")
    st.stop()

# Sidebar
st.sidebar.header("📁 Upload & Settings")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=['pdf'],
    help="Upload a PDF book to convert to audiobook"
)

# Audio enhancement options
st.sidebar.subheader("🎵 Audio Enhancement")
include_background_music = st.sidebar.checkbox("🎼 Background Music", value=True, help="Add emotional background music")
include_sound_effects = st.sidebar.checkbox("🔊 Sound Effects", value=True, help="Add contextual sound effects")

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
        if st.button("🎵 Generate Enhanced Audiobook", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total, message):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(f"{message} ({current}/{total})")
            
            try:
                with st.spinner("🎧 Creating enhanced audiobook segments..."):
                    segments, emotion_counts, sound_effects = st.session_state.generator.create_audiobook_segments(
                        text, progress_callback=update_progress
                    )
                
                # Store segments in session state
                st.session_state.segments = segments
                st.session_state.emotion_counts = emotion_counts
                st.session_state.sound_effects = sound_effects
                
                # Show results
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.success("🎉 Enhanced audiobook segments generated successfully!")
                
                # Three columns for results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("🎭 Emotion Analysis")
                    total_segments = sum(emotion_counts.values())
                    for emotion, count in emotion_counts.items():
                        percentage = (count / total_segments) * 100
                        voice_style = st.session_state.generator.emotion_settings[emotion]['tld']
                        st.write(f"**{emotion.title()}**: {count} ({percentage:.1f}%)")
                        st.caption(f"Voice: {voice_style}")
                
                with col2:
                    st.subheader("🎵 Background Music")
                    music_types = set()
                    for segment in segments:
                        if segment['emotion'] == 'sadness':
                            music_types.add("🌧️ Rain & Low Tones")
                        elif segment['emotion'] == 'joy':
                            music_types.add("🎶 Happy Harmonies")
                        elif segment['emotion'] == 'fear':
                            music_types.add("😰 Tension Drones")
                        elif segment['emotion'] == 'love':
                            music_types.add("💕 Warm Ambient")
                        elif segment['emotion'] == 'anger':
                            music_types.add("🥁 Rhythmic Beats")
                        else:
                            music_types.add("🎵 Gentle Ambient")
                    
                    for music in sorted(music_types):
                        st.write(f"• {music}")
                
                with col3:
                    st.subheader("🔊 Sound Effects")
                    if sound_effects:
                        st.write("Sound effects detected in:")
                        for effect in sound_effects:
                            st.write(f"• {effect}")
                    else:
                        st.write("• No specific sound effects detected")
                        st.caption("(Based on keywords like door, wind, fire, etc.)")
                
                st.write(f"📊 **Total Enhanced Segments**: {total_segments}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error generating audiobook: {e}")

# Show enhanced segments if available
if 'segments' in st.session_state:
    st.subheader("🎧 Enhanced Audio Segments")
    
    # Sample a few segments to show
    segments = st.session_state.segments
    sample_segments = segments[:3] if len(segments) > 3 else segments
    
    for i, segment in enumerate(sample_segments):
        with st.expander(f"Enhanced Segment {i+1}: {segment['emotion'].title()}"):
            st.write(f"**Text**: {segment['text'][:150]}...")
            st.write(f"**Emotion**: {segment['emotion']}")
            st.write(f"**Voice Style**: {segment['voice_style']}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("🎤 **Speech Audio**")
                if segment['speech_audio']:
                    st.audio(segment['speech_audio'], format='audio/mp3')
            
            with col2:
                st.write("🎵 **Background Music**")
                if segment['background_music']:
                    st.audio(segment['background_music'], format='audio/wav')
            
            with col3:
                st.write("🔊 **Sound Effects**")
                if segment['sound_effects']:
                    st.audio(segment['sound_effects'], format='audio/wav')
    
    if len(segments) > 3:
        st.info(f"Showing first 3 segments. Total: {len(segments)} enhanced segments created.")
    
    # Note about combining
    st.warning("""
    🔧 **Note**: This creates separate audio tracks (speech, music, effects). 
    To create a final mixed audiobook, these would need to be combined using audio editing software or advanced mixing algorithms.
    
    **What you get**:
    - 🎤 Speech with emotional voices
    - 🎵 Background music matching emotions  
    - 🔊 Contextual sound effects
    """)

else:
    # Welcome screen
    st.markdown("""
    <div class="feature-box">
    <h3>🌟 Enhanced Features</h3>
    <ul>
    <li>📚 PDF text extraction</li>
    <li>🧠 Emotion detection with contextual analysis</li>
    <li>🎤 6 different emotional voice styles</li>
    <li>🎵 <strong>Adaptive background music</strong> (sad rain, happy melodies, etc.)</li>
    <li>🔊 <strong>Contextual sound effects</strong> (door creaks, footsteps, wind, etc.)</li>
    <li>🎭 Professional emotion-to-audio mapping</li>
    <li>🔍 Detailed audio analysis and breakdown</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("👆 Upload a PDF file to get started!")
    
    st.markdown("""
    ### 🎯 Enhanced Audio Features:
    
    **🎵 Background Music Based on Emotion:**
    - 😢 **Sadness**: Soft rain sounds + low ambient tones
    - 😊 **Joy**: Upbeat harmonies + cheerful melodies  
    - 😰 **Fear**: Tension drones + tremolo effects
    - ❤️ **Love**: Warm ambient pads + soft harmonics
    - 😠 **Anger**: Rhythmic percussive elements
    - 😐 **Neutral**: Gentle ambient soundscape
    
    **🔊 Contextual Sound Effects:**
    - 🚪 **Door mentions** → Door creak sounds
    - 👣 **Footsteps** → Walking rhythm effects
    - 💨 **Wind** → Whoosh and breeze sounds
    - 🔥 **Fire** → Crackling flame effects
    - 💧 **Water** → Flowing river sounds
    - ⛈️ **Thunder** → Deep rumble effects
    
    **🎤 Emotional Voice Styles:**
    - Different accents and speeds based on detected emotions
    - Professional voice mapping for each emotional context
    """) 