import streamlit as st
import os
import tempfile
import re
import subprocess
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
    page_title="🎧 Ultra-Simple Audiobook Generator",
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

class UltraSimpleAudiobookGenerator:
    """Ultra-simple audiobook generator using only gTTS"""
    
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
    
    def create_audiobook_segments(self, text: str, progress_callback=None) -> List[Dict]:
        """Create audiobook segments with emotion analysis"""
        segments = self.split_into_segments(text)
        
        segment_data = []
        emotion_counts = {}
        
        for i, segment in enumerate(segments):
            if progress_callback:
                progress_callback(i + 1, len(segments), f"Processing segment {i+1}...")
            
            # Detect emotion
            emotion = self.simple_emotion_detection(segment)
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Generate speech
            audio_data = self.text_to_speech(segment, emotion)
            
            segment_data.append({
                'text': segment,
                'emotion': emotion,
                'audio_data': audio_data,
                'voice_style': self.emotion_settings[emotion]['tld']
            })
        
        return segment_data, emotion_counts
    
    def combine_audio_segments(self, segments: List[Dict], progress_callback=None) -> bytes:
        """Combine audio segments using ffmpeg"""
        try:
            # Create temporary directory for individual files
            with tempfile.TemporaryDirectory() as temp_dir:
                segment_files = []
                
                # Save each segment to a temporary file
                for i, segment in enumerate(segments):
                    if progress_callback:
                        progress_callback(i + 1, len(segments), f"Preparing segment {i+1}...")
                    
                    if segment['audio_data']:
                        temp_file = os.path.join(temp_dir, f"segment_{i:03d}.mp3")
                        with open(temp_file, 'wb') as f:
                            f.write(segment['audio_data'])
                        segment_files.append(temp_file)
                
                if not segment_files:
                    return b""
                
                # Create ffmpeg concat file
                concat_file = os.path.join(temp_dir, "concat_list.txt")
                with open(concat_file, 'w') as f:
                    for segment_file in segment_files:
                        f.write(f"file '{segment_file}'\n")
                
                # Output file
                output_file = os.path.join(temp_dir, "combined_audiobook.mp3")
                
                # Run ffmpeg to combine
                if progress_callback:
                    progress_callback(len(segments), len(segments), "Combining all segments...")
                
                cmd = [
                    'ffmpeg', '-f', 'concat', '-safe', '0', 
                    '-i', concat_file, '-c', 'copy', output_file, '-y'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Read the combined file
                    with open(output_file, 'rb') as f:
                        return f.read()
                else:
                    st.error(f"FFmpeg error: {result.stderr}")
                    return b""
                    
        except FileNotFoundError:
            st.warning("⚠️ FFmpeg not found. Install FFmpeg for audio combining, or download individual segments.")
            return b""
        except Exception as e:
            st.error(f"Error combining audio: {e}")
            return b""

# Initialize
if 'generator' not in st.session_state:
    st.session_state.generator = UltraSimpleAudiobookGenerator()

# UI Header
st.markdown('<h1 class="main-header">🎧 Ultra-Simple Audiobook Generator</h1>', unsafe_allow_html=True)

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
        if st.button("🎵 Generate Audiobook Segments", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total, message):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(f"{message} ({current}/{total})")
            
            try:
                with st.spinner("🎧 Creating your audiobook segments..."):
                    segments, emotion_counts = st.session_state.generator.create_audiobook_segments(
                        text, progress_callback=update_progress
                    )
                
                # Show results
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.success("🎉 Audiobook segments generated successfully!")
                
                # Emotion breakdown
                st.subheader("🎭 Emotion Analysis")
                total_segments = sum(emotion_counts.values())
                for emotion, count in emotion_counts.items():
                    percentage = (count / total_segments) * 100
                    voice_style = st.session_state.generator.emotion_settings[emotion]['tld']
                    st.write(f"**{emotion.title()}**: {count} segments ({percentage:.1f}%) - Voice: {voice_style}")
                
                st.write(f"📊 **Total Segments**: {total_segments}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Individual segment downloads
                st.subheader("🎧 Individual Audio Segments")
                
                for i, segment in enumerate(segments):
                    with st.expander(f"Segment {i+1}: {segment['emotion'].title()} ({segment['voice_style']})"):
                        st.write(f"**Text**: {segment['text'][:100]}...")
                        st.write(f"**Emotion**: {segment['emotion']}")
                        st.write(f"**Voice Style**: {segment['voice_style']}")
                        
                        if segment['audio_data']:
                            st.audio(segment['audio_data'], format='audio/mp3')
                            
                            # Download button for individual segment
                            st.download_button(
                                label=f"📥 Download Segment {i+1}",
                                data=segment['audio_data'],
                                file_name=f"segment_{i+1}_{segment['emotion']}.mp3",
                                mime="audio/mp3"
                            )
                
                # Combined audiobook option
                st.subheader("🎧 Complete Audiobook")
                
                if st.button("🔗 Combine Into Single Audiobook", type="secondary"):
                    combine_progress = st.progress(0)
                    combine_status = st.empty()
                    
                    def combine_progress_callback(current, total, message):
                        progress = current / total
                        combine_progress.progress(progress)
                        combine_status.text(message)
                    
                    combined_audio = st.session_state.generator.combine_audio_segments(
                        segments, progress_callback=combine_progress_callback
                    )
                    
                    if combined_audio:
                        st.success("✅ Combined audiobook created!")
                        
                        # Download button for combined audiobook
                        filename = f"complete_audiobook_{uploaded_file.name.replace('.pdf', '')}.mp3"
                        st.download_button(
                            label="📥 Download Complete Audiobook",
                            data=combined_audio,
                            file_name=filename,
                            mime="audio/mp3"
                        )
                        
                        # Audio player for combined
                        st.audio(combined_audio, format='audio/mp3')
                        
                    else:
                        st.error("❌ Could not combine audiobook. Try downloading individual segments.")
                
                # Instructions
                st.info("""
                💡 **Options**:
                - **Combined Audiobook**: Single file with all segments (requires FFmpeg)
                - **Individual Segments**: Download separately and combine manually
                
                If FFmpeg combining fails, you can still:
                1. Download all individual segments
                2. Use audio editing software (Audacity, GarageBand) to combine them
                3. Or use online tools to merge MP3 files
                """)
                
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
    <li>🧠 Pattern-based emotion detection</li>
    <li>🎤 Multiple voice styles (6 different accents/speeds)</li>
    <li>🎭 Emotional voice mapping</li>
    <li>📥 Individual segment downloads</li>
    <li>🔍 Detailed emotion analysis</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("👆 Upload a PDF file to get started!")
    
    st.markdown("""
    ### 🎯 How It Works:
    1. **Upload** your PDF book
    2. **Analyze** emotions in each text segment
    3. **Generate** audio with appropriate voice styles:
       - 😊 **Joy**: Australian accent (cheerful)
       - 😢 **Sadness**: Slow US voice (melancholic) 
       - ❤️ **Love**: British accent (warm)
       - 😲 **Surprise**: Canadian accent (animated)
       - 😠 **Anger**: Fast US voice (assertive)
       - 😰 **Fear**: Slow US voice (nervous)
    4. **Download** individual segments or combine them yourself
    """) 