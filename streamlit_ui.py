# ================================================================
# 🌟 STREAMLIT UI FOR EMOTIONAL AUDIOBOOK GENERATOR
# ================================================================

import streamlit as st
import os
import tempfile
import zipfile
from pathlib import Path
import torch
import time
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="🎧 Emotional Audiobook Generator",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .warning-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #2d3436;
        margin: 1rem 0;
    }
    .stProgress .st-bo {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generated_audiobook' not in st.session_state:
    st.session_state.generated_audiobook = None
if 'generation_complete' not in st.session_state:
    st.session_state.generation_complete = False
if 'processing' not in st.session_state:
    st.session_state.processing = False

def load_audiobook_generator():
    """Load the audiobook generator components"""
    try:
        # Import the modules (assuming they're in the same directory)
        from audiobook_classes import AudiobookConfig, EmotionAnalyzer, load_document
        from bark_narrator import BarkNarrator
        from sound_effects import SoundEffectMixer, AudiobookExporter
        from main_generator import EmotionalAudiobookGenerator
        
        return EmotionalAudiobookGenerator, AudiobookConfig
    except ImportError as e:
        st.error(f"❌ Failed to import audiobook modules: {str(e)}")
        st.info("💡 Make sure all the audiobook generator files are in the same directory as this Streamlit app.")
        return None, None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎧 Emotional Audiobook Generator</h1>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    <div class="feature-box">
    <h3>🌟 Transform your books into immersive audiobooks!</h3>
    <p>Upload a PDF or ePub file and get an emotional audiobook with adaptive sound effects based on the text's sentiment.</p>
    <ul>
        <li>🎙️ AI-powered emotional narration using Bark TTS</li>
        <li>🧠 Sentiment analysis for mood detection</li>
        <li>🎵 Adaptive background sound effects</li>
        <li>🎧 Professional audio mixing and export</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Load generator components
    EmotionalAudiobookGenerator, AudiobookConfig = load_audiobook_generator()
    
    if EmotionalAudiobookGenerator is None:
        st.stop()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Voice selection
    voice_options = {
        "Neutral (Speaker 6)": "v2/en_speaker_6",
        "Warm (Speaker 1)": "v2/en_speaker_1", 
        "Energetic (Speaker 7)": "v2/en_speaker_7",
        "Calm (Speaker 0)": "v2/en_speaker_0",
        "Dramatic (Speaker 8)": "v2/en_speaker_8",
        "Gentle (Speaker 2)": "v2/en_speaker_2",
        "Confident (Speaker 3)": "v2/en_speaker_3"
    }
    
    selected_voice = st.sidebar.selectbox(
        "🎤 Choose Voice Style",
        options=list(voice_options.keys()),
        index=0,
        help="Select the primary voice style for narration"
    )
    
    # Background volume
    bg_volume = st.sidebar.slider(
        "🔊 Background Volume",
        min_value=0.0,
        max_value=0.8,
        value=0.3,
        step=0.1,
        help="Volume of background sound effects relative to speech"
    )
    
    # Text chunk length
    chunk_length = st.sidebar.slider(
        "📝 Text Chunk Length",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Maximum characters per audio segment"
    )
    
    # Audio quality settings
    st.sidebar.subheader("🎵 Audio Quality")
    
    normalize_audio = st.sidebar.checkbox(
        "Normalize Audio",
        value=True,
        help="Apply audio normalization for consistent volume"
    )
    
    crossfade_duration = st.sidebar.slider(
        "Crossfade Duration (ms)",
        min_value=0,
        max_value=2000,
        value=500,
        step=100,
        help="Crossfade between audio segments"
    )
    
    # Output format
    output_format = st.sidebar.selectbox(
        "💾 Output Format",
        ["WAV (High Quality)", "MP3 (Compressed)"],
        index=0
    )
    
    # System info
    st.sidebar.subheader("🖥️ System Info")
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        st.sidebar.success("✅ GPU Available")
        gpu_name = torch.cuda.get_device_name(0)
        st.sidebar.info(f"GPU: {gpu_name}")
    else:
        st.sidebar.warning("⚠️ Using CPU (slower)")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📤 Upload Your Book")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF or ePub file",
            type=['pdf', 'epub'],
            help="Upload your book file to convert to an audiobook"
        )
        
        if uploaded_file is not None:
            # File info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
                "File type": uploaded_file.type
            }
            
            st.success("✅ File uploaded successfully!")
            
            with st.expander("📊 File Details"):
                for key, value in file_details.items():
                    st.write(f"**{key}:** {value}")
            
            # Preview option
            if st.checkbox("📖 Preview first few pages"):
                with st.spinner("Loading preview..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Extract preview text
                        from audiobook_classes import load_document
                        text = load_document(tmp_path)
                        
                        # Show preview
                        preview_text = text[:1000] + "..." if len(text) > 1000 else text
                        st.text_area("Preview", preview_text, height=200)
                        
                        # Clean up
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.error(f"❌ Error loading preview: {str(e)}")
    
    with col2:
        st.header("🎭 Mood Settings")
        
        # Mood presets
        mood_presets = {
            "📚 Neutral Reading": {
                "description": "Balanced narration for general content",
                "voice": "v2/en_speaker_6",
                "bg_volume": 0.2
            },
            "🌙 Bedtime Story": {
                "description": "Calm, soothing voice for relaxation",
                "voice": "v2/en_speaker_1",
                "bg_volume": 0.1
            },
            "🎭 Dramatic Reading": {
                "description": "Expressive narration with strong emotions",
                "voice": "v2/en_speaker_8",
                "bg_volume": 0.4
            },
            "⚡ Energetic": {
                "description": "Dynamic and engaging delivery",
                "voice": "v2/en_speaker_7",
                "bg_volume": 0.3
            },
            "🧘 Meditation": {
                "description": "Very calm and peaceful narration",
                "voice": "v2/en_speaker_0",
                "bg_volume": 0.05
            }
        }
        
        selected_preset = st.selectbox(
            "Choose a Mood Preset",
            options=list(mood_presets.keys()),
            help="Select a preset that matches your desired listening experience"
        )
        
        if selected_preset:
            preset = mood_presets[selected_preset]
            st.info(f"**{selected_preset}**\n\n{preset['description']}")
    
    # Generation section
    st.header("🚀 Generate Audiobook")
    
    if uploaded_file is not None:
        
        if st.button("🎧 Generate Emotional Audiobook", type="primary", use_container_width=True):
            
            # Create configuration based on selections
            if selected_preset and selected_preset in mood_presets:
                preset = mood_presets[selected_preset]
                config = AudiobookConfig(
                    bark_voice_preset=preset["voice"],
                    background_volume=preset["bg_volume"],
                    max_chunk_length=chunk_length,
                    normalize_audio=normalize_audio,
                    crossfade_duration=crossfade_duration,
                    use_gpu=gpu_available
                )
            else:
                config = AudiobookConfig(
                    bark_voice_preset=voice_options[selected_voice],
                    background_volume=bg_volume,
                    max_chunk_length=chunk_length,
                    normalize_audio=normalize_audio,
                    crossfade_duration=crossfade_duration,
                    use_gpu=gpu_available
                )
            
            # Set processing state
            st.session_state.processing = True
            st.session_state.generation_complete = False
            
            # Create progress containers
            progress_container = st.container()
            status_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            try:
                with status_container:
                    st.info("🚀 Starting audiobook generation...")
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    input_path = tmp_file.name
                
                # Create output directory
                output_dir = tempfile.mkdtemp()
                
                # Initialize generator
                generator = EmotionalAudiobookGenerator(config)
                
                # Generate audiobook with progress tracking
                status_text.text("📖 Loading document...")
                progress_bar.progress(10)
                
                status_text.text("🧠 Analyzing sentiment...")
                progress_bar.progress(25)
                
                status_text.text("🎙️ Loading TTS models...")
                progress_bar.progress(40)
                
                status_text.text("🎵 Generating narration...")
                progress_bar.progress(60)
                
                # Determine output filename
                output_filename = f"audiobook_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                output_filename += ".wav" if "WAV" in output_format else ".mp3"
                
                # Generate the audiobook
                audiobook_path = generator.generate_audiobook(
                    input_file=input_path,
                    output_dir=output_dir,
                    output_filename=output_filename
                )
                
                status_text.text("🎧 Finalizing audiobook...")
                progress_bar.progress(90)
                
                # Store result in session state
                st.session_state.generated_audiobook = audiobook_path
                st.session_state.generation_complete = True
                st.session_state.processing = False
                
                progress_bar.progress(100)
                status_text.text("✅ Generation complete!")
                
                # Clean up input file
                os.unlink(input_path)
                
            except Exception as e:
                st.session_state.processing = False
                st.error(f"❌ Error during generation: {str(e)}")
                st.info("💡 Try reducing the chunk length or using a smaller file for testing.")
    
    # Results section
    if st.session_state.generation_complete and st.session_state.generated_audiobook:
        st.header("🎉 Your Audiobook is Ready!")
        
        audiobook_path = st.session_state.generated_audiobook
        
        if os.path.exists(audiobook_path):
            # File info
            file_size = os.path.getsize(audiobook_path) / 1024 / 1024
            
            st.markdown(f"""
            <div class="success-box">
            <h3>📊 Audiobook Statistics</h3>
            <p><strong>File Size:</strong> {file_size:.1f} MB</p>
            <p><strong>Format:</strong> {Path(audiobook_path).suffix.upper()}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Audio player
            st.subheader("🎵 Preview Your Audiobook")
            with open(audiobook_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/wav')
            
            # Download section
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💾 Download")
                st.download_button(
                    label="📥 Download Audiobook",
                    data=audio_bytes,
                    file_name=os.path.basename(audiobook_path),
                    mime="audio/wav" if audiobook_path.endswith('.wav') else "audio/mpeg",
                    use_container_width=True
                )
            
            with col2:
                if st.button("🔄 Generate Another", use_container_width=True):
                    st.session_state.generation_complete = False
                    st.session_state.generated_audiobook = None
                    st.experimental_rerun()
        
        else:
            st.error("❌ Generated audiobook file not found.")
    
    elif not uploaded_file:
        st.markdown("""
        <div class="warning-box">
        <h3>👆 Get Started</h3>
        <p>Upload a PDF or ePub file above to begin generating your emotional audiobook!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <p>🎧 Emotional Audiobook Generator | Powered by Bark TTS & Transformers</p>
    <p>Made with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# ================================================================
# 🚀 DEPLOYMENT INSTRUCTIONS
# ================================================================

"""
To run this Streamlit app:

1. Install dependencies:
   pip install streamlit torch transformers pydub librosa soundfile PyPDF2 ebooklib beautifulsoup4 nltk

2. Make sure all audiobook generator files are in the same directory:
   - audiobook_classes.py
   - bark_narrator.py
   - sound_effects.py
   - main_generator.py
   - streamlit_ui.py

3. Run the app:
   streamlit run streamlit_ui.py

4. For deployment:
   - Use Streamlit Cloud, Heroku, or similar platforms
   - Ensure GPU availability for better performance
   - Consider memory limits for large files

Note: This app requires significant computational resources.
GPU acceleration is highly recommended for reasonable performance.
""" 