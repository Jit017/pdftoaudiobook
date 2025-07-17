# ================================================================
# 🎧 EMOTIONAL AUDIOBOOK GENERATOR FOR GOOGLE COLAB
# ================================================================
# Convert PDF/ePub books to emotional audiobooks with adaptive sound effects
# Built for Google Colab with GPU support and free open-source tools

# ================================================================
# 📥 1. INSTALL DEPENDENCIES
# ================================================================

# Run these commands in Google Colab cells:
"""
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q git+https://github.com/suno-ai/bark.git
!pip install -q transformers datasets accelerate
!pip install -q pydub librosa soundfile
!pip install -q PyPDF2 ebooklib beautifulsoup4
!pip install -q nltk textstat scipy
!pip install -q IPython tqdm reportlab ipywidgets

!apt-get update -qq
!apt-get install -y -qq ffmpeg
"""

import os
import re
import io
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm.auto import tqdm
import pickle
import json
import time

# Audio processing
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import normalize
from scipy import signal

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import textstat

# ML models
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from bark import SAMPLE_RATE, generate_audio, preload_models
from bark.generation import generate_text_semantic

# File processing
import PyPDF2
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# Display and utilities
from IPython.display import Audio, display, HTML, clear_output

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("✅ NLTK data downloaded successfully!")
except:
    print("⚠️ NLTK download skipped - run manually if needed")

# Suppress warnings
warnings.filterwarnings('ignore')

print("✅ All dependencies loaded successfully!")

# ================================================================
# 🎛️ CONFIGURATION & DATA CLASSES
# ================================================================

@dataclass
class AudiobookConfig:
    """Configuration for audiobook generation"""
    # Text processing
    max_chunk_length: int = 200  # Max characters per audio chunk
    sentence_overlap: int = 1    # Sentences to overlap between chunks
    
    # Audio settings
    sample_rate: int = SAMPLE_RATE
    audio_format: str = "wav"
    normalize_audio: bool = True
    
    # Bark TTS settings
    bark_voice_preset: str = "v2/en_speaker_6"  # Default voice
    bark_text_temp: float = 0.7
    bark_waveform_temp: float = 0.7
    
    # Sentiment analysis
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    emotion_threshold: float = 0.6
    
    # Sound effects
    background_volume: float = 0.3  # Background sound volume relative to speech
    crossfade_duration: int = 500   # Crossfade between segments (ms)
    
    # Processing
    use_gpu: bool = True
    batch_size: int = 4
    save_intermediate: bool = True

@dataclass
class TextSegment:
    """Represents a segment of text with metadata"""
    text: str
    sentiment: Dict[str, float]
    emotion: str
    confidence: float
    start_index: int
    end_index: int
    audio_path: Optional[str] = None

@dataclass
class SoundEffect:
    """Sound effect configuration"""
    name: str
    file_path: str
    emotions: List[str]
    volume: float = 0.3
    loop: bool = True

# ================================================================
# 📚 2. LOAD AND PARSE PDF/EPUB
# ================================================================

class DocumentLoader:
    """Handles loading and parsing of PDF and ePub files"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                print(f"📖 Processing PDF with {total_pages} pages...")
                
                for page_num in tqdm(range(total_pages), desc="Extracting pages"):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    
            return DocumentLoader._clean_text(text)
            
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    @staticmethod
    def extract_text_from_epub(file_path: str) -> str:
        """Extract text from ePub file"""
        try:
            book = epub.read_epub(file_path)
            text = ""
            
            chapters = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
            print(f"📖 Processing ePub with {len(chapters)} chapters...")
            
            for chapter in tqdm(chapters, desc="Extracting chapters"):
                soup = BeautifulSoup(chapter.get_content(), 'html.parser')
                chapter_text = soup.get_text()
                text += chapter_text + "\n"
                
            return DocumentLoader._clean_text(text)
            
        except Exception as e:
            raise Exception(f"Error reading ePub: {str(e)}")
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove page numbers and headers/footers (basic)
        text = re.sub(r'\n\d+\n', '\n', text)
        text = re.sub(r'\n[A-Z\s]{10,}\n', '\n', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
        
        return text.strip()

def load_document(file_path: str) -> str:
    """Load document based on file extension"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    extension = file_path.suffix.lower()
    
    if extension == '.pdf':
        return DocumentLoader.extract_text_from_pdf(str(file_path))
    elif extension == '.epub':
        return DocumentLoader.extract_text_from_epub(str(file_path))
    else:
        raise ValueError(f"Unsupported file format: {extension}")

# ================================================================
# 🧠 3. ANALYZE SENTIMENT
# ================================================================

class EmotionAnalyzer:
    """Analyzes text sentiment and emotion for adaptive audio generation"""
    
    def __init__(self, config: AudiobookConfig):
        self.config = config
        self.sentiment_pipeline = None
        self.emotion_mapping = {
            'NEGATIVE': ['horror', 'tension', 'sadness', 'anger'],
            'POSITIVE': ['joy', 'excitement', 'peaceful', 'uplifting'],
            'NEUTRAL': ['calm', 'neutral', 'ambient']
        }
        
    def load_models(self):
        """Load sentiment analysis models"""
        print("🧠 Loading sentiment analysis models...")
        
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.config.sentiment_model,
                device=0 if self.config.use_gpu and torch.cuda.is_available() else -1
            )
            print("✅ Sentiment models loaded successfully!")
            
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            # Fallback to simpler model
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1
            )
            print("✅ Fallback sentiment model loaded!")
    
    def analyze_text_segments(self, text: str) -> List[TextSegment]:
        """Split text into segments and analyze each for emotion"""
        if self.sentiment_pipeline is None:
            self.load_models()
        
        # Split into sentences
        sentences = sent_tokenize(text)
        segments = []
        
        print(f"🔍 Analyzing sentiment for {len(sentences)} sentences...")
        
        # Group sentences into chunks
        current_chunk = ""
        current_start = 0
        
        for i, sentence in enumerate(tqdm(sentences, desc="Processing sentences")):
            # Check if adding this sentence would exceed chunk length
            if len(current_chunk + sentence) > self.config.max_chunk_length and current_chunk:
                # Process current chunk
                segment = self._create_segment(
                    current_chunk.strip(), 
                    current_start, 
                    current_start + len(current_chunk)
                )
                segments.append(segment)
                
                # Start new chunk
                current_chunk = sentence + " "
                current_start = current_start + len(current_chunk)
            else:
                current_chunk += sentence + " "
        
        # Process final chunk
        if current_chunk.strip():
            segment = self._create_segment(
                current_chunk.strip(),
                current_start,
                current_start + len(current_chunk)
            )
            segments.append(segment)
        
        return segments
    
    def _create_segment(self, text: str, start_idx: int, end_idx: int) -> TextSegment:
        """Create a text segment with sentiment analysis"""
        try:
            # Get sentiment
            result = self.sentiment_pipeline(text)[0]
            sentiment_label = result['label']
            confidence = result['score']
            
            # Map to emotion
            emotion = self._map_sentiment_to_emotion(sentiment_label, confidence)
            
            # Create sentiment dict
            sentiment_dict = {
                'label': sentiment_label,
                'score': confidence,
                'compound': confidence if sentiment_label == 'POSITIVE' else -confidence
            }
            
            return TextSegment(
                text=text,
                sentiment=sentiment_dict,
                emotion=emotion,
                confidence=confidence,
                start_index=start_idx,
                end_index=end_idx
            )
            
        except Exception as e:
            print(f"⚠️ Error analyzing segment: {e}")
            # Return neutral segment
            return TextSegment(
                text=text,
                sentiment={'label': 'NEUTRAL', 'score': 0.5, 'compound': 0.0},
                emotion='neutral',
                confidence=0.5,
                start_index=start_idx,
                end_index=end_idx
            )
    
    def _map_sentiment_to_emotion(self, sentiment: str, confidence: float) -> str:
        """Map sentiment to specific emotion for sound effects"""
        if confidence < self.config.emotion_threshold:
            return 'neutral'
        
        emotions = self.emotion_mapping.get(sentiment, ['neutral'])
        
        # For now, return the first emotion, but this could be more sophisticated
        return emotions[0]

# ================================================================
# 🎙️ 4. GENERATE EMOTIONAL NARRATION (BARK)
# ================================================================

class BarkNarrator:
    """Handles text-to-speech generation using Bark with emotional variation"""
    
    def __init__(self, config: AudiobookConfig):
        self.config = config
        self.models_loaded = False
        
        # Voice presets for different emotions
        self.emotion_voices = {
            'neutral': 'v2/en_speaker_6',
            'joy': 'v2/en_speaker_9',
            'excitement': 'v2/en_speaker_7',
            'peaceful': 'v2/en_speaker_1',
            'uplifting': 'v2/en_speaker_5',
            'horror': 'v2/en_speaker_8',
            'tension': 'v2/en_speaker_4',
            'sadness': 'v2/en_speaker_2',
            'anger': 'v2/en_speaker_3',
            'calm': 'v2/en_speaker_0',
            'ambient': 'v2/en_speaker_6'
        }
    
    def load_models(self):
        """Load Bark TTS models"""
        if self.models_loaded:
            return
            
        print("🎙️ Loading Bark TTS models...")
        print("⏳ This may take a few minutes on first run...")
        
        try:
            # Preload models
            preload_models()
            self.models_loaded = True
            print("✅ Bark models loaded successfully!")
            
        except Exception as e:
            raise Exception(f"Failed to load Bark models: {str(e)}")
    
    def generate_segment_audio(self, segment: TextSegment, output_dir: str) -> str:
        """Generate audio for a text segment with emotional voice"""
        if not self.models_loaded:
            self.load_models()
        
        # Select voice based on emotion
        voice_preset = self.emotion_voices.get(segment.emotion, self.config.bark_voice_preset)
        
        # Add emotional markers to text for Bark
        emotional_text = self._add_emotional_markers(segment.text, segment.emotion)
        
        try:
            # Generate audio
            audio_array = generate_audio(
                emotional_text,
                history_prompt=voice_preset,
                text_temp=self.config.bark_text_temp,
                waveform_temp=self.config.bark_waveform_temp,
            )
            
            # Save audio file
            output_path = os.path.join(output_dir, f"segment_{segment.start_index}_{segment.end_index}.wav")
            sf.write(output_path, audio_array, self.config.sample_rate)
            
            return output_path
            
        except Exception as e:
            print(f"⚠️ Error generating audio for segment: {e}")
            # Create silent audio as fallback
            silence = np.zeros(int(self.config.sample_rate * 2))  # 2 seconds of silence
            output_path = os.path.join(output_dir, f"segment_{segment.start_index}_{segment.end_index}_error.wav")
            sf.write(output_path, silence, self.config.sample_rate)
            return output_path
    
    def _add_emotional_markers(self, text: str, emotion: str) -> str:
        """Add emotional context to text for better Bark generation"""
        
        # Emotional prefixes to influence Bark's voice generation
        emotion_prefixes = {
            'joy': "[laughs] ",
            'excitement': "[excitedly] ",
            'peaceful': "[softly] ",
            'uplifting': "[warmly] ",
            'horror': "[fearfully] ",
            'tension': "[tensely] ",
            'sadness': "[sadly] ",
            'anger': "[angrily] ",
            'calm': "[calmly] ",
            'neutral': "",
            'ambient': "[quietly] "
        }
        
        prefix = emotion_prefixes.get(emotion, "")
        return prefix + text
    
    def generate_audiobook_narration(self, segments: List[TextSegment], output_dir: str) -> List[str]:
        """Generate narration for all segments"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        audio_files = []
        
        print(f"🎙️ Generating narration for {len(segments)} segments...")
        
        for i, segment in enumerate(tqdm(segments, desc="Generating audio")):
            try:
                audio_path = self.generate_segment_audio(segment, output_dir)
                audio_files.append(audio_path)
                segment.audio_path = audio_path
                
                # Progress update
                if (i + 1) % 10 == 0:
                    print(f"   Generated {i + 1}/{len(segments)} segments")
                    
            except Exception as e:
                print(f"⚠️ Error processing segment {i}: {e}")
                continue
        
        print(f"✅ Generated audio for {len(audio_files)} segments!")
        return audio_files

# ================================================================
# 🔊 5. ADD ADAPTIVE SOUND FX
# ================================================================

class SoundEffectMixer:
    """Handles background sound effects based on text emotion"""
    
    def __init__(self, config: AudiobookConfig):
        self.config = config
        self.sound_effects = {}
        self.setup_default_sounds()
    
    def setup_default_sounds(self):
        """Setup default sound effects (these would be downloaded/generated)"""
        
        # Note: In a real implementation, you'd download these from freesound.org
        # or generate them programmatically. For this demo, we'll create simple tones.
        
        self.sound_templates = {
            'horror': {'freq': [100, 150], 'type': 'noise', 'volume': 0.2},
            'tension': {'freq': [200, 300], 'type': 'drone', 'volume': 0.15},
            'sadness': {'freq': [150, 200], 'type': 'gentle', 'volume': 0.1},
            'anger': {'freq': [300, 500], 'type': 'harsh', 'volume': 0.25},
            'joy': {'freq': [400, 600], 'type': 'bright', 'volume': 0.15},
            'excitement': {'freq': [500, 800], 'type': 'energetic', 'volume': 0.2},
            'peaceful': {'freq': [200, 300], 'type': 'ambient', 'volume': 0.1},
            'uplifting': {'freq': [350, 550], 'type': 'warm', 'volume': 0.12},
            'calm': {'freq': [150, 250], 'type': 'gentle', 'volume': 0.08},
            'neutral': {'freq': [250, 350], 'type': 'ambient', 'volume': 0.05},
            'ambient': {'freq': [100, 200], 'type': 'ambient', 'volume': 0.05}
        }
    
    def generate_background_sound(self, emotion: str, duration: float, output_path: str) -> str:
        """Generate a background sound effect for given emotion and duration"""
        
        template = self.sound_templates.get(emotion, self.sound_templates['neutral'])
        
        # Generate simple background tone/noise
        sample_rate = self.config.sample_rate
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        if template['type'] == 'noise':
            # Pink noise for horror/tension
            audio = self._generate_pink_noise(len(t)) * template['volume']
        elif template['type'] == 'drone':
            # Low frequency drone
            freq = template['freq'][0]
            audio = np.sin(2 * np.pi * freq * t) * template['volume']
        elif template['type'] == 'ambient':
            # Gentle ambient sound
            freq1, freq2 = template['freq']
            audio = (np.sin(2 * np.pi * freq1 * t) + 
                    0.5 * np.sin(2 * np.pi * freq2 * t)) * template['volume']
        else:
            # Default gentle tone
            freq = np.mean(template['freq'])
            audio = np.sin(2 * np.pi * freq * t) * template['volume']
        
        # Apply fade in/out
        fade_samples = int(0.5 * sample_rate)  # 0.5 second fade
        if len(audio) > 2 * fade_samples:
            audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
            audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        # Save background sound
        sf.write(output_path, audio, sample_rate)
        return output_path
    
    def _generate_pink_noise(self, length: int) -> np.ndarray:
        """Generate pink noise for atmospheric effects"""
        # Simple pink noise approximation
        white_noise = np.random.normal(0, 1, length)
        
        # Apply simple filter to approximate pink noise
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        
        from scipy import signal
        pink_noise = signal.lfilter(b, a, white_noise)
        
        return pink_noise / np.max(np.abs(pink_noise))
    
    def mix_audio_with_background(self, speech_path: str, emotion: str, output_path: str) -> str:
        """Mix speech audio with appropriate background sound"""
        
        try:
            # Load speech audio
            speech = AudioSegment.from_wav(speech_path)
            speech_duration = len(speech) / 1000.0  # Convert to seconds
            
            # Generate background sound
            bg_temp_path = output_path.replace('.wav', '_bg_temp.wav')
            self.generate_background_sound(emotion, speech_duration, bg_temp_path)
            
            # Load background sound
            background = AudioSegment.from_wav(bg_temp_path)
            
            # Adjust volumes
            background = background - (20 - int(self.config.background_volume * 20))  # Reduce volume
            
            # Mix audio
            mixed = speech.overlay(background)
            
            # Normalize if requested
            if self.config.normalize_audio:
                mixed = normalize(mixed)
            
            # Export mixed audio
            mixed.export(output_path, format="wav")
            
            # Clean up temp file
            os.remove(bg_temp_path)
            
            return output_path
            
        except Exception as e:
            print(f"⚠️ Error mixing audio: {e}")
            # Return original speech if mixing fails
            return speech_path

# ================================================================
# 💾 6. EXPORT FINAL AUDIO
# ================================================================

class AudiobookExporter:
    """Handles combining all audio segments and exporting final audiobook"""
    
    def __init__(self, config: AudiobookConfig):
        self.config = config
    
    def combine_audio_segments(self, audio_files: List[str], output_path: str) -> str:
        """Combine all audio segments into final audiobook"""
        
        if not audio_files:
            raise ValueError("No audio files to combine")
        
        print(f"🎵 Combining {len(audio_files)} audio segments...")
        
        try:
            # Load first segment
            combined = AudioSegment.from_wav(audio_files[0])
            
            # Add crossfades between segments
            for i, audio_file in enumerate(tqdm(audio_files[1:], desc="Combining audio"), 1):
                
                if not os.path.exists(audio_file):
                    print(f"⚠️ Skipping missing file: {audio_file}")
                    continue
                
                try:
                    segment = AudioSegment.from_wav(audio_file)
                    
                    # Add crossfade
                    if self.config.crossfade_duration > 0:
                        combined = combined.append(segment, crossfade=self.config.crossfade_duration)
                    else:
                        combined = combined + segment
                        
                except Exception as e:
                    print(f"⚠️ Error loading segment {audio_file}: {e}")
                    continue
            
            # Final normalization
            if self.config.normalize_audio:
                combined = normalize(combined)
            
            # Export final audiobook
            print(f"💾 Exporting final audiobook to: {output_path}")
            
            if output_path.endswith('.mp3'):
                combined.export(output_path, format="mp3", bitrate="192k")
            else:
                combined.export(output_path, format="wav")
            
            # Get final stats
            duration_minutes = len(combined) / (1000 * 60)
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            
            print(f"✅ Audiobook created successfully!")
            print(f"   Duration: {duration_minutes:.1f} minutes")
            print(f"   File size: {file_size_mb:.1f} MB")
            print(f"   Location: {output_path}")
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Error combining audio segments: {str(e)}")
    
    def create_metadata_file(self, segments: List[TextSegment], output_dir: str) -> str:
        """Create metadata file with segment information"""
        
        metadata = {
            'total_segments': len(segments),
            'total_duration_estimate': len(segments) * 5,  # Rough estimate
            'emotions_used': list(set(seg.emotion for seg in segments)),
            'segments': []
        }
        
        for i, segment in enumerate(segments):
            seg_data = {
                'index': i,
                'text_preview': segment.text[:100] + "..." if len(segment.text) > 100 else segment.text,
                'emotion': segment.emotion,
                'sentiment_score': segment.confidence,
                'audio_file': os.path.basename(segment.audio_path) if segment.audio_path else None
            }
            metadata['segments'].append(seg_data)
        
        # Save metadata
        metadata_path = os.path.join(output_dir, 'audiobook_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata_path

# ================================================================
# 🎬 MAIN AUDIOBOOK GENERATOR CLASS
# ================================================================

class EmotionalAudiobookGenerator:
    """Main class that orchestrates the entire audiobook generation process"""
    
    def __init__(self, config: AudiobookConfig = None):
        self.config = config or AudiobookConfig()
        
        # Initialize components
        self.emotion_analyzer = EmotionAnalyzer(self.config)
        self.narrator = BarkNarrator(self.config)
        self.sound_mixer = SoundEffectMixer(self.config)
        self.exporter = AudiobookExporter(self.config)
        
        # Progress tracking
        self.current_step = 0
        self.total_steps = 6
    
    def generate_audiobook(self, 
                         input_file: str, 
                         output_dir: str,
                         output_filename: str = "emotional_audiobook.wav") -> str:
        """Generate complete emotional audiobook from input file"""
        
        print("🎧 Starting Emotional Audiobook Generation")
        print("=" * 50)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Step 1: Load document
            self._update_progress("Loading document")
            text = load_document(input_file)
            print(f"📖 Loaded document with {len(text)} characters")
            
            # Step 2: Analyze sentiment
            self._update_progress("Analyzing text sentiment")
            segments = self.emotion_analyzer.analyze_text_segments(text)
            print(f"🧠 Created {len(segments)} emotional segments")
            
            # Step 3: Load TTS models
            self._update_progress("Loading text-to-speech models")
            self.narrator.load_models()
            
            # Step 4: Generate narration
            self._update_progress("Generating emotional narration")
            narration_dir = os.path.join(output_dir, "narration_segments")
            audio_files = self.narrator.generate_audiobook_narration(segments, narration_dir)
            
            # Step 5: Add sound effects
            self._update_progress("Adding background sound effects")
            mixed_audio_files = []
            mixed_dir = os.path.join(output_dir, "mixed_segments")
            os.makedirs(mixed_dir, exist_ok=True)
            
            for i, (segment, audio_file) in enumerate(zip(segments, audio_files)):
                if segment.audio_path and os.path.exists(segment.audio_path):
                    mixed_path = os.path.join(mixed_dir, f"mixed_{i:04d}.wav")
                    mixed_file = self.sound_mixer.mix_audio_with_background(
                        audio_file, segment.emotion, mixed_path
                    )
                    mixed_audio_files.append(mixed_file)
            
            # Step 6: Export final audiobook
            self._update_progress("Exporting final audiobook")
            output_path = os.path.join(output_dir, output_filename)
            final_audiobook = self.exporter.combine_audio_segments(mixed_audio_files, output_path)
            
            # Create metadata
            metadata_path = self.exporter.create_metadata_file(segments, output_dir)
            
            print("\n🎉 AUDIOBOOK GENERATION COMPLETE!")
            print(f"📁 Output directory: {output_dir}")
            print(f"🎵 Final audiobook: {final_audiobook}")
            print(f"📊 Metadata: {metadata_path}")
            
            return final_audiobook
            
        except Exception as e:
            print(f"❌ Error during audiobook generation: {str(e)}")
            raise
    
    def _update_progress(self, step_description: str):
        """Update progress display"""
        self.current_step += 1
        print(f"\n[{self.current_step}/{self.total_steps}] {step_description}...")

# ================================================================
# 🚀 EXAMPLE USAGE FOR GOOGLE COLAB
# ================================================================

def demo_audiobook_generation():
    """Demo function showing how to use the audiobook generator"""
    
    print("🎧 EMOTIONAL AUDIOBOOK GENERATOR DEMO")
    print("=" * 40)
    
    # Create configuration
    config = AudiobookConfig(
        max_chunk_length=150,  # Shorter chunks for demo
        background_volume=0.25,
        normalize_audio=True,
        use_gpu=torch.cuda.is_available()
    )
    
    # Initialize generator
    generator = EmotionalAudiobookGenerator(config)
    
    # Example with sample text (replace with actual file upload)
    sample_text = """
    It was a dark and stormy night. The wind howled through the trees, 
    creating an atmosphere of pure terror. Sarah felt her heart pounding 
    as she approached the old mansion.
    
    But then, the sun broke through the clouds, and everything changed. 
    The birds began to sing, and Sarah felt a wave of joy wash over her. 
    The mansion no longer looked scary, but welcoming and warm.
    
    She took a deep breath and smiled. Today was going to be a wonderful day.
    """
    
    # Save sample text to file
    sample_file = "/content/sample_book.txt"
    with open(sample_file, 'w') as f:
        f.write(sample_text)
    
    # Convert to PDF for demonstration
    print("📝 Creating sample PDF...")
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    
    pdf_file = "/content/sample_book.pdf"
    c = canvas.Canvas(pdf_file, pagesize=letter)
    width, height = letter
    
    # Add text to PDF
    lines = sample_text.split('\n')
    y = height - 50
    for line in lines:
        if line.strip():
            c.drawString(50, y, line.strip())
            y -= 20
    c.save()
    
    # Generate audiobook
    output_dir = "/content/audiobook_output"
    audiobook_path = generator.generate_audiobook(
        input_file=pdf_file,
        output_dir=output_dir,
        output_filename="demo_emotional_audiobook.wav"
    )
    
    # Display audio player
    print("\n🎵 Playing generated audiobook:")
    display(Audio(audiobook_path))
    
    return audiobook_path

# ================================================================
# 📱 INTERACTIVE COLAB INTERFACE
# ================================================================

def create_interactive_interface():
    """Create an interactive interface for Google Colab"""
    
    from google.colab import files
    from IPython.display import display, HTML
    import ipywidgets as widgets
    
    # File upload widget
    print("📤 Upload your PDF or ePub file:")
    uploaded = files.upload()
    
    if not uploaded:
        print("❌ No file uploaded!")
        return
    
    # Get uploaded file
    input_file = list(uploaded.keys())[0]
    print(f"✅ Uploaded: {input_file}")
    
    # Configuration widgets
    print("\n⚙️ Configure your audiobook:")
    
    # Voice selection
    voice_options = [
        ('Neutral - Speaker 6', 'v2/en_speaker_6'),
        ('Warm - Speaker 1', 'v2/en_speaker_1'),
        ('Energetic - Speaker 7', 'v2/en_speaker_7'),
        ('Calm - Speaker 0', 'v2/en_speaker_0'),
        ('Dramatic - Speaker 8', 'v2/en_speaker_8')
    ]
    
    voice_widget = widgets.Dropdown(
        options=voice_options,
        value='v2/en_speaker_6',
        description='Voice:',
    )
    
    # Background volume
    volume_widget = widgets.FloatSlider(
        value=0.3,
        min=0.0,
        max=0.8,
        step=0.1,
        description='Background Volume:',
    )
    
    # Chunk length
    chunk_widget = widgets.IntSlider(
        value=200,
        min=50,
        max=500,
        step=50,
        description='Chunk Length:',
    )
    
    # Display widgets
    display(voice_widget, volume_widget, chunk_widget)
    
    # Generate button
    generate_button = widgets.Button(
        description='🎧 Generate Audiobook',
        disabled=False,
        button_style='success',
        layout=widgets.Layout(width='200px', height='50px')
    )
    
    output = widgets.Output()
    
    def on_generate_click(b):
        with output:
            clear_output(wait=True)
            
            # Create config from widgets
            config = AudiobookConfig(
                bark_voice_preset=voice_widget.value,
                background_volume=volume_widget.value,
                max_chunk_length=chunk_widget.value,
                use_gpu=torch.cuda.is_available()
            )
            
            # Generate audiobook
            generator = EmotionalAudiobookGenerator(config)
            
            try:
                audiobook_path = generator.generate_audiobook(
                    input_file=input_file,
                    output_dir="/content/audiobook_output",
                    output_filename="my_emotional_audiobook.wav"
                )
                
                print("🎉 Generation complete!")
                print("🎵 Your audiobook:")
                display(Audio(audiobook_path))
                
                # Download link
                print("\n💾 Download your audiobook:")
                files.download(audiobook_path)
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    generate_button.on_click(on_generate_click)
    display(generate_button, output)

# ================================================================
# 🎯 MAIN EXECUTION
# ================================================================

if __name__ == "__main__":
    print("🎧 Emotional Audiobook Generator Loaded!")
    print("\nTo get started, run one of these:")
    print("1. demo_audiobook_generation() - Quick demo with sample text")
    print("2. create_interactive_interface() - Upload your own book")
    print("\nOr create your own generator:")
    print("generator = EmotionalAudiobookGenerator()")
    print("audiobook = generator.generate_audiobook('your_book.pdf', 'output_dir')")

# ================================================================
# 📋 REQUIREMENTS FOR COLAB
# ================================================================

"""
Required packages (already installed in the setup section):
- torch, torchaudio, torchvision
- bark (Suno's TTS)
- transformers, datasets, accelerate  
- pydub, librosa, soundfile
- PyPDF2, ebooklib, beautifulsoup4
- nltk, textstat
- IPython, tqdm
- reportlab (for demo)
- ipywidgets (for interactive interface)

GPU Requirements:
- Recommended: GPU with 8GB+ VRAM
- Will work on CPU but much slower
- Google Colab Pro recommended for best performance

Usage in Colab:
1. Run all cells to install dependencies and load functions
2. Use demo_audiobook_generation() for quick test
3. Use create_interactive_interface() for full experience
4. Or create custom AudiobookConfig and use EmotionalAudiobookGenerator directly
""" 