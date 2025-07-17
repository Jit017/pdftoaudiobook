# ================================================================
# 🎛️ CONFIGURATION & DATA CLASSES
# ================================================================

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import torch
from bark import SAMPLE_RATE

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
    use_gpu: bool = torch.cuda.is_available()
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

import re
import os
from pathlib import Path
from tqdm.auto import tqdm
import PyPDF2
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

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

import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline

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