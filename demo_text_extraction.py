#!/usr/bin/env python3
"""
Demo: Text Extraction and Processing from PDF
This demonstrates the core functionality without heavy ML dependencies
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    print("✅ NLTK data downloaded")
except:
    print("⚠️ NLTK download skipped")

@dataclass
class TextSegment:
    """Represents a segment of text with metadata"""
    text: str
    emotion: str  # Simulated for demo
    confidence: float
    start_index: int
    end_index: int

class DocumentLoader:
    """Handles loading and parsing of PDF files"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        import PyPDF2
        
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

class SimpleEmotionAnalyzer:
    """Simulated emotion analysis for demo purposes"""
    
    def __init__(self, max_chunk_length=200):
        self.max_chunk_length = max_chunk_length
        
        # Simple keyword-based emotion detection for demo
        self.emotion_keywords = {
            'joy': ['happy', 'joy', 'smile', 'laugh', 'delight', 'cheerful', 'merry', 'glad'],
            'sadness': ['sad', 'cry', 'tears', 'sorrow', 'grief', 'melancholy', 'wept'],
            'fear': ['afraid', 'fear', 'terror', 'scared', 'frightened', 'dread'],
            'anger': ['angry', 'rage', 'fury', 'mad', 'irritated', 'annoyed'],
            'surprise': ['surprise', 'amazed', 'astonished', 'shocked', 'stunned'],
            'love': ['love', 'affection', 'adore', 'cherish', 'devotion', 'tender'],
            'peaceful': ['calm', 'peaceful', 'serene', 'tranquil', 'quiet', 'gentle']
        }
    
    def analyze_text_segments(self, text: str) -> List[TextSegment]:
        """Split text into segments and analyze each for emotion"""
        
        # Split into sentences
        sentences = sent_tokenize(text)
        segments = []
        
        print(f"🔍 Analyzing text with {len(sentences)} sentences...")
        
        # Group sentences into chunks
        current_chunk = ""
        current_start = 0
        
        for i, sentence in enumerate(tqdm(sentences, desc="Processing sentences")):
            # Check if adding this sentence would exceed chunk length
            if len(current_chunk + sentence) > self.max_chunk_length and current_chunk:
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
        """Create a text segment with simulated emotion analysis"""
        
        # Simple keyword-based emotion detection
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        # Determine dominant emotion
        if emotion_scores:
            emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = min(emotion_scores[emotion] / len(text.split()) * 10, 1.0)
        else:
            emotion = 'neutral'
            confidence = 0.5
        
        return TextSegment(
            text=text,
            emotion=emotion,
            confidence=confidence,
            start_index=start_idx,
            end_index=end_idx
        )

def demonstrate_audiobook_pipeline(pdf_path: str):
    """Demonstrate the core audiobook generation pipeline"""
    
    print("🎧 EMOTIONAL AUDIOBOOK GENERATOR - TEXT PROCESSING DEMO")
    print("=" * 60)
    
    # Step 1: Extract text from PDF
    print(f"\n📚 Step 1: Loading PDF '{pdf_path}'...")
    try:
        text = DocumentLoader.extract_text_from_pdf(pdf_path)
        print(f"✅ Extracted {len(text)} characters from PDF")
        
        # Show a preview of the text
        preview = text[:500] + "..." if len(text) > 500 else text
        print(f"\n📖 Text Preview:")
        print("-" * 40)
        print(preview)
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return
    
    # Step 2: Analyze text segments
    print(f"\n🧠 Step 2: Analyzing text for emotions...")
    analyzer = SimpleEmotionAnalyzer(max_chunk_length=150)
    segments = analyzer.analyze_text_segments(text)
    
    print(f"✅ Created {len(segments)} text segments")
    
    # Step 3: Show emotion analysis results
    print(f"\n🎭 Step 3: Emotion Analysis Results:")
    print("-" * 60)
    
    emotion_counts = {}
    for i, segment in enumerate(segments[:10]):  # Show first 10 segments
        emotion = segment.emotion
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print(f"Segment {i+1:2d}: {emotion:10s} ({segment.confidence:.2f}) - {segment.text[:80]}...")
    
    if len(segments) > 10:
        print(f"... and {len(segments) - 10} more segments")
    
    # Step 4: Show emotion summary
    print(f"\n📊 Step 4: Emotion Summary:")
    print("-" * 30)
    for emotion, count in sorted(emotion_counts.items()):
        percentage = (count / len(segments)) * 100
        print(f"{emotion:12s}: {count:3d} segments ({percentage:5.1f}%)")
    
    # Step 5: Simulate what would happen next
    print(f"\n🎙️ Step 5: What happens next in full pipeline:")
    print("-" * 50)
    print("For each segment, the system would:")
    print("1. 🎤 Generate emotional speech using Bark TTS")
    print("2. 🎵 Add adaptive background sounds based on emotion")
    print("3. 🎧 Mix audio with professional crossfades")
    print("4. 💾 Export as high-quality audiobook")
    
    print(f"\n🎉 Text processing complete!")
    print(f"📈 Ready for audio generation with {len(segments)} emotional segments")
    
    return segments

if __name__ == "__main__":
    # Test with the user's PDF
    pdf_file = "1-the_gift_of_the_magi_0.pdf"
    
    if os.path.exists(pdf_file):
        segments = demonstrate_audiobook_pipeline(pdf_file)
    else:
        print(f"❌ PDF file '{pdf_file}' not found!")
        print("📁 Available files:")
        for file in os.listdir("."):
            if file.endswith(".pdf"):
                print(f"   - {file}") 