#!/usr/bin/env python3
"""
Simple PDF Text Extraction Demo
Shows basic text extraction from PDF without heavy dependencies
"""

import os
import re
from tqdm import tqdm

def extract_pdf_text(file_path):
    """Extract text from PDF file"""
    try:
        import PyPDF2
        
        print(f"📖 Opening PDF: {file_path}")
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            print(f"📄 Found {total_pages} pages")
            
            text = ""
            for page_num in range(total_pages):
                print(f"   Reading page {page_num + 1}...")
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += page_text + "\n"
            
            return text.strip()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def clean_text(text):
    """Basic text cleaning"""
    # Remove excessive whitespace
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def analyze_text_emotion_simple(text):
    """Simple emotion analysis based on keywords"""
    emotions = {
        'joy': ['happy', 'joy', 'smile', 'laugh', 'merry', 'glad', 'delight'],
        'sadness': ['sad', 'cry', 'tears', 'sorrow', 'grief', 'poor'],
        'love': ['love', 'dear', 'heart', 'affection', 'precious'],
        'surprise': ['surprise', 'amazed', 'wonder', 'astonished'],
        'peaceful': ['calm', 'quiet', 'gentle', 'soft', 'peace']
    }
    
    text_lower = text.lower()
    emotion_scores = {}
    
    for emotion, keywords in emotions.items():
        count = sum(1 for keyword in keywords if keyword in text_lower)
        if count > 0:
            emotion_scores[emotion] = count
    
    return emotion_scores

def main():
    print("🎧 SIMPLE AUDIOBOOK TEXT EXTRACTION DEMO")
    print("=" * 45)
    
    pdf_file = "1-the_gift_of_the_magi_0.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"❌ File '{pdf_file}' not found!")
        return
    
    # Extract text
    print(f"\n📚 Extracting text from '{pdf_file}'...")
    raw_text = extract_pdf_text(pdf_file)
    
    if not raw_text:
        print("❌ Failed to extract text")
        return
    
    # Clean text
    clean_text_content = clean_text(raw_text)
    
    print(f"✅ Extracted {len(clean_text_content)} characters")
    
    # Show preview
    print(f"\n📖 TEXT PREVIEW:")
    print("-" * 50)
    preview = clean_text_content[:800] + "..." if len(clean_text_content) > 800 else clean_text_content
    print(preview)
    print("-" * 50)
    
    # Simple emotion analysis
    print(f"\n🎭 EMOTION ANALYSIS:")
    emotions = analyze_text_emotion_simple(clean_text_content)
    
    if emotions:
        print("Found emotional keywords:")
        for emotion, count in emotions.items():
            print(f"   {emotion:10s}: {count} occurrences")
    else:
        print("   No specific emotional keywords detected (neutral tone)")
    
    # Simulate audiobook generation
    print(f"\n🎙️ AUDIOBOOK GENERATION SIMULATION:")
    print("This text would be processed as follows:")
    
    # Split into rough chunks
    words = clean_text_content.split()
    chunk_size = 50  # words per chunk
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    print(f"1. 📝 Split into {len(chunks)} audio segments")
    print(f"2. 🧠 Analyze emotion for each segment")
    print(f"3. 🎤 Generate speech with appropriate voice")
    print(f"4. 🎵 Add background sounds based on emotions")
    print(f"5. 🎧 Mix and export final audiobook")
    
    print(f"\n🎉 TEXT PROCESSING COMPLETE!")
    print(f"Your '{pdf_file}' is ready for audiobook conversion!")
    
    # Show what the segments would look like
    print(f"\n📋 SAMPLE SEGMENTS:")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"Segment {i+1}: {chunk[:100]}...")

if __name__ == "__main__":
    main() 