# 🎧 Google Colab Setup Guide

## 📥 Quick Setup in Google Colab

### Step 1: Installation Cell
Copy and run this in a Colab cell:

```python
# Install core ML packages with GPU support
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Bark TTS
!pip install -q git+https://github.com/suno-ai/bark.git

# Install transformers and related packages
!pip install -q transformers datasets accelerate

# Install audio processing libraries
!pip install -q pydub librosa soundfile scipy

# Install text and file processing libraries
!pip install -q PyPDF2 ebooklib beautifulsoup4 nltk textstat

# Install utilities and UI components
!pip install -q IPython tqdm reportlab ipywidgets

# Install system dependencies
!apt-get update -qq
!apt-get install -y -qq ffmpeg

print("✅ Installation complete!")
```

### Step 2: Upload Python Files
Upload these files to your Colab environment:
- `audiobook_classes.py`
- `bark_narrator.py` 
- `sound_effects.py`
- `main_generator.py`

### Step 3: Import and Run
```python
# Import the main generator
from main_generator import demo_audiobook_generation, create_interactive_interface
from audiobook_classes import AudiobookConfig
from main_generator import EmotionalAudiobookGenerator

# Option 1: Quick demo
demo_audiobook_generation()

# Option 2: Interactive interface
create_interactive_interface()

# Option 3: Custom usage
config = AudiobookConfig(max_chunk_length=150, background_volume=0.3)
generator = EmotionalAudiobookGenerator(config)
audiobook = generator.generate_audiobook("your_book.pdf", "./output")
```

## 🎛️ Configuration Options

### Basic Configuration
```python
config = AudiobookConfig(
    max_chunk_length=200,        # Characters per audio segment
    background_volume=0.3,       # Background sound volume (0.0-1.0)
    bark_voice_preset="v2/en_speaker_6",  # Voice style
    emotion_threshold=0.6,       # Emotion detection sensitivity
    crossfade_duration=500,      # Crossfade between segments (ms)
    use_gpu=True                 # Use GPU if available
)
```

### Voice Options
| Voice | Description | Best For |
|-------|-------------|----------|
| `v2/en_speaker_0` | Very calm, peaceful | Meditation, relaxation |
| `v2/en_speaker_1` | Warm, friendly | General narration |
| `v2/en_speaker_2` | Gentle, soft | Emotional content |
| `v2/en_speaker_6` | Neutral, clear | Default choice |
| `v2/en_speaker_7` | Energetic, dynamic | Adventure, action |
| `v2/en_speaker_8` | Dramatic, intense | Thrillers, horror |
| `v2/en_speaker_9` | Happy, upbeat | Comedy, light content |

## 📱 Interactive Interface Features

The `create_interactive_interface()` function provides:
- 📤 File upload widget for PDF/ePub
- 🎤 Voice selection dropdown
- 🔊 Background volume slider
- 📝 Text chunk length adjustment
- 🎧 One-click generation
- 💾 Automatic download

## 🔧 Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| "CUDA out of memory" | Reduce `max_chunk_length` to 100-150 |
| "Models not loading" | Restart runtime and re-run cells |
| "ImportError: bark" | Re-run installation cell |
| "Audio export failed" | Ensure ffmpeg is installed |
| Slow processing | Check GPU is enabled in Runtime settings |

### Memory Optimization
```python
# For limited memory environments
config = AudiobookConfig(
    max_chunk_length=100,    # Smaller chunks
    background_volume=0.1,   # Minimal background
    use_gpu=False           # Use CPU if GPU memory is low
)
```

### GPU Settings
1. Go to Runtime → Change runtime type
2. Select "GPU" as Hardware accelerator
3. Choose "High-RAM" if available
4. Save and restart runtime

## 📊 Performance Expectations

| Hardware | Processing Speed | Recommended Settings |
|----------|------------------|---------------------|
| Colab Free (GPU) | ~30 sec/minute | chunk_length=150 |
| Colab Pro (GPU) | ~15 sec/minute | chunk_length=200 |
| CPU Only | ~300 sec/minute | chunk_length=100 |

## 🎵 Audio Output Formats

### WAV (Recommended)
- High quality, no compression
- Larger file size
- Best for archival

### MP3 (Compressed)
- Smaller file size
- Good quality at 192kbps
- Better for sharing

## 📋 Example Workflow

1. **Upload your book** (PDF or ePub)
2. **Choose voice style** based on content type
3. **Adjust settings** for your preferences
4. **Generate audiobook** (15-30 minutes for typical book)
5. **Download and enjoy** your emotional audiobook!

## 🎯 Tips for Best Results

### Text Preparation
- Use clean, well-formatted PDFs
- Remove headers/footers if possible
- Ensure proper sentence punctuation

### Voice Selection
- **Fiction**: Use dramatic voices (Speaker 8)
- **Non-fiction**: Use neutral voices (Speaker 6)
- **Children's books**: Use upbeat voices (Speaker 9)
- **Meditation**: Use calm voices (Speaker 0)

### Background Sounds
- **Horror/Thriller**: Set volume to 0.4
- **Romance/Drama**: Set volume to 0.2
- **Technical/Educational**: Set volume to 0.1
- **No effects**: Set volume to 0.0

## 🚀 Advanced Usage

### Custom Emotion Mapping
```python
# Modify emotion analyzer for custom mappings
analyzer = EmotionAnalyzer(config)
analyzer.emotion_mapping['POSITIVE'] = ['joy', 'excitement', 'peaceful']
```

### Batch Processing
```python
books = ['book1.pdf', 'book2.pdf', 'book3.pdf']
for book in books:
    output_name = f"audiobook_{Path(book).stem}.wav"
    generator.generate_audiobook(book, "./batch_output", output_name)
```

### Custom Sound Effects
```python
# Add your own background sounds
mixer = SoundEffectMixer(config)
mixer.sound_templates['custom'] = {
    'freq': [300, 400], 
    'type': 'ambient', 
    'volume': 0.15
}
```

---

## 🎉 Ready to Create!

Your emotional audiobook generator is ready to transform any text into an immersive audio experience. Start with the demo, then try your own books!

**Happy listening! 🎧** 