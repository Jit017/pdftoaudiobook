# 🎧 Emotional Audiobook Generator

Convert PDF/ePub books into immersive audiobooks with emotional narration and adaptive sound effects!

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![GPU](https://img.shields.io/badge/GPU-recommended-orange.svg)

## ✨ Features

- 🎙️ **Emotional Narration**: Uses Bark TTS to generate expressive, emotional speech
- 🧠 **Sentiment Analysis**: Automatically detects text emotions using HuggingFace transformers
- 🎵 **Adaptive Sound Effects**: Adds background sounds based on text sentiment
- 📚 **Multiple Formats**: Supports PDF and ePub input files
- 🎧 **Professional Audio**: High-quality audio mixing and export
- 🌐 **Google Colab Ready**: Optimized for cloud execution
- 🖥️ **Streamlit UI**: User-friendly web interface

## 🚀 Quick Start

### Google Colab (Recommended)

1. **Open Google Colab** and create a new notebook
2. **Copy the installation commands**:
```python
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q git+https://github.com/suno-ai/bark.git
!pip install -q transformers datasets accelerate
!pip install -q pydub librosa soundfile
!pip install -q PyPDF2 ebooklib beautifulsoup4
!pip install -q nltk textstat scipy
!pip install -q IPython tqdm reportlab ipywidgets

!apt-get update -qq
!apt-get install -y -qq ffmpeg
```

3. **Upload the Python files** to your Colab environment:
   - `audiobook_classes.py`
   - `bark_narrator.py`
   - `sound_effects.py`
   - `main_generator.py`

4. **Run the demo**:
```python
from main_generator import demo_audiobook_generation
demo_audiobook_generation()
```

5. **Or use the interactive interface**:
```python
from main_generator import create_interactive_interface
create_interactive_interface()
```

### Local Installation

```bash
# Clone or download the repository
git clone <repository-url>
cd emotional-audiobook-generator

# Install dependencies
pip install -r requirements.txt

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install ffmpeg

# Run the Streamlit UI
streamlit run streamlit_ui.py
```

## 📁 Project Structure

```
emotional-audiobook-generator/
├── audiobook_classes.py      # Core data classes and document loader
├── bark_narrator.py          # Bark TTS narrator with emotional voices
├── sound_effects.py          # Sound effect mixer and audio exporter
├── main_generator.py         # Main orchestrator and demo functions
├── streamlit_ui.py          # Web interface using Streamlit
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🎛️ Configuration Options

### AudiobookConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_chunk_length` | 200 | Maximum characters per audio segment |
| `bark_voice_preset` | "v2/en_speaker_6" | Default voice for narration |
| `background_volume` | 0.3 | Volume of background effects (0.0-1.0) |
| `emotion_threshold` | 0.6 | Confidence threshold for emotion detection |
| `crossfade_duration` | 500 | Crossfade between segments (ms) |
| `use_gpu` | True | Enable GPU acceleration if available |

### Voice Presets

| Emotion | Voice Preset | Description |
|---------|--------------|-------------|
| Neutral | v2/en_speaker_6 | Balanced, clear narration |
| Joy | v2/en_speaker_9 | Happy, upbeat delivery |
| Calm | v2/en_speaker_0 | Peaceful, soothing voice |
| Dramatic | v2/en_speaker_8 | Intense, expressive narration |
| Sadness | v2/en_speaker_2 | Gentle, melancholic tone |

## 🎵 Sound Effects

The system automatically generates background sounds based on detected emotions:

- **Horror/Tension**: Pink noise and low-frequency drones
- **Joy/Excitement**: Bright, energetic tones
- **Sadness**: Gentle, low-frequency ambient sounds
- **Calm/Peaceful**: Soft ambient textures
- **Neutral**: Minimal background ambience

## 💻 Usage Examples

### Basic Usage

```python
from audiobook_classes import AudiobookConfig
from main_generator import EmotionalAudiobookGenerator

# Create configuration
config = AudiobookConfig(
    max_chunk_length=200,
    background_volume=0.3,
    use_gpu=True
)

# Initialize generator
generator = EmotionalAudiobookGenerator(config)

# Generate audiobook
audiobook_path = generator.generate_audiobook(
    input_file="my_book.pdf",
    output_dir="./output",
    output_filename="my_audiobook.wav"
)
```

### Custom Voice Configuration

```python
config = AudiobookConfig(
    bark_voice_preset="v2/en_speaker_1",  # Warm voice
    background_volume=0.2,                # Quieter background
    emotion_threshold=0.7,                # Higher emotion sensitivity
    crossfade_duration=1000              # Longer crossfades
)
```

### Streamlit Interface

```bash
streamlit run streamlit_ui.py
```

Then upload your PDF/ePub file and configure settings through the web interface.

## 🔧 System Requirements

### Recommended (Google Colab Pro)
- GPU with 8GB+ VRAM
- 16GB+ RAM
- Fast internet connection

### Minimum
- CPU with 8GB+ RAM
- 10GB free disk space
- Python 3.8+

### Dependencies

- **Core ML**: `torch`, `transformers`, `bark`
- **Audio**: `pydub`, `librosa`, `soundfile`
- **Text Processing**: `nltk`, `textstat`
- **File Handling**: `PyPDF2`, `ebooklib`
- **UI**: `streamlit`, `ipywidgets`

## 📊 Performance Metrics

| Hardware | Processing Speed | GPU Memory |
|----------|------------------|------------|
| Google Colab (Free) | ~30 sec/minute | 8GB |
| Google Colab Pro | ~15 sec/minute | 16GB |
| RTX 3090 | ~10 sec/minute | 24GB |
| CPU Only | ~300 sec/minute | N/A |

*Processing time varies based on text complexity and chunk length*

## 🐛 Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce `max_chunk_length` to 100-150
- Use CPU-only mode: `use_gpu=False`
- Restart Colab runtime

**"Models not loading"**
- Check internet connection
- Restart and re-run installation cells
- Try fallback models

**"Audio export failed"**
- Install ffmpeg: `!apt-get install -y ffmpeg`
- Check output directory permissions
- Ensure sufficient disk space

### Error Messages

| Error | Solution |
|-------|----------|
| `ImportError: No module named 'bark'` | Run installation cells again |
| `RuntimeError: CUDA error` | Reduce batch size or use CPU |
| `FileNotFoundError: PDF/ePub` | Check file path and format |
| `ValueError: Invalid audio format` | Use WAV or MP3 output formats |

## 🔮 Future Enhancements

- [ ] **Multiple Speakers**: Different voices for dialogue
- [ ] **Music Integration**: Add background music tracks
- [ ] **Voice Cloning**: Custom voice training
- [ ] **Real-time Preview**: Live audio generation
- [ ] **Batch Processing**: Multiple books at once
- [ ] **Cloud Deployment**: Web service API

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Suno AI** for the amazing Bark TTS model
- **Hugging Face** for transformer models and tools
- **Google Colab** for providing accessible GPU resources
- **Streamlit** for the excellent web framework

## 📞 Support

- 📧 Email: [your-email@example.com]
- 🐛 Issues: [GitHub Issues](link-to-issues)
- 💬 Discussions: [GitHub Discussions](link-to-discussions)

---

<div align="center">

**Made with ❤️ for the audiobook community**

[⭐ Star this repo](link-to-repo) | [🐛 Report Bug](link-to-issues) | [💡 Request Feature](link-to-issues)

</div> 