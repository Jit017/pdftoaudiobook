# 🎧 Emotional Audiobook Generator Suite

Convert PDF/ePub books into immersive audiobooks with emotional narration, adaptive background music, and contextual sound effects!

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

## 🚀 Multiple Versions Available

Choose the version that best fits your needs:

### 🌟 Enhanced Audiobook Generator (`enhanced_audiobook_ui.py`)
**✨ Full-Featured Production Version**
- 🎭 **Emotional Voice Synthesis**: 6 different accents/speeds based on detected emotions
- 🎵 **Adaptive Background Music**: Rain for sadness, harmonies for joy, tension for fear
- 🔊 **Contextual Sound Effects**: Door creaks, footsteps, wind, fire, water, thunder
- 🧠 **Pattern-Based Emotion Detection**: Joy, sadness, fear, love, anger, surprise
- 🎨 **Professional UI**: Beautiful Streamlit interface with detailed analytics

### 🎧 Working Audiobook Generator (`working_audiobook_ui.py`)
**🔧 Production-Ready with Audio Combining**
- ✅ **Combined Audiobook Export**: Single MP3 file using FFmpeg
- 📂 **Individual Segments**: Download each emotional segment separately  
- 🎤 **Multiple Voice Styles**: British, Australian, Canadian, US accents
- 📊 **Emotion Analysis**: Real-time emotion breakdown and statistics

### 🎯 Simple Audiobook Generator (`simple_streamlit_ui.py`)
**⚡ Lightweight Version**
- 🎤 **Basic Emotional Voices**: Different accents for different emotions
- 📱 **Mobile-Friendly**: Optimized for quick generation
- 💾 **Individual Downloads**: Perfect for testing and demos

### 🔧 Ultra-Simple Generator (`ultra_simple_ui.py`)
**🛠️ Maximum Compatibility**
- ✅ **Python 3.13 Compatible**: Works with latest Python versions
- 🎵 **Pattern-Based Detection**: Reliable keyword-based emotion analysis
- 📦 **Minimal Dependencies**: Only requires gTTS and PyPDF2

## ✨ Features

### 🎭 Emotional Voice Mapping
- **😊 Joy**: Australian accent (cheerful and upbeat)
- **😢 Sadness**: Slow US voice (melancholic and gentle)
- **❤️ Love**: British accent (warm and romantic)
- **😲 Surprise**: Canadian accent (animated and expressive)
- **😠 Anger**: Fast US voice (assertive and strong)
- **😰 Fear**: Slow nervous voice (tense and cautious)

### 🎵 Adaptive Background Music (Enhanced Version)
- **Sadness**: Soft rain sounds + low ambient tones
- **Joy**: Happy harmonies + upbeat melodies
- **Fear**: Tension drones + tremolo effects
- **Love**: Warm ambient pads + soft harmonics
- **Anger**: Rhythmic percussive elements
- **Neutral**: Gentle ambient soundscape

### 🔊 Contextual Sound Effects (Enhanced Version)
- **🚪 Door mentions** → Realistic door creak sounds
- **👣 Footsteps** → Rhythmic walking effects
- **💨 Wind** → Whoosh and breeze sounds
- **🔥 Fire** → Crackling flame effects
- **💧 Water** → Flowing river sounds
- **⛈️ Thunder** → Deep rumble effects

## 🚀 Quick Start

### Option 1: Enhanced Version (Recommended)
```bash
# Install dependencies
pip install streamlit gtts PyPDF2

# Run enhanced version
streamlit run enhanced_audiobook_ui.py
```

### Option 2: Working Version (Audio Combining)
```bash
# Requires FFmpeg for audio combining
brew install ffmpeg  # macOS
# or apt-get install ffmpeg  # Ubuntu

streamlit run working_audiobook_ui.py
```

### Option 3: Simple Version
```bash
streamlit run simple_streamlit_ui.py
```

### Option 4: Ultra-Compatible
```bash
streamlit run ultra_simple_ui.py
```

## 📖 How to Use

1. **Choose your version** and run the Streamlit app
2. **Upload a PDF** book file
3. **Configure settings** (if available in your chosen version)
4. **Click "Generate Audiobook"**
5. **Review emotion analysis** and voice assignments
6. **Download** individual segments or combined audiobook
7. **Listen** to your emotional audiobook!

## 🎯 Example Results

### Input Text:
> "The sad man walked through the creaking door as thunder rumbled outside, but suddenly his heart filled with joy when he saw her beautiful smile."

### Generated Output:
- **"The sad man walked"** → Slow US voice + rain background
- **"creaking door"** → Door creak sound effect
- **"thunder rumbled"** → Thunder rumble effect + tension drone
- **"heart filled with joy"** → Australian accent + happy harmony
- **"beautiful smile"** → British accent + warm ambient

## 🛠️ Technical Stack

- **Frontend**: Streamlit with custom CSS
- **Text Processing**: PyPDF2, regex-based segmentation
- **Emotion Detection**: Pattern-based keyword matching
- **Text-to-Speech**: Google Text-to-Speech (gTTS)
- **Audio Generation**: Pure Python sine wave synthesis
- **Audio Combining**: FFmpeg (working version)
- **File Formats**: MP3, WAV export

## 📁 Project Structure

```
audiobook/
├── enhanced_audiobook_ui.py      # Full-featured version
├── working_audiobook_ui.py       # Audio combining version  
├── simple_streamlit_ui.py        # Lightweight version
├── ultra_simple_ui.py            # Ultra-compatible version
├── audiobook_classes.py          # Core classes (original)
├── bark_narrator.py              # Bark TTS integration (original)
├── sound_effects.py              # Advanced audio effects (original)
├── main_generator.py             # Main orchestrator (original)
├── streamlit_ui.py               # Original full UI (requires PyTorch)
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── 1-the_gift_of_the_magi_0.pdf  # Test PDF
```

## 🔄 Version Comparison

| Feature | Enhanced | Working | Simple | Ultra-Simple |
|---------|----------|---------|--------|-------------|
| Emotional Voices | ✅ 6 styles | ✅ 6 styles | ✅ 6 styles | ✅ 6 styles |
| Background Music | ✅ Full | ❌ No | ❌ No | ❌ No |
| Sound Effects | ✅ Full | ❌ No | ❌ No | ❌ No |
| Audio Combining | ❌ No | ✅ FFmpeg | ❌ No | ❌ No |
| Python 3.13 | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| Dependencies | Medium | Medium | Low | Minimal |
| File Size | Large | Medium | Small | Tiny |

## 🚀 Production Roadmap

### Current Status: MVP Complete ✅
- ✅ Multiple voice styles with emotional mapping
- ✅ Pattern-based emotion detection
- ✅ Background music generation (Enhanced version)
- ✅ Contextual sound effects (Enhanced version)
- ✅ Professional web interface
- ✅ Audio export and combining

### Phase 1: AI Enhancement (Next)
- 🔄 **ML Emotion Detection**: HuggingFace transformers model
- 🔄 **Advanced TTS**: Bark or ElevenLabs integration
- 🔄 **Professional Audio**: Real instrument samples
- 🔄 **GPU Acceleration**: Faster processing

### Phase 2: Platform Development
- 🔄 **User Accounts**: Authentication and libraries
- 🔄 **Cloud Deployment**: AWS/GCP infrastructure
- 🔄 **API Development**: RESTful backend
- 🔄 **Mobile Apps**: iOS/Android native apps

### Phase 3: Business Features
- 🔄 **Subscription Model**: Free/Premium tiers
- 🔄 **Payment Processing**: Stripe integration
- 🔄 **Content Management**: Book categorization
- 🔄 **Analytics Dashboard**: Usage statistics

## 📊 Demo Files

- **Test PDF**: `1-the_gift_of_the_magi_0.pdf` - Perfect for testing emotional range
- **Expected Results**: Joy (6%), Sadness (5%), Love (8%), Neutral (64%), Surprise (4%), etc.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test with multiple PDF files
4. Submit a pull request

## 📝 License

MIT License - feel free to use for commercial or personal projects!

## 🔗 Links

- **GitHub**: https://github.com/Jit017/pdftoaudiobook
- **Demo**: Upload a PDF and try it live!
- **Issues**: Report bugs or request features

## 🙏 Acknowledgments

- **Google Text-to-Speech**: Reliable voice synthesis
- **Streamlit**: Beautiful web interface framework
- **FFmpeg**: Professional audio processing
- **"The Gift of the Magi"**: Perfect emotional test content

---

**Start creating emotional audiobooks today!** 🎧✨ 