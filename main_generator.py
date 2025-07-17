# ================================================================
# 🎬 MAIN AUDIOBOOK GENERATOR CLASS
# ================================================================

import os
from typing import List
from audiobook_classes import AudiobookConfig, EmotionAnalyzer, load_document
from bark_narrator import BarkNarrator
from sound_effects import SoundEffectMixer, AudiobookExporter

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

import torch
from IPython.display import Audio, display

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
    try:
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
        
    except ImportError:
        print("⚠️ ReportLab not available. Using text file instead.")
        
        # Use text file directly (modify DocumentLoader to handle .txt)
        audiobook_path = generator.generate_audiobook(
            input_file=sample_file,
            output_dir="/content/audiobook_output",
            output_filename="demo_emotional_audiobook.wav"
        )
        
        print("\n🎵 Playing generated audiobook:")
        display(Audio(audiobook_path))
        
        return audiobook_path

# ================================================================
# 📱 INTERACTIVE COLAB INTERFACE
# ================================================================

def create_interactive_interface():
    """Create an interactive interface for Google Colab"""
    
    try:
        from google.colab import files
        from IPython.display import display, HTML, clear_output
        import ipywidgets as widgets
    except ImportError:
        print("⚠️ This function requires Google Colab environment")
        return
    
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