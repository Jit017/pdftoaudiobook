# ================================================================
# 🎙️ 4. GENERATE EMOTIONAL NARRATION (BARK)
# ================================================================

import os
import numpy as np
import soundfile as sf
from bark import generate_audio, preload_models
from tqdm.auto import tqdm
from typing import List
from audiobook_classes import AudiobookConfig, TextSegment

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