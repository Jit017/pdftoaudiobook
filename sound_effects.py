# ================================================================
# 🔊 5. ADD ADAPTIVE SOUND FX
# ================================================================

import os
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import normalize
from scipy import signal
from audiobook_classes import AudiobookConfig

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

import json
from typing import List
from tqdm.auto import tqdm
from audiobook_classes import TextSegment

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