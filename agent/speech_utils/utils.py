import librosa
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
from espnet2.bin.tts_inference import Text2Speech
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch
import os
import torch
import nltk
import time
import re
import numpy as np
from transformers import pipeline

HF_TOKEN = os.getenv('HF_TOKEN')
CACHE_DIR = os.getenv('CACHE_DIR',"tmp/cache")

# Improved device handling - consistent v
# 
# ariable for all components
if torch.cuda.is_available():
    device = "cuda"
    torch_device = torch.device("cuda")
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    torch_device = torch.device("cpu")
    print("CUDA is not available. Using CPU for processing.")


try:
    nltk_resources = ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']
    for resource in nltk_resources:
        try:
            nltk.data.find(f'taggers/{resource}')
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)
except Exception as e:
    print(f"Failed to download NLTK resources: {e}")


# Cache for loaded models
asr_models = {}
llm_models = {}
espnet_models = {}
tts_models = {}
owsm_models = {}

# ESPnet TTS options
ESPNET_TTS_OPTIONS = {
    "ESPnet LJSpeech VITS": "kan-bayashi/ljspeech_vits", 
    "ESPnet LJSpeech FastSpeech2": "kan-bayashi/ljspeech_fastspeech2"
}

ASR_OPTIONS = {
    "OWSM CTC v3.1 1B": "espnet/owsm_ctc_v3.1_1B",
    "Whisper Tiny": "openai/whisper-tiny",
    "OWSM v3.2": "espnet/owsm_v3.2",
    "OWSM v3.1 EBF Small": "espnet/owsm_v3.1_ebf_small",
    # "Wav2Vec2 Small": "facebook/wav2vec2-base-960h",
    # "ESPnet English": "espnet/kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan",
    
    # "OWSM CTC v3.2 1B": "espnet/owsm_ctc_v3.2_ft_1B"
}




def load_asr_model(model_name):
    """Load ASR model with error handling"""
    global asr_models, espnet_models, owsm_models
    
    if model_name not in asr_models and model_name not in espnet_models and model_name not in owsm_models:
        print(f"Loading ASR model: {model_name}")
        
        if "OWSM" in model_name:
            model_id = ASR_OPTIONS[model_name]
            try:
                d = ModelDownloader(cachedir = CACHE_DIR)
                model_dict = d.download_and_unpack(model_id)
                
                if "train_config" in model_dict:
                    model_dict.pop("train_config")
                owsm_models[model_name] = Speech2TextGreedySearch(
                    **model_dict,
                    device=device,  # Use global device
                    use_flash_attn=torch.cuda.is_available(),  # Only use if CUDA is available
                    lang_sym='<eng>',
                    task_sym='<asr>',
                )
                print(f"Successfully loaded OWSM ASR model: {model_name}")
            except Exception as e:
                print(f"Error loading OWSM ASR model {model_name}: {e}")
                owsm_models[model_name] = None
                
            return owsm_models[model_name]
            
        elif "ESPnet" in model_name and "OWSM" not in model_name:
            model_id = ASR_OPTIONS[model_name]
            try:
                d = ModelDownloader(cachedir = CACHE_DIR)
                model_dict = d.download_and_unpack(model_id)
                
                if "train_config" in model_dict:
                    model_dict.pop("train_config")
                
                speech2text = Speech2Text(
                    **model_dict,
                    device=device,  # Use global device
                    minlenratio=0.0,
                    maxlenratio=0.0,
                    ctc_weight=0.3,
                    beam_size=10,
                    batch_size=0,
                    nbest=1,
                )
                espnet_models[model_name] = speech2text
                print(f"Successfully loaded ESPnet ASR model: {model_name}")
            except Exception as e:
                print(f"Error loading ESPnet ASR model {model_name}: {e}")
                espnet_models[model_name] = None
            
            return espnet_models[model_name]
        else:
            model_id = ASR_OPTIONS[model_name]
            try:
                asr_models[model_name] = pipeline(
                    "automatic-speech-recognition", 
                    model=model_id,
                    device=0 if device == "cuda" else -1  # Use 0 for CUDA, -1 for CPU
                )
                print(f"Successfully loaded ASR model: {model_name}")
            except Exception as e:
                print(f"Error loading ASR model {model_name}: {e}")
                asr_models[model_name] = None
    
    if model_name in owsm_models:
        return owsm_models[model_name]
    elif model_name in espnet_models:
        return espnet_models[model_name]
    else:
        return asr_models[model_name]
    

def load_tts_model(model_name):
    """Load TTS model with error handling"""
    global tts_models
    
    if model_name not in tts_models:
        print(f"Loading TTS model: {model_name}")
        
        if model_name in ESPNET_TTS_OPTIONS:
            model_id = ESPNET_TTS_OPTIONS[model_name]
            try:
                # Use the global device variable
                tts_models[model_name] = Text2Speech.from_pretrained(model_id, device=device)
                print(f"Successfully loaded ESPnet TTS model: {model_name}")
            except Exception as e:
                print(f"Error loading ESPnet TTS model {model_name}: {e}")
                tts_models[model_name] = None
    
    return tts_models[model_name]


def synthesize_speech(text, tts_model_name=None):
    """Synthesize speech from text using the selected TTS model"""
    print(f"Starting TTS synthesis with model: {tts_model_name}")
    
    audio_output = None
    
    try:
        if not tts_model_name or tts_model_name not in ESPNET_TTS_OPTIONS:
            print("Invalid or missing TTS model name")
            return None
            
        model = load_tts_model(tts_model_name)
        if model is None:
            print("Failed to load TTS model")
            return None
            
        # Check if text is valid
        if not text or len(text.strip()) == 0:
            print("Empty text input provided to TTS, skipping synthesis")
            return None
        
        # Pre-process the text to ensure it's valid for the model
        cleaned_text = re.sub(r'[^\w\s.,!?;:\-\'"]', ' ', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        if len(cleaned_text) == 0:
            print("Text contains only special characters, using fallback text")
            cleaned_text = "I'm sorry, I couldn't process that text."
        
        print(f"Processing TTS with text: '{cleaned_text}'")
        
        # Try synthesis with different text lengths if needed
        try:
            output = model(cleaned_text)
        except Exception as e:
            print(f"TTS attempt failed: {e}")
            
            # If text is too long, try with shorter text
            if len(cleaned_text) > 100:
                print("Trying with shorter text")
                shortened_text = cleaned_text[:100] + "..."
                try:
                    output = model(shortened_text)
                except Exception:
                    print("Shortened text attempt failed, using fallback")
                    output = model("I'm sorry, I couldn't synthesize the response.")
            else:
                # If text is already short, use fallback
                print("Using fallback text for TTS")
                output = model("I'm sorry, I couldn't synthesize the response.")
        
        # Extract audio data
        sr = model.fs
        audio_data = output["wav"].cpu().numpy()
        audio_output = (sr, audio_data)
        print(f"Successfully synthesized speech with {tts_model_name}")
            
    except Exception as e:
        print(f"Error in TTS synthesis: {e}")
        # No need for additional fallback here since we already have fallbacks above
    
    return audio_output

def transcribe_audio(audio_data, sr, asr_model_name):
    """Transcribe audio using selected ASR model"""
    
    try:
        model = load_asr_model(asr_model_name)
        
        if model is None:
            transcript = "ASR model unavailable. Please try a different model."
        else:
            # Convert audio to float32 for compatibility with ASR models
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                if audio_data.max() > 1.0:
                    audio_data = audio_data / 32768.0  # Normalize from int16 range
            
            # Check if it's an OWSM model 
            if "OWSM" in asr_model_name:
                # Ensure audio is at the right sample rate
                if sr != 16000:
                    # Resample audio to 16kHz if needed
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
                    sr = 16000
                
                # Try direct call method
                result = model(audio_data)
                
                # Process and clean the result - remove <eng><asr> tags
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], tuple) and len(result[0]) > 0:
                        transcript = result[0][0]  # Get first hypothesis
                    else:
                        transcript = str(result[0])
                else:
                    transcript = str(result)
                
                # Clean OWSM output by removing tags
                if "<eng><asr>" in transcript:
                    transcript = transcript.replace("<eng><asr>", "")
                
            # Check if it's an ESPnet model
            elif "ESPnet" in asr_model_name:
                nbests = model(audio_data)
                transcript = nbests[0][0]  # Get the best hypothesis
            else:
                # Use transformers pipeline
                result = model({"array": audio_data, "sampling_rate": sr})
                transcript = result["text"]
        
    except Exception as e:
        print(f"Error in ASR transcription: {e}")
        transcript = "Error transcribing audio. Please try again."
    
    return transcript

def get_transcript(audio, sr, asr_model_name):
    """
    Get user input via recording and transcription
    
    Args:
        audio_input: Audio input data from Gradio
        asr_model_name: The ASR model to use for transcription
        
    Returns:
        str: The transcribed text from the user's speech
    """
    if audio is None:
        print("No audio input provided")
        return ""
    
    try:
        print(f"Processing user input - Audio: SR={sr}, shape={audio.shape}")
        
        # Audio data is already loaded as numpy array, no need to save to disk
        # We can pass it directly to the transcribe_audio function
        transcript = transcribe_audio(audio, sr, asr_model_name)
        print(f"User input transcribed: {transcript}")
        return transcript
    except Exception as e:
        print(f"Error in transcription: {e}")
        return ""
    
def get_audio(response_text, tts_model_name=None):
    """
    Generate speech output for agent response
    
    Args:
        response_text: Text response from the agent
        tts_model_name: The TTS model to use for synthesis
        
    Returns:
        tuple: Audio output data (sample_rate, audio_data)
    """
    print(f"Generating speech for agent response: {response_text}")
    
    # Synthesize speech from the response text
    audio_output = synthesize_speech(response_text, tts_model_name)
    
    # Handle TTS failure
    if audio_output is None:
        print("Failed to synthesize speech, returning error message")
        
    return audio_output


