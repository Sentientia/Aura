import os
import time
import numpy as np
import gradio as gr
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from collections import deque
import datetime
import librosa
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
from espnet2.bin.tts_inference import Text2Speech
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

latency_ASR = 0.0
latency_LLM = 0.0
latency_TTS = 0.0
conversation_history = []

HF_KEY = os.getenv('HF_API_KEY')

use_gpu = torch.cuda.is_available()
device = "cuda" if use_gpu else "cpu"
print(f"Using device: {device}")

ASR_OPTIONS = {
    "Whisper Tiny": "openai/whisper-tiny",
    # "Wav2Vec2 Small": "facebook/wav2vec2-base-960h",
    "ESPnet Librispeech": "espnet/simpleoier_librispeech_asr_train_asr_conformer_raw_en_bpe5000_sp",
    # "ESPnet English": "espnet/kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan",
    "OWSM CTC v3.1 1B": "espnet/owsm_ctc_v3.1_1B",
    # "OWSM CTC v3.2 1B": "espnet/owsm_ctc_v3.2_ft_1B"
}

# ESPnet TTS options
ESPNET_TTS_OPTIONS = {
    "ESPnet LJSpeech VITS": "kan-bayashi/ljspeech_vits", 
    "ESPnet LJSpeech FastSpeech2": "kan-bayashi/ljspeech_fastspeech2"
}

LLM_OPTIONS = {
    "DistilGPT2": "distilgpt2",
    "GPT-2 Small": "gpt2",
    # "Llama Fine-tuned": "llama-ft",
    # Add other options here
}

# Cache for loaded models
asr_models = {}
llm_models = {}
espnet_models = {}
tts_models = {}
owsm_models = {}

def synthesize_speech(text, tts_model_name=None):
    """Synthesize speech from text using the selected TTS model"""
    global latency_TTS
    
    start_time = time.time()
    print(f"Starting TTS synthesis with model: {tts_model_name}")
    
    audio_output = None
    
    try:
        try:
            import nltk
            nltk_resources = ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']
            for resource in nltk_resources:
                try:
                    nltk.data.find(f'taggers/{resource}')
                except LookupError:
                    print(f"Downloading NLTK resource: {resource}")
                    nltk.download(resource, quiet=True)
        except Exception:
            pass
        
        if tts_model_name and tts_model_name in ESPNET_TTS_OPTIONS:
            model = load_tts_model(tts_model_name)
            try:
                output = model(text)
                sr = model.fs
                audio_data = output["wav"].numpy()
                audio_output = (sr, audio_data)
                print(f"Successfully synthesized speech with {tts_model_name}")
            except Exception as tts_error:
                print(f"Error in ESPnet TTS: {tts_error}")
        
    except Exception as e:
        print(f"Error in TTS synthesis: {e}")

    
    latency_TTS = time.time() - start_time
    print(f"TTS synthesis completed in {latency_TTS:.2f} seconds")
    return audio_output

def load_tts_model(model_name):
    """Load TTS model with error handling"""
    global tts_models
    
    if model_name not in tts_models:
        print(f"Loading TTS model: {model_name}")
        
        if model_name in ESPNET_TTS_OPTIONS:
            model_id = ESPNET_TTS_OPTIONS[model_name]
            try:
                tts_models[model_name] = Text2Speech.from_pretrained(model_id)
                print(f"Successfully loaded ESPnet TTS model: {model_name}")
            except Exception as e:
                print(f"Error loading ESPnet TTS model {model_name}: {e}")
                tts_models[model_name] = None
    
    return tts_models[model_name]

def load_asr_model(model_name):
    """Load ASR model with error handling"""
    global asr_models, espnet_models, owsm_models
    
    if model_name not in asr_models and model_name not in espnet_models and model_name not in owsm_models:
        print(f"Loading ASR model: {model_name}")
        
        if "OWSM" in model_name:
            model_id = ASR_OPTIONS[model_name]
            try:
                owsm_models[model_name] = Speech2TextGreedySearch.from_pretrained(
                    model_id,
                    device="cpu",  
                    use_flash_attn=False,
                    lang_sym='<eng>',
                    task_sym='<asr>',
                    token=HF_KEY,
                )
                print(f"Successfully loaded OWSM ASR model: {model_name}")
            except Exception as e:
                print(f"Error loading OWSM ASR model {model_name}: {e}")
                owsm_models[model_name] = None
                
            return owsm_models[model_name]
            
        elif "ESPnet" in model_name and "OWSM" not in model_name:
            model_id = ASR_OPTIONS[model_name]
            try:
                d = ModelDownloader()
                model_dict = d.download_and_unpack(model_id)
                
                if "train_config" in model_dict:
                    model_dict.pop("train_config")
                
                speech2text = Speech2Text(
                    **model_dict,
                    device="cpu",
                    minlenratio=0.0,
                    maxlenratio=0.0,
                    ctc_weight=0.3,
                    beam_size=10,
                    batch_size=0,
                    nbest=1,
                    token=HF_KEY
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
                    device=-1  # Fix this
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

def load_llm_model(model_name):
    """Load LLM model with error handling"""
    global llm_models
    
    if model_name not in llm_models:
        print(f"Loading LLM model: {model_name}")
        model_id = LLM_OPTIONS[model_name]
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_KEY)
            model = AutoModelForCausalLM.from_pretrained(model_id, token=HF_KEY)
            
            llm_models[model_name] = {
                "model": model,
                "tokenizer": tokenizer
            }
            print(f"Successfully loaded LLM model: {model_name}")
        except Exception as e:
            print(f"Error loading LLM model {model_name}: {e}")
            llm_models[model_name] = None
    
    return llm_models[model_name]

def transcribe_audio(audio_data, sr, asr_model_name):
    """Transcribe audio using selected ASR model"""
    global latency_ASR
    
    start_time = time.time()
    
    try:
        model = load_asr_model(asr_model_name)
        
        if model is None:
            transcript = "ASR model unavailable. Please try a different model."
        else:
            # Convert audio to float32 for compatibility with ASR models
            if audio_data.dtype != np.float32:
                # Normalize to [-1, 1] range for float32
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
    
    latency_ASR = time.time() - start_time
    return transcript

def generate_response(transcript, llm_model_name, system_prompt):
    """Generate response using selected LLM model"""
    global latency_LLM, conversation_history
    
    start_time = time.time()
    print(f"Starting LLM response generation with model: {llm_model_name}")
    
    try:
        model_info = load_llm_model(llm_model_name)
        
        if model_info is None:
            print(f"LLM model {llm_model_name} is unavailable")
            response = "LLM model unavailable. Please try a different model."
        else:
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            # Create input prompt
            prompt = f"{system_prompt}\nUser: {transcript}\nAssistant:"
            
            # Generate text
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode the response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the response part
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            # Add to conversation history
            conversation_history.append({"role": "user", "content": transcript})
            conversation_history.append({"role": "assistant", "content": response})
    
    except Exception as e:
        print(f"Error in LLM response generation: {e}")
        response = "I'm sorry, I encountered an error generating a response. Please try again."
    
    latency_LLM = time.time() - start_time
    print(f"LLM response generation completed in {latency_LLM:.2f} seconds")
    return response

def preload_all_models():
    """Pre-download and cache all models at startup"""
    print("=" * 50)
    print("Starting pre-downloading models...")
    print("=" * 50)
    
    # Pre-load ASR models
    for model_name in ASR_OPTIONS.keys():
        print(f"Pre-downloading ASR model: {model_name}")
        load_asr_model(model_name)
    
    # Pre-load LLM models
    for model_name in LLM_OPTIONS.keys():
        print(f"Pre-downloading LLM model: {model_name}")
        load_llm_model(model_name)
    
    # Pre-load TTS models
    for model_name in ESPNET_TTS_OPTIONS.keys():
        print(f"Pre-downloading TTS model: {model_name}")
        load_tts_model(model_name)
    
    print("=" * 50)
    print("Finished pre-downloading all models")
    print("=" * 50)

def process_speech(audio_input, asr_option, llm_option, system_prompt, tts_option=None):
    """Process speech: ASR -> LLM -> TTS pipeline"""
    print("Starting speech processing pipeline")
    
    # Check if audio input is available
    if audio_input is None:
        print("No audio input provided")
        return None, "", "", None
    
    try:
        # Get audio data
        sr, audio_data = audio_input
        
        # Log audio characteristics for debugging
        print(f"Audio input: SR={sr}, shape={audio_data.shape}, dtype={audio_data.dtype}")
        
        # ASR: Speech to text
        print("Starting ASR step")
        transcript = transcribe_audio(audio_data, sr, asr_option)
        print(f"ASR transcript: {transcript}")
        
        # LLM: Generate response
        print("Starting LLM step")
        response = generate_response(transcript, llm_option, system_prompt)
        print(f"LLM response: {response}")
        
        # TTS: Text to speech
        print("Starting TTS step")
        audio_output = synthesize_speech(response, tts_option)
        
        # Check if audio_output is None (indicating an error in speech synthesis)
        if audio_output is None:
            print("Failed to synthesize speech, returning error")
            return audio_input, transcript, response, None
            
        print("Speech processing pipeline completed successfully")
        
        # Return results
        return audio_input, transcript, response, audio_output
    except Exception as e:
        print(f"Error in process_speech: {e}")
        return audio_input, f"Error processing audio: {str(e)}", "I couldn't process your speech properly. Please try again.", None

def display_latency():
    """Display latency information"""
    return f"""
    ASR Latency: {latency_ASR:.2f} seconds
    LLM Latency: {latency_LLM:.2f} seconds
    TTS Latency: {latency_TTS:.2f} seconds
    Total Latency: {latency_ASR + latency_LLM + latency_TTS:.2f} seconds
    """

def reset_conversation():
    """Reset the conversation history"""
    global conversation_history
    print("Resetting conversation history")
    conversation_history = []
    return None, "", "", None, ""

def display_conversation():
    """Display the conversation history with enhanced styling and animations"""
    if not conversation_history:
        return """
        <div style="background-color: #f5f7fa; border-radius: 12px; padding: 20px; font-family: 'Segoe UI', Arial, sans-serif; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <div style="display: flex; justify-content: center; align-items: center; height: 100px;">
                <p style="color: #8796ab; text-align: center; font-size: 16px;">No conversation yet. Start by recording your message.</p>
            </div>
        </div>
        """
    
    # Start with container div with improved styling and max-height for scrolling
    output = """
    <div style="background-color: #f5f7fa; border-radius: 12px; padding: 0; max-height: 500px; overflow-y: auto; font-family: 'Segoe UI', Arial, sans-serif; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
        <div style="position: sticky; top: 0; background-color: #ffffff; border-radius: 12px 12px 0 0; padding: 15px 20px; border-bottom: 1px solid #e1e5eb; z-index: 10;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="color: #2d3748; margin: 0; font-weight: 600; font-size: 18px;">Conversation History</h3>
                <div style="background-color: #e6f2ff; color: #3182ce; font-size: 12px; padding: 4px 10px; border-radius: 12px; font-weight: 500;">{} messages</div>
            </div>
        </div>
        <div style="padding: 20px;">
    """.format(len(conversation_history))
    
    # Format timestamp with improved styling
    timestamp = datetime.datetime.now().strftime("%B %d, %Y %H:%M")
    output += f'<div style="font-size: 12px; color: #8796ab; text-align: center; margin-bottom: 20px; padding: 6px 12px; background-color: #edf2f7; border-radius: 12px; display: inline-block;">{timestamp}</div>'
    
    # Add each message with animation classes and improved styling
    for i, item in enumerate(conversation_history):
        role = item["role"]
        content = item["content"]
        
        # Add animation delay based on message position
        animation_delay = i * 0.1
        
        if role == "user":
            # User message styling with animation
            output += f"""
            <div class="message-container" style="margin-bottom: 20px; animation: fadeIn 0.3s ease forwards; animation-delay: {animation_delay}s; opacity: 0;">
                <div style="display: flex; align-items: flex-start;">
                    <div style="background-color: #4285f4; color: white; border-radius: 50%; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; margin-right: 12px; flex-shrink: 0; box-shadow: 0 2px 5px rgba(66,133,244,0.3);">
                        <span style="font-weight: 600;">U</span>
                    </div>
                    <div style="background-color: #e6f2ff; border-radius: 3px 18px 18px 18px; padding: 12px 16px; max-width: 85%; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                        <p style="margin: 0; color: #2d3748; line-height: 1.5;">{content}</p>
                        <div style="font-size: 11px; color: #8796ab; text-align: right; margin-top: 5px;">
                            {datetime.datetime.now().strftime("%H:%M")}
                        </div>
                    </div>
                </div>
            </div>
            """
        else:
            # Assistant message styling with animation
            output += f"""
            <div class="message-container" style="margin-bottom: 20px; display: flex; justify-content: flex-end; animation: fadeIn 0.3s ease forwards; animation-delay: {animation_delay}s; opacity: 0;">
                <div style="display: flex; align-items: flex-start; flex-direction: row-reverse;">
                    <div style="background-color: #10b981; color: white; border-radius: 50%; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; margin-left: 12px; flex-shrink: 0; box-shadow: 0 2px 5px rgba(16,185,129,0.3);">
                        <span style="font-weight: 600;">A</span>
                    </div>
                    <div style="background-color: #f8fafc; border-radius: 18px 3px 18px 18px; padding: 12px 16px; max-width: 85%; box-shadow: 0 2px 5px rgba(0,0,0,0.05); border-left: 3px solid #10b981;">
                        <p style="margin: 0; color: #2d3748; line-height: 1.5;">{content}</p>
                        <div style="font-size: 11px; color: #8796ab; text-align: right; margin-top: 5px;">
                            {datetime.datetime.now().strftime("%H:%M")}
                        </div>
                    </div>
                </div>
            </div>
            """
    
    # Add CSS animations
    output += """
        </div>
    </div>
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message-container {
            transition: all 0.3s ease;
        }
        
        /* Scrollbar styling */
        div::-webkit-scrollbar {
            width: 8px;
        }
        
        div::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        div::-webkit-scrollbar-thumb {
            background: #cbd5e0;
            border-radius: 10px;
        }
        
        div::-webkit-scrollbar-thumb:hover {
            background: #a0aec0;
        }
    </style>
    """
    
    return output

# Create Gradio interface with styled conversation
def create_demo():
    with gr.Blocks(title="Speech Conversation System", css="""
        .conversation-container {border: 1px solid #ddd; border-radius: 10px; overflow: hidden;}
        .conversation-container h3 {background-color: #f5f5f5; padding: 12px; margin: 0; border-bottom: 1px solid #ddd;}
    """) as demo:
        gr.Markdown(
            """
            # Speech-based Task Oriented Dialogue System with Tool Use Integration
            
            This demo showcases a speech-to-speech task-oriented dialogue agent that results in tool use
            
            **Record your speech using the microphone, then click submit to process your query.**
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                audio_input = gr.Audio(
                    label="Record your speech",
                    sources=["microphone"],
                    type="numpy",
                    streaming=False
                )
                
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value="You are a helpful AI assistant."
                )
                
                with gr.Row():
                    asr_dropdown = gr.Dropdown(
                        choices=list(ASR_OPTIONS.keys()),
                        value=list(ASR_OPTIONS.keys())[0],
                        label="Select ASR Model"
                    )
                    
                    llm_dropdown = gr.Dropdown(
                        choices=list(LLM_OPTIONS.keys()),
                        value=list(LLM_OPTIONS.keys())[0],
                        label="Select LLM Model"
                    )
                
                tts_choices = list(ESPNET_TTS_OPTIONS.keys()) 
                tts_dropdown = gr.Dropdown(
                    choices=tts_choices,
                    value=tts_choices[0],
                    label="Select TTS Model"
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Process Speech", variant="primary")
                    reset_btn = gr.Button("Reset Conversation")
                
                # Output elements
                user_transcript = gr.Textbox(label="Your Speech (ASR Output)")
                system_response = gr.Textbox(label="AI Response (LLM Output)")
                audio_output = gr.Audio(label="AI Voice Response", autoplay=True)
                latency_info = gr.Textbox(label="Performance Metrics")
            
            # Right column for conversation history
            with gr.Column(scale=1, elem_classes="conversation-container"):
                conversation_display = gr.HTML(label="Conversation History")
        
        # Event handlers
        submit_btn.click(
            process_speech,
            inputs=[audio_input, asr_dropdown, llm_dropdown, system_prompt, tts_dropdown],
            outputs=[audio_input, user_transcript, system_response, audio_output]
        ).then(
            display_latency,
            inputs=[],
            outputs=[latency_info]
        ).then(
            display_conversation,
            inputs=[],
            outputs=[conversation_display]
        )
        
        reset_btn.click(
            reset_conversation,
            inputs=[],
            outputs=[audio_input, user_transcript, system_response, audio_output, latency_info]
        ).then(
            display_conversation,
            inputs=[],
            outputs=[conversation_display]
        )
    
    return demo

if __name__ == "__main__":
    print("=" * 50)
    print("Starting Speech Conversation System")
    print(f"Running in {'GPU' if device == 'cuda' else 'CPU'} mode")
    print("=" * 50)
    
    #preload_all_models()
    
    demo = create_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True
    )