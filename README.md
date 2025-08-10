# AURA: Agent for Understanding, Reasoning, and Automated Tool Use

[![Paper](https://img.shields.io/badge/arXiv-2506.23049-b31b1b.svg)](https://arxiv.org/abs/2506.23049)
[![Website](https://img.shields.io/badge/Website-Coming%20Soon-blue.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.43.0-orange.svg)](https://gradio.app)

**AURA** is the first open-source, speech-native assistant capable of completing complex, goal-driven tasks through dynamic tool invocation and multi-turn conversation. Despite advances in language and speech technologies, no previous open-source system enabled full speech-to-speech, multi-turn dialogue with integrated tool use and agentic reasoning.

![AURA Overview](docs/images/aura_overview.png)

## 🎯 Key Features

- **Speech-Native**: Full speech-to-speech interaction with natural conversation flow
- **Tool Integration**: Dynamic tool invocation for calendar booking, contact lookup, web search, email, and more
- **Multi-Turn Dialogue**: Maintains context across conversation turns for complex task completion
- **Modular Design**: Easy integration of new tools using natural language prompts and action classes
- **Open-Source**: Built entirely with open-weight models (ASR, TTS, LLMs)
- **High Performance**: 92.75% on OpenBookQA (outperforming all open-weight systems), 90% task success on human evaluation

## 🎬 Demo

[![Aura Demo](https://img.youtube.com/vi/cb7w0GVwwF0/0.jpg)](https://www.youtube.com/watch?v=cb7w0GVwwF0)

## 🏗️ System Architecture

![Aura System Architecture](docs/images/aura_system_white.png)

AURA employs a cascaded pipeline architecture that combines:
- **ASR (Automatic Speech Recognition)**: Converts speech input to text with accent adaptation
- **LLM Agent**: Processes text, reasons about tasks, and decides on tool usage using ReAct-based reasoning
- **TTS (Text-to-Speech)**: Converts agent responses back to natural speech
- **Tool Integration**: Seamless integration with external APIs and services

### Technical Approach

AURA combines open-weight ASR, TTS, and LLMs in a cascaded pipeline design that enables:

1. **Accent-Adaptive ASR**: Fine-tuned speech recognition models that adapt to different accents and speaking styles
2. **ReAct-Based Reasoning**: The agent uses Reasoning and Acting (ReAct) paradigm to break down complex tasks
3. **Dynamic Tool Selection**: Intelligent selection and invocation of appropriate tools based on user intent
4. **Context Preservation**: Maintains conversation context across multiple turns for coherent task completion
5. **Modular Architecture**: Easy extensibility for adding new tools and capabilities

## 📁 Repository Structure

```
.
├── agent/                      # Core agent implementation
│   ├── actions/                # Action handlers for different tasks
│   │   ├── calendar_action.py  # Calendar booking functionality
│   │   ├── contact_action.py   # Contact lookup and management
│   │   ├── email_action.py     # Email composition and sending
│   │   ├── web_search_action.py # Web search capabilities
│   │   └── chat_action.py      # General chat functionality
│   ├── controller/             # Agent state and control logic
│   ├── llm/                    # Language model integration
│   ├── agenthub/               # Agent implementations (QA, Chat agents)
│   ├── speech_utils/           # Speech processing utilities
│   ├── dst/                    # Dialog State Tracking components
│   └── secrets_example/        # Example credential configuration
│
├── ui/                         # User interface components
│   ├── local_speech_app.py     # Gradio-based speech interface
│   └── requirements.txt        # UI-specific dependencies
│
├── accent_adaptive_asr/        # Accent-adaptive ASR with fine-tuning
│   ├── asr_ft.py              # ASR fine-tuning script
│   ├── evaluate.py            # ASR evaluation utilities
│   └── config/                # ASR configuration files
│
├── dst/                        # Dialog State Tracking fine-tuning
│   ├── finetune/              # DST model fine-tuning scripts
│   ├── generate/              # DST inference scripts
│   └── evaluation_JGA_DSP/    # DST evaluation metrics
│
├── evaluation/                 # Evaluation frameworks
│   └── voicebench/            # VoiceBench evaluation scripts
│
├── llm_serve/                  # Language model serving utilities
│
├── LLaMA-Factory/              # LLM fine-tuning framework
│
├── docs/                       # Documentation and assets
│   └── images/                # System diagrams and screenshots
│
└── environment.yml            # Conda environment configuration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Conda or Miniconda
- CUDA-compatible GPU (recommended for optimal performance)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sentientia/Aura.git
   cd Aura
   ```

2. **Create the conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate sentientia
   ```

3. **Set Python path**
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

### Configuration

4. **Set LLM environment variables**
   ```bash
   export LLM_API_KEY="your_api_key_here"
   export LLM_API_BASE="your_api_base_url"
   export LLM_MODEL="your_model_identifier"
   ```

5. **Setup tool credentials** (Required for tool use, optional for chat-only functionality)
   
   Copy the example secrets directory and configure your credentials:
   ```bash
   cp -r agent/secrets_example agent/secrets
   ```
   
   Configure the following in `agent/secrets/`:
   - **Google Cloud Platform**: Set up GCP account, enable necessary APIs, and place `credentials.json`
   - **SerpAPI**: Get API key for web search functionality
   - **Email**: Configure email service credentials if using email actions

### Running AURA

6. **Launch the speech interface**
   ```bash
   python ui/local_speech_app.py
   ```
   
   The Gradio interface will be available at `http://localhost:7860`

## 🛠️ Advanced Usage

### Fine-tuning Components

#### ASR Fine-tuning
```bash
cd accent_adaptive_asr
bash ft.sh
```

#### Dialog State Tracking Fine-tuning
```bash
cd dst
python prepare_data.py  # Prepare training data
# Use the configuration files in dst/finetune/configs/ for fine-tuning
```

### Evaluation

#### VoiceBench Evaluation
```bash
cd evaluation/voicebench
python eval.py
```

#### ASR Evaluation
```bash
cd accent_adaptive_asr
python evaluate.py
```

## 📊 Performance

AURA achieves state-of-the-art performance on multiple benchmarks:

- **VoiceBench OpenBookQA**: 92.75% (outperforming all open-weight systems, approaching GPT-4o)
- **AlpacaEval**: 4.39 (competitive with other open-weight systems)
- **Human Evaluation**: 90% task success rate on complex, multi-turn speech tasks

## 🔧 Adding New Tools

AURA's modular design makes it easy to add new tools:

1. Create a new action class in `agent/actions/`
2. Implement the required methods following the existing action patterns
3. Register the action in the agent controller
4. Add any necessary credentials to the secrets configuration

Example action structure:
```python
from agent.actions.action import Action

class MyCustomAction(Action):
    def __init__(self):
        super().__init__()
        self.name = "my_custom_action"
        self.description = "Description of what this action does"
    
    def execute(self, params):
        # Implementation here
        pass
```

## 🔗 Supported Tools

- **📅 Calendar**: Book appointments, check availability, manage schedules
- **👥 Contacts**: Look up contact information, add new contacts
- **🔍 Web Search**: Search the web for information using SerpAPI
- **📧 Email**: Compose and send emails
- **💬 Chat**: General conversation and question answering

## 🔧 Troubleshooting

### Common Issues

**Environment Setup**
- Ensure you're using Python 3.9+ and have activated the conda environment
- If you encounter CUDA issues, verify your GPU drivers and CUDA installation
- For M1/M2 Macs, some dependencies may require additional configuration

**Speech Processing**
- Microphone permissions may be required for speech input
- Audio quality affects ASR performance - use a good quality microphone
- Check audio device settings if speech input is not working

**Tool Integration**
- Verify all required API keys are set in environment variables
- Check that credentials in `agent/secrets/` are properly configured
- Some tools require specific permissions (e.g., Google Calendar API access)

**Memory Requirements**
- AURA requires significant memory for speech models
- Consider using smaller model variants if running on limited hardware
- GPU memory of 8GB+ recommended for optimal performance

### Getting Help

- Check the [Issues](https://github.com/Sentientia/Aura/issues) page for known problems
- Review the paper for technical details and methodology
- Examine the example configurations in `agent/secrets_example/`

## 📚 Research & Development

This project is part of ongoing research in speech-native AI assistants. For detailed technical information, please refer to our paper:

**"AURA: Agent for Understanding, Reasoning, and Automated Tool Use in Voice-Driven Tasks"**  
*Leander Melroy Maben, Gayathri Ganesh Lakshmy, Srijith Radhakrishnan, Siddhant Arora, Shinji Watanabe*

### Human-in-the-Loop Data
Research data and annotations are available at: [Google Sheets](https://docs.google.com/spreadsheets/d/16_DApAlgunmG3pR4f8p9JYjO-v-2m8ZxduN9fZ-AblI/edit?usp=sharing)

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open-source. Please check the repository for license details.

## 📖 Citation

If you use AURA in your research, please cite our paper:

```bibtex
@article{maben2025aura,
  title={AURA: Agent for Understanding, Reasoning, and Automated Tool Use in Voice-Driven Tasks},
  author={Maben, Leander Melroy and Lakshmy, Gayathri Ganesh and Radhakrishnan, Srijith and Arora, Siddhant and Watanabe, Shinji},
  journal={arXiv preprint arXiv:2506.23049},
  year={2025}
}
```

## 🙏 Acknowledgments

We thank the open-source community for the foundational models and tools that made AURA possible, including ESPnet, Transformers, and the broader speech and NLP research community.