# Aura: Agent for Understanding, Reasoning and Automation

We develop a cascaded voice assistant system that includes ASR, TTS and a 
ReAct based Agent for reasoning and action taking.

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://sentientia.github.io/Aura/)

## Aura: Demo

[![Aura Demo](https://img.youtube.com/vi/cb7w0GVwwF0/0.jpg)](https://www.youtube.com/watch?v=cb7w0GVwwF0)

## System Architecture

![Aura System Architecture](docs/images/aura_system_white.png)

## Documentation

For detailed documentation, please visit our [Documentation Website](https://sentientia.github.io/Aura/).

The documentation includes:
- [Installation Guide](https://sentientia.github.io/Aura/installation/)
- [Architecture Overview](https://sentientia.github.io/Aura/architecture/)
- [Agent Documentation](https://sentientia.github.io/Aura/agents/)
- [Action Documentation](https://sentientia.github.io/Aura/actions/)
- [UI Documentation](https://sentientia.github.io/Aura/ui/)
- [Contributing Guide](https://sentientia.github.io/Aura/contributing/)

## Repository Structure

```
.
├── agent/                   # Core agent implementation
│   ├── actions/             # Action handlers for different tasks
│   ├── controller/          # Agent state and control logic
│   ├── llm/                 # Language model integration
│   ├── secrets/             # Secure credential storage
│   └── agenthub/            # Agent implementations
│
├── ui/                      # User interface components
│   ├── local_speech_app.py  # Speech interface implementation (using gradio)
│   └── requirements.txt     # UI dependencies
│
├── accent_adaptive_asr/     # Accent-adaptive speech recognition including finetuning
│
├── llm_serve/               # Language model serving script
│
├── dst/                     # Dialog State Tracking. Has the scripts for finetuning LLMs for DST
│
└── environment.yaml         # Conda environment configuration
```

## Quick Setup

1. Create the conda environment:
   ```bash
   conda env create -f environment.yaml
   ```

2. Set python Path
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```
3. Set LLM related environment variables:
    - `LLM_API_KEY`: API key for the language model
    - `LLM_API_BASE`: Base URL for the language model API
    - `LLM_MODEL`: Model identifier for the language model

4. Setup secrets (Required for tool use. Not needed if you are only using the chat functionality)
    Secrets are used to communicate with external APIs. follow the format in agent/secrets_example and change the name of the directory to agent/secrets
    You will need to setup a google Cloud Platform account, give necessary permissions, get the credential.json file. You will also need a SerpAPI key for web search.

5. Launch the gradio app:
 ```bash
    python ui/local_speech_app.py
 ```

For more detailed setup instructions, please refer to the [Installation Guide](https://sentientia.github.io/Aura/installation/) in our documentation.

## Human in the Loop Data

For human-in-the-loop data, please visit [this Google Sheets document](https://docs.google.com/spreadsheets/d/16_DApAlgunmG3pR4f8p9JYjO-v-2m8ZxduN9fZ-AblI/edit?usp=sharing).