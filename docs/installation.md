# Installation

This guide will walk you through the process of setting up Aura on your local machine.

## Prerequisites

Before installing Aura, make sure you have the following prerequisites installed:

- Python 3.8 or higher
- Conda (recommended for environment management)
- Git

## Setup Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/Sentientia/Aura.git
   cd Aura
   ```

2. Create the conda environment:
   ```bash
   conda env create -f environment.yaml
   ```

3. Activate the conda environment:
   ```bash
   conda activate aura
   ```

4. Set the Python path:
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

5. Set LLM related environment variables:
   - `LLM_API_KEY`: API key for the language model
   - `LLM_API_BASE`: Base URL for the language model API
   - `LLM_MODEL`: Model identifier for the language model

   Example:
   ```bash
   export LLM_API_KEY="your-api-key"
   export LLM_API_BASE="https://api.openai.com/v1"
   export LLM_MODEL="gpt-4"
   ```

6. Setup secrets (Required for tool use):
   
   Secrets are used to communicate with external APIs. Follow the format in `agent/secrets_example` and change the name of the directory to `agent/secrets`.
   
   You will need to:
   - Set up a Google Cloud Platform account
   - Give necessary permissions
   - Get the credential.json file
   - Get a SerpAPI key for web search

## Running the Application

Launch the Gradio app:
```bash
python ui/local_speech_app.py
```

This will start a local web server that you can access in your browser.

## Troubleshooting

If you encounter any issues during installation, check the following:

1. Make sure all environment variables are set correctly
2. Ensure that the conda environment is activated
3. Check that all dependencies are installed correctly
4. Verify that the secrets directory is set up properly

If you continue to experience issues, please open an issue on the [GitHub repository](https://github.com/Sentientia/Aura/issues).