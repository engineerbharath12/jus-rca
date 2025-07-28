# AI-Powered Log Analyzer

This project analyzes log files using a generative AI model to provide a root cause analysis for errors. It supports two providers: Juspay Neurolink and Google Vertex AI.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set Environment Variables:**
    Create a `.env` file by copying the `env.example` file:
    ```bash
    cp env.example .env
    ```
    Then, fill in the required values in the `.env` file.

## Usage

1.  **Place your log file** in the root of the project directory. For example, `sample.log`.

2.  **Run the script:**
    ```bash
    python main.py
    ```
    You will be prompted to choose between two services:
    1.  **Juspay Neurolink:** Uses the `@juspay/neurolink` CLI to analyze logs.
    2.  **Vertex AI:** Uses Google's Vertex AI for analysis.

    The script will use `sample.log` by default. To use a different log file, you will need to modify the `main.py` script.
