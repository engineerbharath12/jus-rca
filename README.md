# AI-Powered Log Analyzer

This project analyzes log files using a generative AI model to provide a root cause analysis for errors.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set Environment Variables:**
    Create a `.env` file in the root of the project and add your AI API key:
    ```
    AI_API_KEY="YOUR_API_KEY_HERE"
    ```

## Usage

1.  **Place your log file** in the root of the project directory. For example, `sample.log`.

2.  **Run the script:**
    ```bash
    python main.py
    ```
    The script will use `sample.log` by default. To use a different log file, you will need to modify the `main.py` script.
