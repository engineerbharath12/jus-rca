import os
import json
import logging
import subprocess
import shlex
from dotenv import load_dotenv
import concurrent.futures

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogAnalyzer:
    def __init__(self, google_api_key, hugging_face_api_key):
        self.google_api_key = google_api_key
        self.hugging_face_api_key = hugging_face_api_key

    def _call_ai_provider(self, prompt):
        command = f"npx @juspay/neurolink generate {shlex.quote(prompt)} --provider google-ai --model gemini-2.5-pro --timeout 60s"
        
        env = os.environ.copy()
        env["HUGGINGFACE_API_KEY"] = self.hugging_face_api_key
        env["GOOGLE_AI_API_KEY"] = self.google_api_key

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                shell=True
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                logger.error(f"neurolink CLI failed with error: {stderr}")
                return {"summary": "AI analysis failed"}

            return {"summary": stdout.strip()}
        except FileNotFoundError:
            logger.error("npx command not found. Make sure Node.js and npm are installed and in your PATH.")
            return {"summary": "AI analysis failed"}
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return {"summary": "AI analysis failed"}

    def _read_log_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Log file not found at: {file_path}")
            return None

    def _split_into_chunks(self, text, chunk_size=4096):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def analyze_logs(self, log_file_path, error_message):
        log_data = self._read_log_file(log_file_path)
        if not log_data:
            return {"summary": "Failed to read log file."}

        log_chunks = self._split_into_chunks(log_data)
        
        def process_chunk(chunk_info):
            chunk, i = chunk_info
            logger.info(f"Processing chunk {i+1}/{len(log_chunks)}")
            summary_prompt = f"""
1.You are a Senior System Engineer with deep expertise in analyzing logs from distributed systems. You have comprehensive knowledge of UPI system architecture and are well-versed in diagnosing common issues that occur within such ecosystems.
2.Read the following log Chunk carefully and summarize the key events, errors, warnings, and any contextual information specifically related to the error message: "{error_message}".
3.Ignore unrelated information or errors not matching this error message.
4.Keep your summary concise but detailed enough to show important patterns. 

Log Chunk:
{chunk}
"""
            summary = self._call_ai_provider(summary_prompt)
            if "AI analysis failed" in summary.get("summary", ""):
                logger.error(f"Failed to get summary for chunk {i+1}")
                return None
            return summary["summary"]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            chunk_summaries = list(executor.map(process_chunk, zip(log_chunks, range(len(log_chunks)))))

        # Filter out failed chunks
        chunk_summaries = [s for s in chunk_summaries if s is not None]

        if not chunk_summaries:
            return {"summary": "Failed to generate summaries for any log chunks."}

        combined_summary = "\n\n".join(chunk_summaries)
        
        FINAL_ANALYSIS_PROMPT = f"""
**ROLE:** You are a **Senior System Engineer** at Juspay, specializing in the **UPI TPAP SDK**. You possess an expert-level understanding of the system's architecture, its various services, and the intricate flow of communication between them.

**CONTEXT:** You will be provided with a collection of structured log payloads from a single user session. Your primary objective is to perform a deep and insightful **Root Cause Analysis (RCA)**.

---

### **INTERNAL KNOWLEDGE BASE (For Your Reference Only)**

#### **System Flow Overview**
*(This is for your internal understanding. **DO NOT** mention "stages" in your final analysis.)*

The system's architecture is modular and supports a variety of actions. Your analysis must be contextualized by the specific action and the merchant's integration mode.

1.  **Stage 1: SDK Initialization**
    *   **Action:** The Merchant App initializes the main Juspay SDK.
    *   **Details:** This is the mandatory first step for all operations, requiring a signature and payload from the Merchant Server.

2.  **Stage 2: Service Routing (`hyperapi` or `ec`)**
    *   **Action:** The SDK connects to a routing service.
    *   **Details:** This service (`hyperapi` or `ec`) acts as a gateway, directing requests to the correct downstream UPI service based on the merchant's configuration.

3.  **Stage 3: Action Execution (Context-Dependent)**
    *   **Action:** The routing service directs the request to perform a specific UPI action.
    *   **Key Actions Include (but are not limited to):**
        *   **Onboarding & Setup:** `upiCheckPermission`, `Get Session Token`, the full `UPI Onboarding` flow.
        *   **Core Transactions:** `UPI Transaction` for payments.
        *   **Account Management:** `Check Balance`, `Change/Set MPIN`.
        *   **Mandates:** Creating/managing recurring payments.
        *   **External Triggers:** Handling `Incoming UPI Intent` or `Approving UPI Collect`.
    *   ***Note:*** *This list is not exhaustive. If you encounter an unlisted action, analyze it based on the general system flow.*

4.  **Stage 4: UPI Service Interaction (Mode-Dependent)**
    *   **Action:** The specific UPI action is executed by one of two services.
    *   **Modes:**
        *   **UI-Driven Mode:** The request goes to **`hyperupi`** (UI management), which then internally calls **`inapp-upi`** (backend API calls). A failure here could be in the UI layer or the API layer.
        *   **Headless (API-Only) Mode:** The request goes directly to **`inapp-upi`** for backend processing.

#### **Juspay Error Code Reference**
*   **JP_000:** Reason Unavailable.
*   **JP_001:** This is mainly caused due to an incorrect business logic.
*   **JP_002:** This error code is received when a user backpressed.
*   **JP_003:** This is an Integration error caused due to type mismatch in parameters.
*   **JP_004:** User based errors.
*   **JP_005:** User is not connected to the internet.
*   **JP_006:** Delay in updation of transaction status. Awaiting response from PG.
*   **JP_007:** Unable to redirect to a valid URL to proceed with transaction.
*   **JP_008:** Mandatory configurations on Juspay dashboard are incorrect/incomplete.
*   **JP_009:** This is an error specific to Native OTP flow, where the user had exceeded the limit of incorrect OTP submissions.
*   **JP_010:** Feature is not supported.
*   **JP_011:** Server error.
*   **JP_012:** Transaction failure at PG end.
*   **JP_014:** Minimised before launching cct activity.
*   **JP_015:** Transaction can not go through the UPI application.
*   **JP_016:** Juspay Safe Mode could not rescue the transaction.
*   **JP_017:** Initiate was called multiple times on an instance before terminating the SDK.
*   **JP_018:** Required permissions to run SDK does not exists.
*   **JP_019:** When NPCI CL does not return the challenge or throws a technical error.
*   ***Note:*** *If you encounter an error code not on this list, use the error message and surrounding log context to deduce its meaning.*

---

### **YOUR TASK**

**Primary Error to Analyze:** **"{error_message}"**

**Instructions:**
Based on the provided log summaries, perform a root cause analysis. Follow these steps meticulously:

1.  **Identify the Context:** First, determine the **action** being performed (e.g., Onboarding, Payment) and the **operational mode** (UI-Driven or Headless) by examining the service call sequence (`sdk` -> `hyperapi`/`ec` -> `hyperupi`/`inapp-upi`).
2.  **Pinpoint the Root Cause:** Using the system flow and error code reference, analyze the sequence of events in the logs to identify the **fundamental reason** for the failure. Be precise and use specific details from the logs.
3.  **Explain the "Why":** Your analysis must go beyond *what* happened and explain *why* it happened. Connect the log events to the expected system behavior and the error code definitions.
    *   *Example:* "The UPI Transaction failed because the signature payload from the Merchant Server was invalid, as indicated by the 'Signature Validation Failed' error in the Juspay SDK logs."
4.  **Handle Missing Error Messages:** If the `Primary Error to Analyze` is not explicitly found, **do not** state that it's missing. Instead, analyze the entire session to deduce the most probable root cause from the available context and event sequence.
5.  **Be Factual and Direct:** Present your findings as a clear, concise, and highly detailed paragraph. Avoid ambiguity and generic statements.

**Output Format:**
Respond with a **single JSON object** containing one key: `"root_cause"`.

**Combined Log Summaries:**
{combined_summary}
"""
        final_summary = self._call_ai_provider(FINAL_ANALYSIS_PROMPT)
        return final_summary

if __name__ == "__main__":
    hugging_face_api_key = os.getenv("HUGGINGFACE_API_KEY")
    google_api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not hugging_face_api_key:
        logger.error("HUGGINGFACE_API_KEY environment variable not set.")
    else:
        analyzer = LogAnalyzer(google_api_key, hugging_face_api_key)
        result = analyzer.analyze_logs("sample.log", """RegexValidation \"customerVpa regex failed\"""")
        print(json.dumps(result, indent=2))