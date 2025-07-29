"""
An integrated script to perform log analysis using a Gemma model on Vertex AI.

This script uses the proven method of passing all parameters within the instances
object to ensure correct behavior and prevent truncated responses. This version
is updated to include the detailed Internal Knowledge Base in the final prompt.
"""
import os
from google.cloud import aiplatform, exceptions
import concurrent.futures
import logging
import time # Import the time module for performance measurement

# --- Configuration ---
# TODO: Replace with your actual project details and Gemma endpoint ID.
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "xxxx")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "yyyy")
ENDPOINT_ID = "zzz"  # IMPORTANT: This must be the ID for your deployed Gemma model.

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LogAnalyzerGemma:
    def __init__(self, project_id: str, location: str, endpoint_id: str):
        """Initializes the analyzer with Vertex AI endpoint details."""
        if not all([project_id, location, endpoint_id]):
            raise ValueError("Project ID, Location, and Endpoint ID are required.")
        
        logger.info("Initializing Vertex AI client...")
        aiplatform.init(project=project_id, location=location)
        self.endpoint = aiplatform.Endpoint(
            endpoint_name=f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"
        )
        logger.info("Vertex AI client and endpoint initialized.")

    def _call_gemma_endpoint(self, prompt_content: str, call_type: str = "Analysis"):
        """
        Sends a formatted prompt to the Gemma Vertex AI endpoint.
        """
        # All parameters are packed into the instance dictionary along with the prompt.
        # This is the required format for this specific model endpoint.
        instances = [
            {
                "prompt": f"<start_of_turn>user\n{prompt_content}<end_of_turn>\n<start_of_turn>model",
                "max_tokens": 8192,
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 40,
                "raw_response": True, # This parameter is specific to some serving containers
            }
        ]
        
        try:
            # We no longer pass a separate 'parameters' dictionary.
            prediction_response = self.endpoint.predict(instances=instances)
            if prediction_response.predictions:
                raw_response = prediction_response.predictions[0]
                
                # Print the raw, unaltered output from the model with a descriptive header.
                print(f"\n--- Raw Model Output ({call_type}) ---")
                print(raw_response)
                
                return raw_response
            return None
        except Exception as e:
            logger.error(f"Failed to get prediction from endpoint: {e}")
            return None

    def _split_into_chunks(self, text, chunk_size=25000):
        """Splits text into chunks suitable for a model with a 32k context window."""
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def analyze_logs(self, log_file_path, error_message):
        """Performs a two-stage analysis (summarize, then final RCA)."""
        logger.info(f"Starting analysis of '{log_file_path}'...")
        
        # Start the timer for the entire analysis process
        start_time = time.monotonic()

        try:
            with open(log_file_path, 'r') as f:
                log_data = f.read()
        except FileNotFoundError:
            logger.error(f"Log file not found: {log_file_path}")
            return "Error: Log file not found.", 0

        log_chunks = self._split_into_chunks(log_data)
        logger.info(f"Log file split into {len(log_chunks)} chunks.")

        def process_chunk(chunk_info):
            chunk, i = chunk_info
            logger.info(f"Summarizing chunk {i+1}/{len(log_chunks)}...")
            summary_prompt = f"""
1. You are a Senior System Engineer.
2. Read the following log chunk and summarize key events, errors, and warnings related to the error message: "{error_message}".
3. Ignore unrelated information.
4. Keep your summary concise but detailed.
5. Respond in plain text only.

Log Chunk:
{chunk}
"""
            summary = self._call_gemma_endpoint(summary_prompt, call_type=f"Chunk {i+1} Summary")
            if summary:
                logger.info(f"Successfully summarized chunk {i+1}.")
                return summary
            else:
                logger.error(f"Failed to summarize chunk {i+1}.")
                return None

        # Limit the number of concurrent workers to avoid overwhelming the endpoint
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            chunk_summaries = list(executor.map(process_chunk, zip(log_chunks, range(len(log_chunks)))))

        valid_summaries = [s for s in chunk_summaries if s]
        if not valid_summaries:
            end_time_fail = time.monotonic()
            return "Failed to generate summaries for any log chunks.", (end_time_fail - start_time)

        combined_summary = "\n\n--- End of Chunk Summary ---\n\n".join(valid_summaries)
        
        logger.info("All chunks summarized. Performing final Root Cause Analysis...")
        
        # Restored the detailed prompt with the Internal Knowledge Base
        final_analysis_prompt = f"""
**ROLE:** You are a **Senior System Engineer** at Juspay, specializing in the **UPI TPAP SDK**. You possess an expert-level understanding of the system's architecture, its various services, and the intricate flow of communication between them.

**CONTEXT:** You will be provided with a collection of structured log payloads from a single user session. Your primary objective is to perform a deep and insightful **Root Cause Analysis (RCA)**.

---

### **INTERNAL KNOWLEDGE BASE (For Your Reference Only)**

#### **System Flow Overview**
*(This is for your internal understanding. **DO NOT** mention "stages" in your final analysis.)*

The system's architecture is modular and supports a variety of actions. Your analysis must be contextualized by the specific action and the merchant's integration mode.

1.  **Stage 1: SDK Initialization**
    * **Action:** The Merchant App initializes the main Juspay SDK.
    * **Details:** This is the mandatory first step for all operations, requiring a signature and payload from the Merchant Server.

2.  **Stage 2: Service Routing (`hyperapi` or `ec`)**
    * **Action:** The SDK connects to a routing service.
    * **Details:** This service (`hyperapi` or `ec`) acts as a gateway, directing requests to the correct downstream UPI service based on the merchant's configuration.

3.  **Stage 3: Action Execution (Context-Dependent)**
    * **Action:** The routing service directs the request to perform a specific UPI action.
    * **Key Actions Include (but are not limited to):**
        * **Onboarding & Setup:** `upiCheckPermission`, `Get Session Token`, the full `UPI Onboarding` flow.
        * **Core Transactions:** `UPI Transaction` for payments.
        * **Account Management:** `Check Balance`, `Change/Set MPIN`.
        * **Mandates:** Creating/managing recurring payments.
        * **External Triggers:** Handling `Incoming UPI Intent` or `Approving UPI Collect`.
    * ***Note:*** *This list is not exhaustive. If you encounter an unlisted action, analyze it based on the general system flow.*

4.  **Stage 4: UPI Service Interaction (Mode-Dependent)**
    * **Action:** The specific UPI action is executed by one of two services.
    * **Modes:**
        * **UI-Driven Mode:** The request goes to **`hyperupi`** (UI management), which then internally calls **`inapp-upi`** (backend API calls). A failure here could be in the UI layer or the API layer.
        * **Headless (API-Only) Mode:** The request goes directly to **`inapp-upi`** for backend processing.

#### **Juspay Error Code Reference**
* **JP_000:** Reason Unavailable.
* **JP_001:** This is mainly caused due to an incorrect business logic.
* **JP_002:** This error code is received when a user backpressed.
* **JP_003:** This is an Integration error caused due to type mismatch in parameters.
* **JP_004:** User based errors.
* **JP_005:** User is not connected to the internet.
* **JP_006:** Delay in updation of transaction status. Awaiting response from PG.
* **JP_007:** Unable to redirect to a valid URL to proceed with transaction.
* **JP_008:** Mandatory configurations on Juspay dashboard are incorrect/incomplete.
* **JP_009:** This is an error specific to Native OTP flow, where the user had exceeded the limit of incorrect OTP submissions.
* **JP_010:** Feature is not supported.
* **JP_011:** Server error.
* **JP_012:** Transaction failure at PG end.
* **JP_014:** Minimised before launching cct activity.
* **JP_015:** Transaction can not go through the UPI application.
* **JP_016:** Juspay Safe Mode could not rescue the transaction.
* **JP_017:** Initiate was called multiple times on an instance before terminating the SDK.
* **JP_018:** Required permissions to run SDK does not exists.
* **JP_019:** When NPCI CL does not return the challenge or throws a technical error.
* ***Note:*** *If you encounter an error code not on this list, use the error message and surrounding log context to deduce its meaning.*

---

### **YOUR TASK**

**Primary Error to Analyze:** **"{error_message}"**

**Instructions:**
Based on the provided log summaries, perform a root cause analysis. Follow these steps meticulously:

1.  **Identify the Context:** First, determine the **action** being performed (e.g., Onboarding, Payment) and the **operational mode** (UI-Driven or Headless) by examining the service call sequence (`sdk` -> `hyperapi`/`ec` -> `hyperupi`/`inapp-upi`).
2.  **Pinpoint the Root Cause:** Using the system flow and error code reference, analyze the sequence of events in the logs to identify the **fundamental reason** for the failure. Be precise and use specific details from the logs.
3.  **Explain the "Why":** Your analysis must go beyond *what* happened and explain *why* it happened. Connect the log events to the expected system behavior and the error code definitions.
4.  **Handle Missing Error Messages:** If the `Primary Error to Analyze` is not explicitly found, **do not** state that it's missing. Instead, analyze the entire session to deduce the most probable root cause from the available context and event sequence.
5.  **Be Factual and Direct:** Present your findings as a clear, concise, and highly detailed paragraph. Respond in plain text.

**Combined Log Summaries:**
{combined_summary}
"""
        final_rca_raw = self._call_gemma_endpoint(final_analysis_prompt, call_type="Final RCA")
        
        # Stop the timer
        end_time = time.monotonic()
        total_time = end_time - start_time
        
        # Since this method returns only the summary without the prompt, we can use it directly.
        if final_rca_raw:
            logger.info("Final RCA generated successfully.")
            return final_rca_raw, total_time
        else:
            return "Failed to generate the final Root Cause Analysis.", total_time


if __name__ == "__main__":
    if "your-gemma-endpoint-id" in ENDPOINT_ID:
        print("Please update your project details (PROJECT_ID, LOCATION, ENDPOINT_ID) at the top of this script.")
    else:
        log_file = "sample.log"
        error_to_analyze = 'RegexValidation "customerVpa regex failed"'

        # Create a large sample log file if it doesn't exist for testing
        if not os.path.exists(log_file):
            print(f"'{log_file}' not found. Creating a large sample file for testing.")
            with open(log_file, "w") as f:
                base_log = '{"timestamp": "2025--29T10:00:02Z", "level": "ERROR", "service": "inappupi", "action": "upiValidVpa", "status": "FAILURE", "errorDescription": "RegexValidation \\"customerVpa regex failed\\"", "customerVpa": "000@000"}\n'
                f.write(base_log * 1000) # Reduced size for 32k context model

        # Initialize the analyzer and run the analysis
        analyzer = LogAnalyzerGemma(project_id=PROJECT_ID, location=LOCATION, endpoint_id=ENDPOINT_ID)
        result, analysis_time = analyzer.analyze_logs(log_file, error_to_analyze)
        
        print("\n--- FINAL ROOT CAUSE ANALYSIS ---")
        print(result)
        print(f"\n--- Total analysis time: {analysis_time:.2f} seconds ---")


