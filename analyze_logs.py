import json
import os
import subprocess
import logging
import re
from typing import Dict, List, Optional, Any
from json import JSONDecoder

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogAnalyzer:
    def __init__(self, logs_file: str, error_message: str, endpoint_url: str):
        self.logs_file = logs_file
        self.error_message = error_message
        self.endpoint_url = endpoint_url
        self.MAX_CHUNK_SIZE = 10000

    def _extract_structured_data(self, log_entry: Dict) -> Optional[Dict]:
        try:
            timestamp = log_entry.get("at", "N/A")  # Use default timestamp if not available
            level = log_entry.get("level")
            payload_str = log_entry.get("value")

            if not all([level, payload_str]):
                return None

            payload = json.loads(payload_str) if isinstance(payload_str, str) else payload_str

            return {
                "timestamp": timestamp,
                "level": level,
                "payload": payload
            }
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Skipping log entry due to parsing error: {log_entry}. Error: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while extracting data from log entry: {log_entry}. Error: {e}")
            return None

    def _parse_ai_json(self, content_str: str) -> Dict[str, Any]:
        logger.debug(f"Raw AI response before parsing:\n{content_str}")

        if '```json' in content_str:
            start = content_str.find('```json') + len('```json')
            end = content_str.find('```', start)
            if end != -1:
                content_str = content_str[start:end].strip()
        elif '```' in content_str:
            start = content_str.find('```') + len('```')
            end = content_str.find('```', start)
            if end != -1:
                content_str = content_str[start:end].strip()

        # logger.debug(f"Cleaned AI response string:\n{content_str}")

        try:
            return json.loads(content_str)
        except Exception:
            pass

        try:
            decoder = JSONDecoder()
            obj, idx = decoder.raw_decode(content_str)
            logger.debug(f"Parsed JSON up to index {idx}")
            return obj
        except Exception as e:
            logger.warning(f"raw_decode failed: {e}")

        start = content_str.find('{')
        end = content_str.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = content_str[start:end+1]
            try:
                return json.loads(candidate)
            except Exception as e:
                logger.warning(f"Heuristic JSON parse failed: {e}")

        m = re.search(r'({.*?})', content_str, re.DOTALL)
        if m:
            raw = m.group(1)
            try:
                return json.loads(raw)
            except Exception as e:
                logger.warning(f"Regex-extracted JSON invalid: {e}")

        return {"summary": content_str, "parse_error": "Unable to extract JSON"}

    async def _call_google_ai_api_with_curl(self, prompt: str, is_json_output: bool = False) -> Dict[str, Any]:
        # logger.debug(f"Sending prompt:\n{prompt}")
        try:
            token = subprocess.check_output("gcloud auth print-access-token", shell=True, text=True).strip()

            instance = {"instances": [{"prompt": prompt}]}
            if is_json_output:
                instance["parameters"] = {
                    "response_mime_type": "application/json",
                    "max_output_tokens": 100000,
                    "temperature": 0.1
                }

            curl_command = [
                'curl', '-X', 'POST',
                '-H', f'Authorization: Bearer {token}',
                '-H', 'Content-Type: application/json',
                self.endpoint_url,
                '-d', json.dumps(instance)
            ]

            logger.info("Executing curl command...")
            process = subprocess.Popen(curl_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                logger.error(f"Curl command failed with exit code {process.returncode}")
                logger.error(f"Stderr: {stderr}")
                return {"summary": f"AI analysis failed: Curl command error. {stderr}"}

            logger.info(f"Raw stdout from curl:\n{stdout}")

            try:
                response_json = json.loads(stdout)
                logger.debug(f"Parsed AI API response:\n{stdout}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode top-level AI API response: {stdout}. Error: {e}")
                return {"summary": f"AI analysis failed: Invalid top-level JSON response. Raw output: {stdout}"}

            prediction = response_json.get("predictions", [{}])
            raw_pred = prediction[0]

            if isinstance(raw_pred, str):
                if '```json' in raw_pred:
                    start = raw_pred.find('```json') + len('```json')
                    end = raw_pred.find('```', start)
                    if end != -1:
                        raw_pred = raw_pred[start:end].strip()
                elif '```' in raw_pred:
                    start = raw_pred.find('```') + len('```')
                    end = raw_pred.find('```', start)
                    if end != -1:
                        raw_pred = raw_pred[start:end].strip()
                content_str = raw_pred
            else:
                content_str = json.dumps(raw_pred)

            if is_json_output:
                return self._parse_ai_json(content_str)
            else:
                return {"summary": content_str}

        except subprocess.CalledProcessError as e:
            logger.error(f"Gcloud auth token command failed: {e}")
            return {"summary": "AI analysis failed: Could not get gcloud auth token."}
        except Exception as e:
            logger.error(f"Unexpected error in _call_google_ai_api_with_curl: {e}")
            return {"summary": f"AI analysis failed: {e}"}

    async def analyze(self) -> Dict[str, Any]:
        logger.info("Starting log analysis...")
        try:
            with open(self.logs_file, 'r') as f:
                logs = json.load(f)
        except FileNotFoundError:
            logger.error(f"Log file not found: {self.logs_file}")
            return {"summary": f"Log file not found: {self.logs_file}"}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in log file: {self.logs_file}")
            return {"summary": f"Invalid JSON in log file: {self.logs_file}"}

        log_entries = logs.get("data", []) if isinstance(logs, dict) else logs
        structured_logs = [self._extract_structured_data(entry) for entry in log_entries]
        structured_logs = [log for log in structured_logs if log]

        if not structured_logs:
            logger.warning("No structured logs extracted.")
            return {"summary": "Could not extract any structured data from the logs."}

        logs_text = json.dumps(structured_logs, indent=2, default=str)
        log_chunks = [logs_text[i:i+self.MAX_CHUNK_SIZE] for i in range(0, len(logs_text), self.MAX_CHUNK_SIZE)]

        chunk_summaries = []
        all_predictions = []
        output_filename = "chunk_predictions.json"

        # Clear the file at the beginning
        with open(output_filename, 'w') as f:
            json.dump([], f)

        for i, chunk in enumerate(log_chunks):
            logger.info(f"Processing chunk {i+1}/{len(log_chunks)}")
            summary_prompt = f"""
                            ONLY analyze the following logs for this error: \"{self.error_message}\".
                            Respond ONLY with a JSON object like: {{
                            "error": {{
                                "count": <number>,
                                "affected_api_tags": [<string>],
                                "details": <string>
                            }}
                            }}
                            Do NOT repeat the prompt or logs. Just return the JSON.

                            Logs:
                            {chunk}
                            """

            summary = await self._call_google_ai_api_with_curl(summary_prompt, is_json_output=True)
            prediction = summary.get("summary", summary.get("raw", ""))
            chunk_summaries.append(prediction)
            all_predictions.append(prediction)

            # Append the current prediction to the file
            with open(output_filename, 'r+') as f:
                file_data = json.load(f)
                file_data.append(prediction)
                f.seek(0)
                json.dump(file_data, f, indent=4)


        combined_summary = json.dumps(chunk_summaries, indent=2)
        logger.info(f"Combined chunk summaries:\n{combined_summary}")

        final_prompt = f"""
ONLY analyze these summaries related to \"{self.error_message}\".
Respond with ONE JSON object. DO NOT repeat the input.

Summaries:
{combined_summary}
"""
        analysis_result = await self._call_google_ai_api_with_curl(final_prompt, is_json_output=True)
        return analysis_result

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()

    LOGS_FILE = "sample.log"
    ERROR_MESSAGE = "Invalid Client Auth Token or signature"
    ENDPOINT_URL = os.getenv("ENDPOINT_URL")

    if not ENDPOINT_URL:
        logger.error("ENDPOINT_URL not set in environment variables.")
    else:
        analyzer = LogAnalyzer(LOGS_FILE, ERROR_MESSAGE, ENDPOINT_URL)
        result = asyncio.run(analyzer.analyze())
        if result:
            print(json.dumps(result, indent=2))
