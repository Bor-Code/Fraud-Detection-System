import os
import requests

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")

def send_alert(message: str) -> None:
    print(f"ALERT: {message}")
    if SLACK_WEBHOOK_URL:
        try:
            payload = {"text": f"🚨 *Fraud Detection System Alert* 🚨\n{message}"}
            requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
        except Exception as e:
            print(f"Failed to send alert to webhook: {e}")
