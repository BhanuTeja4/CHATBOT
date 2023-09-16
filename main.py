import streamlit as st
import requests

API_ENDPOINT = "https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"
API_KEY = "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"

def generate_response(messages):
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    inputs = "\n".join(message["content"] for message in messages)

    payload = {
        "model": "facebook/blenderbot-400M-distill",
        "inputs": inputs,
        "parameters": {
            "max_new_tokens": 1024,
            "typical_p": 0.2,
            "repetition_penalty": 1,
            "truncate": 1000,
            "return_full_text": False
        }
    }

    response = requests.post(API_ENDPOINT, json=payload, headers=headers)
    return response.json()

def main():
    st.title("ChatBot")

    user_input = st.text_area("You:", "")
    if st.button("Chat here"):
        messages = [
            {"role": "user", "content": user_input}
        ]
        response = generate_response(messages)
        bot_reply = response[0]['generated_text']
        st.text(f"Chatbot: {bot_reply}")

if __name__ == "__main__":
    main()