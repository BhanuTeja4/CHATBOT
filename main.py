import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import warnings

# Suppress the padding warning messages
warnings.filterwarnings("ignore", category=UserWarning)

model_name = "microsoft/DialoGPT-medium"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name)

st.title("Chatbot")

chat_history_ids = None

while True:
    user_input = st.text_input("You:", "")
    
    if user_input:
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

        bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if chat_history_ids is not None else input_ids

        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
        )

        bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        st.text("BOT:")

        if bot_response:
            st.text(bot_response)
