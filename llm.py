from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ConversationalAgent:
    def __init__(self):
        print("Loading local conversational model (DialoGPT-small)...")
        # Use DialoGPT-small for lightweight local generation without API keys
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        self.chat_history_ids = None

    def generate_response(self, user_input):
        # Encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')

        # Append the new user input tokens to the chat history
        # We limit the history to the last 100 tokens to prevent repetitive loops
        if self.chat_history_ids is not None:
            bot_input_ids = torch.cat([self.chat_history_ids[:, -100:], new_user_input_ids], dim=-1)
        else:
            bot_input_ids = new_user_input_ids

        # Generate a response
        # Using a fixed attention_mask for open-end generation
        attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)
        
        self.chat_history_ids = self.model.generate(
            bot_input_ids,
            attention_mask=attention_mask,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        # Decode and return the response
        response = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response
