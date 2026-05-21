from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class ConversationalAgent:
    def __init__(self):
        print("Loading local conversational model (DialoGPT-small)...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        self.chat_history_ids = None

    MAX_INPUT_TOKENS = 200

    def generate_response(self, user_input):
        if not user_input or not user_input.strip():
            return "I didn't catch that. Could you please repeat?"

        if len(user_input) > 1000:
            return "That's quite long! Could you keep it shorter?"

        input_ids = self.tokenizer.encode(user_input, return_tensors="pt")
        if input_ids.shape[1] > self.MAX_INPUT_TOKENS:
            return "I can only process about 200 words at a time. Please say that in fewer words."

        new_user_input_ids = self.tokenizer.encode(
            user_input + self.tokenizer.eos_token, return_tensors="pt"
        )

        if self.chat_history_ids is not None:
            bot_input_ids = torch.cat(
                [self.chat_history_ids[:, -100:], new_user_input_ids], dim=-1
            )
        else:
            bot_input_ids = new_user_input_ids

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
            temperature=0.7,
        )

        response = self.tokenizer.decode(
            self.chat_history_ids[:, bot_input_ids.shape[-1] :][0],
            skip_special_tokens=True,
        )
        return response
