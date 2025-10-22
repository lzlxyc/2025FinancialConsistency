from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

from src.config import Qwen2_5_7b_Model_Path
model_path = Qwen2_5_7b_Model_Path
lora_path = '../../models/qwen7b_text_comp'



class QwenRuleComparison:
    def __init__(self):
        self.tokenizer, self.model = self._model_init()


    def _model_init(self):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, model_id=lora_path)
        return tokenizer, model


    def chat(self, system: str, prompt: str) -> str:
        messages = [
            {
                "role": "system", "content": system},
            {
                "role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = self.tokenizer([text], return_tensors="pt").to('cuda')

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=32768
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
