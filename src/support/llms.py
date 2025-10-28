from openai import AsyncOpenAI, OpenAI
from modelscope import (
    AutoModelForCausalLM,
    AutoTokenizer,
    snapshot_download,
    GenerationConfig
)
import logging

from src.config import Qwen2_5_7b_Model_Path


class AiBox:
    '''
    model: ds: depseek; qw: qwen2.5-7b; qw72: qwen2.5b-72
    mode: api, local
    '''
    def __init__(self, mode='api', model='ds',api_key=None, api_url=None):
        logger = logging.getLogger('_client')
        logger.setLevel(logging.WARNING)

        self.mode = mode
        if mode == 'api':
            if api_url is not None and isinstance(api_key, str):
                self.model = model
                self.api_key = api_key
                BASE_URL = api_url
            elif model == 'ds':
                self.model = "deepseek-chat"
                self.api_key = api_key['ds']
                BASE_URL = "https://api.deepseek.com"
            elif model == 'qw32':
                self.model = "qwen2.5-32b-instruct"
                self.api_key = api_key['qwen']
                BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            else:
                self.model = "qwen2.5-72b-instruct"
                self.api_key = api_key['qwen']
                BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            
            print(f"{mode=} || {self.model=} {self.api_key}")

            self.client = OpenAI(api_key=self.api_key, base_url=BASE_URL)
        else:
            self.init_local_model()
            print(f"{mode=} || {model=}")

            
    def init_local_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            Qwen2_5_7b_Model_Path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(Qwen2_5_7b_Model_Path)

        # model_dir = snapshot_download('TongyiFinance/Tongyi-Finance-14B-Chat-Int4')
        # self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:0", trust_remote_code=True).eval()
        # self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        # self.model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)


    def message_make(self, prompt:str, system=None):
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
            
        messages.append({"role": "user", "content": prompt})

        return messages

            
    def local_chat(self, prompt:str, system=None) -> str:
        response, history = self.model.chat(self.tokenizer, query=prompt, history=None,system=system)
        return response


    def chat(self, prompt:str='你好', system=None):
        '''
        对外统一接口
        '''
        if self.mode == 'api':
            messages = self.message_make(prompt, system)

            response = self.client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=0.1,# top_p=0.7 ,
                extra_body={"enable_thinking": False},
                top_p = 0.1,
                seed = 42,
                presence_penalty=-2

            ).choices[0].message.content
        else:
            response = self.local_chat(prompt, system)
        
        return response
    #
    # def chat2(self, prompt:str='你好', system=None):
    #     try:
    #         response = self.__chat(prompt, system)
    #     except Exception as e:
    #         response = None
    #         print(e)
    #
    #     return response

if __name__ == "__main__":
    d = {
        'model': "deepseek-chat",
        'api_url': "https://api.deepseek.com",
        'api_key': "sk-c11b4bd9dadc4e41ad6ae6dccdbbfd6e"
    }
    a = AiBox(**d)
    print(a.model)



        