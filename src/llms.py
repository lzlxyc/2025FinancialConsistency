from openai import AsyncOpenAI, OpenAI
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download, GenerationConfig
import logging

logger = logging.getLogger(__name__)


class AiBox:
    '''
    model: ds: depseek; qw: qwen2.5-7b; qw72: qwen2.5b-72
    mode: api, local
    '''
    def __init__(self, mode='api', model='ds'):
        self.mode = mode
        if mode == 'api':
            if model == 'ds':
                self.MODEL = "deepseek-chat"
                API_KEY = "sk-c6cd584debfa4eba866e430fab25252a"
                BASE_URL = "https://api.deepseek.com"
            elif model == 'qw':
                self.MODEL = "qwen2.5-7b"
                API_KEY = "EMPTY"
                BASE_URL = "http://36.103.167.205:1189/v1"
            else:
                self.MODEL = "qwen2.5-72b-instruct"
                API_KEY = "sk-f208bdeb39714ec68c84c524005104ee"
                BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            
            print(f"{mode=} || {self.MODEL=}")
            self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        else:
            self.init_local_model()
            print(f"{mode=} || {model=}")

            
    def init_local_model(self):
        # model_name = 'D:/LZL/workspace/ModelHub/Qwen2.5-0.5B-Instruct'
        model_dir = snapshot_download('TongyiFinance/Tongyi-Finance-14B-Chat-Int4')
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_dir,
        #     torch_dtype="auto",
        #     device_map="auto"
        # )
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:0", trust_remote_code=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)


    def message_make(self, prompt:str, system=None):
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
            
        messages.append({"role": "user", "content": prompt})

        return messages

            
    def local_chat(self, prompt:str, system=None) -> str:
        # messages = self.message_make(prompt, system)
        # print(messages)
        # text = self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        # model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        #
        # generated_ids = self.model.generate(
        #     **model_inputs,
        #     max_new_tokens=8900
        # )
        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        # ]
        #
        # response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response, history = self.model.chat(self.tokenizer, query=prompt, history=None,system=system)


        return response


    def chat(self, prompt:str='你好', system=None):
        '''
        对外统一接口
        '''
        if self.mode == 'api':
            messages = self.message_make(prompt, system)
            response = self.client.chat.completions.create(
                model=self.MODEL, messages=messages,
                temperature=0.2,# top_p=0.7 #,extra_body={"enable_thinking": False}
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
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    aibox = AiBox(mode='local')
    Comparison_System = f"你是一个专业的金融保险行业信息处理专家"
    print(aibox.chat(prompt='你好，你是谁,你的参数有多大',system=Comparison_System))







        