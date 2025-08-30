import io
import os
import yaml
from openai import OpenAI
from mmengine.utils.dl_utils import TimeCounter


with open("../config.yaml", "r") as f:
    configs = yaml.safe_load(f)


def invoke_api(model, messages):
    base_url, model, api_key = configs[model].values()
    client = OpenAI(
        api_key=api_key, 
        base_url=base_url
    )
    with TimeCounter(tag="Invoke"):
        response = client.chat.completions.create(
            model=model, 
            messages=messages
        )
        result = {
            'answer': response.choices[0].message.content, 
            'prompt_tokens': response.usage.prompt_tokens, 
            'completion_tokens': response.usage.completion_tokens
        }
        return result
    
    
if __name__ == '__main__':
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Who are you?'},
            ],
        }
    ]
    print(invoke_api('gpt-proxy', messages))