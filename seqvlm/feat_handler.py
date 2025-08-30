
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import (
    AutoTokenizer, 
    CLIPModel,
    Blip2Processor, 
    Blip2ForConditionalGeneration
)

class VisualFeatHandler:

    __instance = None
    
    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance
    
    def __init__(self):
        model_path = "../data/huggingface/blip2-flan-t5-xl"
        tokenizer_path = '../data/huggingface/clip-vit-base-patch16'
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, 
                                                       model_max_length=512, 
                                                       use_fast=True, 
                                                       clean_up_tokenization_spaces=True)
        self.clip = CLIPModel.from_pretrained(tokenizer_path).cuda()
        self.processor = Blip2Processor.from_pretrained(model_path, clean_up_tokenization_spaces=True)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
        
    
    def predict_obj_class(self, obj_name, ins_labels) -> str:
        class_list = list(set(ins_labels))
        class_tokens = self.tokenizer([f'a {class_name} in a scene' for class_name in class_list],
                                      padding=True,
                                      return_tensors='pt')
        for name in class_tokens.data:
            class_tokens.data[name] = class_tokens.data[name].cuda()
        
        label_feats = self.clip.get_text_features(**class_tokens)
        label_feats = label_feats / label_feats.norm(p=2, dim=-1, keepdim=True)
        
        query_tokens = self.tokenizer([f'a {obj_name} in a scene'], padding=True, return_tensors='pt')
        for name in query_tokens.data:
            query_tokens.data[name] = query_tokens.data[name].cuda()

        query_feats = self.clip.get_text_features(**query_tokens)
        query_feats = query_feats / query_feats.norm(p=2, dim=-1, keepdim=True)

        pred_scores = torch.matmul(query_feats, label_feats.t())
        pred_cls_idx = pred_scores.argmax(dim=-1)[0]
        pred_cls = class_list[pred_cls_idx]
        return pred_cls
    
    
    def judge_consistency(self, obj_name, images, ratio=0.25) -> bool:
        if len(images) == 0:
            return False
        prompt = [f"Question: Is there a {obj_name}? Answer:"] * len(images)
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to('cuda', torch.float16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return answer.count('yes') / len(answer) >= ratio
        

if __name__ == '__main__':
    pass