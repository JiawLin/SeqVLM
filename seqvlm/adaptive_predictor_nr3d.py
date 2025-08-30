import glob
import json
import os
import random
import pickle
from objprint import op
from colorama import Fore, init
init(autoreset=True)

from prompts.prompt import *
from utils import *
from api import invoke_api
from visprog.program import ProgramInterpreter
from seqvlm.feat_handler_nr3d import VisualFeatHandler



with open('../data/feats_3d.pkl', 'rb') as f:
    feats = pickle.load(f)
def load_pc(scan_id):
    obj_ids = feats[scan_id]['obj_ids']
    inst_locs = feats[scan_id]['inst_locs']
    center = feats[scan_id]['center']
    obj_embeds = feats[scan_id]['obj_embeds']

    return obj_ids, inst_locs, center, obj_embeds


class AdpativePredictor:
    
    def __init__(self, **kwargs):        
        self.image_path = kwargs.get('image_path')
        self.max_retry = kwargs.get('max_retry')
        self.vlm_model = kwargs.get('vlm_model')
        self.max_batch_size = kwargs.get('max_batch_size')
        self.max_vlm_props = kwargs.get('max_vlm_props')
        
        self.prog_inter = ProgramInterpreter()
        self.handler = VisualFeatHandler.get_instance()

    
    def execute(self, scene_id, obj_name, caption, prog_str):
        # ins_labels, ins_locs, ins_scores, _ = load_seg_inst(scene_id)
        obj_ids, ins_locs, center, obj_embeds = load_pc(scene_id)
        # pred_cls = self.handler.predict_obj_class(obj_name, ins_labels)
        pred_cls, pred_class_list = self.handler.predict_obj_class(obj_name, obj_embeds)
        index = []
        prop_images = []
        # seg_conf_score = 0.2
        
        for i in range(len(obj_ids)):
            if pred_class_list[i] == pred_cls:
                obj_id = obj_ids[i]
                canvas = os.path.join(self.image_path, scene_id, str(obj_id), 'canvas.jpg')
                if os.path.exists(canvas):
                    index.append(i)
                    prop_images.append(canvas)
    
        n_props = len(prop_images)
        print('Prop Images:', n_props, prop_images[:20])
        
        if n_props <= self.max_vlm_props:
            pred = self.predict(prop_images, caption)
            if pred is not None:
                return ins_locs[index[pred]], True

        try:
            init_state = {'scene_id': scene_id}
            result = self.prog_inter.execute(prog_str, init_state)
            if result:
                return result[0]['obj_loc'], False
        except Exception as e:
            print(Fore.RED + f'Except: {e}')
            
        return None, False

    
    def predict(self, prop_images, caption):
        max_batch_size = self.max_batch_size
        base64Images = [encode_image_to_base64(image) for image in prop_images]
        
        remain_props = [(i, image) for i, image in enumerate(base64Images)]
        while len(remain_props) > 1:        
            prop_image_groups = []
            for i in range(0, len(remain_props), max_batch_size):
                g = remain_props[i:i + max_batch_size]
                prop_image_groups.append(g)
            
            remain_props = []
            for i, images in enumerate(prop_image_groups):
                image_id = self.select_with_retry(images, caption)
                if image_id != -1:
                    remain_props.append(images[image_id])
                    
        return remain_props[0][0] if remain_props else None
    
    
    def select_with_retry(self, images, caption):
        n_images = len(images)
        assert n_images > 0
        # return random.randint(-1, n_images - 1)
        
        user_prompt = USER_PROMPT.format(
            query=caption, 
            n_images=n_images
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    *map(
                        lambda x: {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{x[1]}",
                                "detail": "high",
                            },
                        },
                        images,
                    ),
                ]
            }
        ]
        
        retry = 0
        while retry < self.max_retry:
            try:
                output = invoke_api(self.vlm_model, messages)
                print('Vision Lang Model Output:')
                op(output)
                answer = json.loads(output['answer'])
                image_id = answer['image_id']
                if isinstance(image_id, int) and -1 <= image_id < n_images:
                    return image_id
                
                guide_prompt = IMAGE_ID_INVALID_PROMPT.format(image_id=image_id)
            except Exception as e:
                print(Fore.RED + f'Except: {e}')
                guide_prompt = WRONG_FORMAT_PROMPT
                
            vlm_message = {
                'role': 'assistant', 
                'content': output['answer']
            }
            messages.append(vlm_message)
            messages.append(
                {"role": "user", "content": guide_prompt}
            )
            retry += 1
    
        return -1