import glob
import os
from PIL import Image

from seqvlm.feat_handler import VisualFeatHandler
from seqvlm.utils import load_seg_inst
from visprog.program import parse_step
from visprog.registry import register_interpreter


@register_interpreter
class LocInterpreter():
    step_name = 'LOC_BLIP'

    def __init__(self):
        self.image_path = '../data/2d_bbox_pred'
        self.handler = VisualFeatHandler.get_instance()


    def execute(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        output_var = parse_result['output_var']

        obj_name = eval(parse_result['args']['object'])
        scene_id = prog_step.state['scene_id']

        ins_labels, ins_locs, ins_scores, center = load_seg_inst(scene_id)
        boxes = self.predict(ins_labels, ins_locs, ins_scores, obj_name, scene_id)
        prog_step.state[output_var] = boxes
        prog_step.state['CENTER'] = [{'obj_id': -1, 'obj_loc': center, 'obj_name': 'CENTER'}]
        return boxes


    def predict(self, ins_labels, ins_locs, ins_scores, obj_name, scene_id):
        pred_cls = self.handler.predict_obj_class(obj_name, ins_labels)
        
        seg_conf_score = 0.2
        prop_insts = []
        for i, score in enumerate(ins_scores):
            if score > seg_conf_score and ins_labels[i] == pred_cls:
                prop_insts.append(i)
        
        ratio = min(len(prop_insts) / 8, 0.5)
        
        boxes = []
        candidate_boxes = []
        for i in prop_insts:
            obj_id = i
            img_file = glob.glob(os.path.join(self.image_path, scene_id, str(obj_id), 'img_*.jpg'))
            
            if len(img_file) > 0:
                images = [Image.open(f) for f in img_file]                
                if self.handler.judge_consistency(obj_name, images, ratio):
                    boxes.append({'obj_id': obj_id, 'obj_name': pred_cls, 'obj_loc': ins_locs[i]})
            else:
                boxes.append({'obj_id': obj_id, 'obj_name': pred_cls, 'obj_loc': ins_locs[i]})
                        
            candidate_boxes.append({'obj_id': obj_id, 'obj_name': pred_cls, 'obj_loc': ins_locs[i]})
            
        if len(boxes) == 0:
            boxes = candidate_boxes

        return boxes
