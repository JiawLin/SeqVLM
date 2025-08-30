import argparse
import json
import pandas as pd

from utils import *
from seqvlm.adaptive_predictor_nr3d import AdpativePredictor


if __name__ == '__main__':
    # add an argument
    parser = argparse.ArgumentParser(description='seqvlm nr3d')
    parser.add_argument('--data_path', type=str, default='../data/nr3d_250.json')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--vlm_model', type=str, default='doubao-vision')
    parser.add_argument('--max_retry', type=int, default=3)
    parser.add_argument('--max_batch_size', type=int, default=4)
    parser.add_argument('--max_vlm_props', type=int, default=40)
    
    args = parser.parse_args()
    
    with open(args.data_path, 'r') as f:
        eval_data = json.load(f)
    
    # # load label map
    # label_map_file = '../data/scannetv2-labels.combined.tsv'
    # labels_pd = pd.read_csv(label_map_file, sep='\t', header=0)
    
    correct_25 = 0
    correct_50 = 0
    correct_easy = 0
    correct_dep = 0
    easy_total = 0
    dep_total = 0

    total = 0
    vlm_total = 0
    unique_total = 0
    except_total = 0
    
    eps = 10 ** -6
    
    vlm_configs = {
        'image_path': args.image_path, 
        'vlm_model': args.vlm_model, 
        'max_retry': args.max_retry, 
        'max_batch_size': args.max_batch_size, 
        'max_vlm_props': args.max_vlm_props
    }        
        
    predictor = AdpativePredictor(**vlm_configs)
    logger = create_logger(args.exp_name)
    
    for i, task in enumerate(eval_data):
        scene_id, obj_id, caption, prog_str, is_easy, is_dep, obj_name = task.values()
        # print(task)
        
        print('Case:', i)
        print('scene_id:', scene_id)
        print('caption:', caption)
        print('obj_id:', obj_id)
        print('obj_name:', obj_name)
        
        # load point cloud data
        obj_ids, obj_labels, obj_locs = load_pc(scene_id)
        # mapped_class_ids = []

        # for obj_label in obj_labels:
        #     label_ids = labels_pd[labels_pd['raw_category'] == obj_label]['nyu40id']
        #     label_id = int(label_ids.iloc[0]) if len(label_ids) > 0 else 0
        #     mapped_class_ids.append(label_id)

        index = obj_ids.index(int(obj_id))
        target_box = obj_locs[index]
        # target_class_id = mapped_class_ids[index]
                
        total += 1
        # unique = (np.array(mapped_class_ids) == target_class_id).sum() == 1
        if is_easy:
            easy_total += 1
        if is_dep:
            dep_total += 1

        pred_box, use_vlm = predictor.execute(scene_id, obj_name, caption, prog_str)
        
        if use_vlm:
            vlm_total += 1
            
        if pred_box is None:
            except_total += 1
        else:
            iou = calc_iou(pred_box, target_box)
            print(f'IoU: {iou:.2f}')
            
            if iou >= 0.25:
                correct_25 += 1
                if is_easy:
                    correct_easy += 1
                if is_dep:
                    correct_dep += 1
            if iou >= 0.5:
                correct_50 += 1

        accuracy_msgs = [
            'Overall@25: {:.3f}'.format(correct_25 / total), 
            'Overall@50: {:.3f}'.format(correct_50 / total), 
            'Easy: {:.3f}'.format(correct_easy / (easy_total + eps)), 
            'Hard: {:.3f}'.format((correct_25 - correct_easy) / (total - easy_total + eps)), 
            'VD: {:.3f}'.format(correct_dep / (dep_total + eps)), 
            'VID: {:.3f}'.format((correct_25 - correct_dep) / (total - dep_total + eps)), 
            'Easy Ratio: {} / {}'.format(correct_easy, easy_total), 
            'Hard Ratio: {} / {}'.format(correct_25 - correct_easy, total - easy_total), 
            'VD Ratio: {} / {}'.format(correct_dep, dep_total),
            'VID Ratio: {} / {}'.format(correct_25 - correct_dep, total - dep_total),
            'Except Ratio: {} / {}'.format(except_total, total), 
            'VLM Usage Ratio: {} / {}'.format(vlm_total, total), 
            '\n'
        ]
        print('\n'.join(accuracy_msgs))
        
        if (i + 1) % 10 == 0:
            for msg in accuracy_msgs:
                logger.info(msg)

    