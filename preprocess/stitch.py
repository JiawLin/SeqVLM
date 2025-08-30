import os
import cv2
import mmengine
import numpy as np
from functools import partial


def image_match(image_path, frame_path):
    image = cv2.imread(image_path)
    frame = cv2.imread(frame_path)
    
    # print(image_path, frame_path)
    
    h, w = image.shape[:2]
    result = cv2.matchTemplate(frame, image, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 3)
    
    # cv2.imshow('Matched Result', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return frame


def stitch_images(images):
    max_w = max(img.shape[1] for img in images)
    sum_h = sum(img.shape[0] for img in images)
    canvas = np.zeros((sum_h, max_w, 3), dtype=np.uint8)
    
    y_offset = 0
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        canvas[y_offset: y_offset + h, :w] = img
        
        # if i > 0:
        #     cv2.line(canvas, (0, y_offset), (max_w, y_offset), (0, 255, 0), 2)
        
        y_offset += h
    
    return canvas


def process_scene(bbox_dir, frame_dir, save_dir, scene_id):
    f = os.path.join(bbox_dir, scene_id)
    for obj_id in os.listdir(f):
        image_dir = os.path.join(f, obj_id)
        image_list = os.listdir(image_dir)
        
        obj_dir = os.path.join(save_dir, scene_id, obj_id)
        os.makedirs(obj_dir, exist_ok=True)
        
        matched = []
        for image_name in image_list:
            index = int(image_name[4:-4])
            # print(os.path.join(bbox_dir, scene_id, obj_id, image_name))

            frame_path = os.path.join(frame_dir, scene_id, '{:05d}.jpg'.format(index))
            image_path = os.path.join(bbox_dir, scene_id, obj_id, image_name)
            
            result = image_match(image_path, frame_path)
            matched.append(result)
            
            cv2.imwrite(os.path.join(obj_dir, image_name), result)
        
        canvas = stitch_images(matched)
        cv2.imwrite(os.path.join(obj_dir, 'canvas.jpg'), canvas)
    

def main():
    bbox_dir = '../data/crop_images'
    frame_dir = '../data/posed_images_20frame'
    save_dir = '../data/preprocessed_images'
    
    scene_ids = os.listdir(bbox_dir)
    mmengine.track_parallel_progress(
        func=partial(process_scene, bbox_dir, frame_dir, save_dir),
        tasks=scene_ids,
        nproc=6,
    )
    

if __name__ == '__main__':
    main()