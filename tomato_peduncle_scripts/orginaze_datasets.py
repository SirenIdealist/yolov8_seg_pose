import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def convert_coco_to_yolo(json_file, save_dir, use_segments=True, use_keypoints=True):
    with open(json_file, 'r') as f:
        data = json.load(f)

    images = {x['id']: x for x in data['images']}
    imgToAnns = defaultdict(list)
    for ann in data['annotations']:
        imgToAnns[ann['image_id']].append(ann)

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for img_id, anns in tqdm(imgToAnns.items(), desc='Converting'):
        img = images[img_id]
        h, w, f = img['height'], img['width'], img['file_name']

        bboxes = []
        segments = []
        keypoints = []
        for ann in anns:
            if ann.get('iscrowd', False):
                continue
            box = np.array(ann['bbox'], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w
            box[[1, 3]] /= h
            if box[2] <= 0 or box[3] <= 0:
                continue
            cls = ann['category_id'] - 1  # 0-based
            box = [cls] + box.tolist()
            if box not in bboxes:
                bboxes.append(box)
            # segmentation
            if use_segments and ann.get('segmentation') is not None:
                segs = ann['segmentation']
                if len(segs) == 0:
                    segments.append([])
                    continue
                s = [j for i in segs for j in i]
                s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                s = [cls] + s
                if s not in segments:
                    segments.append(s)
            # keypoints
            if use_keypoints and ann.get('keypoints') is not None:
                k = (np.array(ann['keypoints']).reshape(-1, 3) / np.array([w, h, 1])).reshape(-1).tolist()
                k = box + k
                keypoints.append(k)

        # 写入标签文件
        label_path = os.path.join(save_dir, f[:-4] + '.txt')
        with open(label_path, 'a') as file:
            for i in range(len(bboxes)):
                if use_keypoints and use_segments:
                    line_kpt = keypoints[i]
                    line_seg = segments[i] if use_segments and len(segments[i]) > 0 else bboxes[i]
                    line = line_kpt + line_seg[1:]  # 去掉重复的class
                elif use_keypoints:
                    line = keypoints[i]
                else:
                    line = segments[i] if use_segments and len(segments[i]) > 0 else bboxes[i]
                file.write(' '.join([str(round(x, 6)) for x in line]) + '\n')

if __name__ == '__main__':
    json_file = '/home/ubuntu/repos/yolov8_segment_pose/data/tomato_peduncle/annotations/peduncle_val_145.json'
    save_dir = '/home/ubuntu/repos/yolov8_segment_pose/data/tomato_peduncle/labels/val2017'
    convert_coco_to_yolo(json_file, save_dir)