import os
import glob

def gen_txt(images_dir, labels_dir, output_txt):
    # 获取所有标签文件名（不带扩展名）
    label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    image_paths = []
    for label_file in label_files:
        base_name = os.path.splitext(os.path.basename(label_file))[0]
        image_path = os.path.join(images_dir, f'{base_name}.jpg')
        if os.path.exists(image_path):
            image_paths.append(f'./{os.path.relpath(image_path)}')
    # 写入 txt 文件
    with open(output_txt, 'w') as f:
        for path in image_paths:
            f.write(f'{path}\n')

if __name__ == '__main__':
    # 修改为你的实际路径
    gen_txt('/home/ubuntu/repos/yolov8_segment_pose/data/tomato_peduncle/images/train2017', '/home/ubuntu/repos/yolov8_segment_pose/data/tomato_peduncle/labels/train2017', '/home/ubuntu/repos/yolov8_segment_pose/data/tomato_peduncle/train2017.txt')
    gen_txt('/home/ubuntu/repos/yolov8_segment_pose/data/tomato_peduncle/images/val2017', '/home/ubuntu/repos/yolov8_segment_pose/data/tomato_peduncle/labels/val2017', '/home/ubuntu/repos/yolov8_segment_pose/data/tomato_peduncle/val2017.txt')