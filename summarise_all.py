import argparse
import os
import h5py
from summary import summarise  # Import the summarise function from summary.py
# from annotation import auto_label
import os
import shutil  # For copying files
import glob
import json
import pickle
import pandas as pd
from pylabel import importer, exporter
# from annotation.auto_label import yolo_to_coco
from pylabel import importer
from pycocotools.coco import COCO

def yolo_to_coco(yolo_labels_dir, frames_dir):
    yoloclasses = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

    dataset = importer.ImportYoloV5(path=yolo_labels_dir, path_to_images=frames_dir, cat_names=yoloclasses, img_ext="jpg", name="coco128")

    dataset.export.ExportToCoco(cat_id_index=1)
def find_value_in_list_of_dicts(list_of_dicts, key):
    for d in list_of_dicts:
        if key in d:
            return d[key]
    return None  # Key not found in any dictionary

def import_annotations(source_dataset_dir, target_image_dir):

    """
    Imports annotations (txt files and coco.json) from a source dataset directory
    (containing multiple datasets) to a target image directory with a different naming
    convention.

    Args:
    source_dataset_dir (str): Path to the folder containing multiple datasets.
    target_image_dir (str): Path to the directory containing images with a different
                                naming convention.

    Raises:
    ValueError: If source dataset folder or target image directory doesn't exist.
    """

    if not os.path.exists(source_dataset_dir):
        raise ValueError(f"Source dataset folder '{source_dataset_dir}' does not exist.")

    if not os.path.exists(target_image_dir):
        raise ValueError(f"Target image directory '{target_image_dir}' does not exist.")
    frames_dict = {}
    files = [f for f in os.listdir(target_image_dir) if f.endswith('.jpg')]
    print(len(files))

    for file in files:
        key = f"{file.split('_')[0]}_{file.split('_')[1]}"
        frames_dict.setdefault(key, []).append(file.split('_')[2])

    label_path = os.path.join(os.path.dirname(target_image_dir), 'labels')
    os.makedirs(label_path, exist_ok=True)

    for key in frames_dict:
        source_coco_path = os.path.join(source_dataset_dir, key, "labels", "coco128.json")
        if not os.path.exists(source_coco_path):
            raise ValueError(f"Source dataset folder '{source_coco_path}' does not exist.")

        for i in frames_dict[key]:
            source_label_path = os.path.join(source_dataset_dir, key, 'labels', f'{i.split(".")[0]}.txt')
            target_label_path = os.path.join(label_path, f'{key}_{i.split(".")[0]}.txt')
            if os.path.exists(source_label_path):
                print(f"copying from {source_label_path} to {target_label_path}")
                shutil.copy(source_label_path, target_label_path)
            if not os.path.exists(source_label_path):
                print(f"Source coco file '{source_label_path}' does not exist.")
        
    # print(len(os.listdir(label_path)))
    # # 3. Creating coco.json using pylabel from the copied labels
    yolo_to_coco(label_path, target_image_dir)
    # #4. Exporting gender data
    new_gender_data = []
    new_gender_path = os.path.join(os.path.dirname(target_image_dir),'labels','gender.data')
    coco = COCO(os.path.join(os.path.dirname(target_image_dir),'labels','coco128.json'))
    
    for key in frames_dict:
        old_coco = COCO(os.path.join(source_dataset_dir, key, "labels", "coco128.json"))
        gender_data = pickle.load(open(os.path.join(source_dataset_dir, key, "labels", "gender.data"), 'rb'))  # getting old gender data
        image_mapping = {}  # Mapping of old image IDs to new image IDs
        for image in coco.dataset['images']:
            image_mapping[image['file_name']] = image['id']
        for i in frames_dict[key]:
            attribute = {}
            old_img_id = None
            for image in old_coco.dataset['images']:
                if image['file_name'] == i:
                    old_img_id = image['id']
                    break
            if old_img_id is not None:
                new_img_id = image_mapping.get(f'{key}_{i}')
                if new_img_id is not None:
                    value = find_value_in_list_of_dicts(gender_data, old_img_id)
                    if value is not None:
                        attribute[int(new_img_id)] = value
                        new_gender_data.append(attribute)
    pickle.dump(new_gender_data,open(new_gender_path,'wb'))



    


if __name__ == "__main__":

    file_paths = ['/mnt/g/Github/fair_video_summarization_tmlr/models/pgl-sum/exp2/tvsum/summaries/split0/']
    # file_paths = [ "/mnt/g/Github/video_summarizer/logs/2024-03-20_08-55-54_DSNTrainer/tvsum_splits.json_preds.h5"]
    #               "/mnt/g/Github/video_summarizer/logs/2024-03-20_08-21-35_SumGANAttTrainer/tvsum_splits.json_preds.h5",
    #               "/mnt/g/Github/video_summarizer/logs/2024-03-18_22-02-22_TransformerTrainer/tvsum_splits.json_preds.h5",
    #               "/mnt/g/Github/video_summarizer/logs/2024-03-20_08-17-47_VASNetTrainer/tvsum_splits.json_preds.h5"]
    # file_paths = ["/mnt/g/Github/video_summarizer/logs/1707472329_SumGANAttTrainer/summe_splits.json_preds.h5",
    #               "/mnt/g/Github/video_summarizer/logs/1706525703_SumGANTrainer/summe_splits.json_preds.h5",
    #               "/mnt/g/Github/video_summarizer/logs/2024-03-19_14-51-35_TransformerTrainer/summe_splits.json_preds.h5",
    #               "/mnt/g/Github/video_summarizer/logs/2024-03-20_08-55-09_VASNetTrainer/summe_splits.json_preds.h5",
    #               "/mnt/g/Github/video_summarizer/logs/1707231795_DSNTrainer/summe_splits.json_preds.h5"]
    # file_paths = ['/mnt/g/Github/PGL-SUM/inference/summaries/table4_models/tvsum/split0.h5',
    #               '/mnt/g/Github/PGL-SUM/inference/summaries/table3_models/tvsum/split1.h5',
    #               '/mnt/g/Github/PGL-SUM/inference/summaries/table4_models/summe/split2.h5',
    #               '/mnt/g/Github/PGL-SUM/inference/summaries/table3_models/summe/split3.h5']



    for file_path in file_paths:
        # with h5py.File(file_path, 'r') as f:
        #     data =f
        #     keys_path = list(f.keys())
        #     # for video in f[keys_path[0]].keys():
        #     for video in f.keys():
        #         # print(video)
        #         args = argparse.Namespace(fps=30, height=240, width=320,save_type =False)
        #         args.video = video
        #         args.path = file_path
        #         args.save_type = False
        #         # if file_path[0].split('_')[2] == 'tvsum':
        #         #     args.frames = f'datasets/video/frames/{video}'
        #         # if keys_path[0].split('_')[2] == 'summe':
        #         #     args.frames = f'datasets/videos/frames/{video}'
        #         if os.path.basename(os.path.dirname(file_path)) == 'tvsum':
        #             args.frames = f'datasets/video/frames/{video}'
        #         if os.path.basename(os.path.dirname(file_path)) == 'summe':
        #             args.frames = f'datasets/videos/frames/{video}'
        #         args.frames = f'datasets/videos/frames/{video}'
        #         # # args.fps = 'your_fps'
        #         # # args.width = 'your_width'
        #         # # args.height = 'your_height'
                
        #         args.dataset = list(f.keys())[0]
                # print(args)
        #         summarise(args)  # Call the main function with args as the argument
        # if os.path.basename(file_path).split('_')[0]=='tvsum':
        #     source_dataset_folder = '/mnt/g/Github/video_summarizer/datasets/video/frames/'
        # if os.path.basename(file_path).split('_')[0] == 'summe':
        #     source_dataset_folder = '/mnt/g/Github/video_summarizer/datasets/videos/frames/'
        # target_image_dir = os.path.join(os.path.dirname(file_path),'summaries',keys_path[0].split('.')[0],'frames')
        # if os.path.basename(os.path.dirname(file_path)) == 'tvsum':
        #     source_dataset_folder = '/mnt/g/Github/video_summarizer/datasets/video/frames/'
        # if os.path.basename(os.path.dirname(file_path)) == 'summe':
        #     source_dataset_folder = '/mnt/g/Github/video_summarizer/datasets/videos/frames/'
        # target_image_dir = os.path.join(os.path.dirname(file_path),'summaries','frames')
        source_dataset_folder = '/mnt/g/Github/video_summarizer/datasets/video/dataset/'
        target_image_dir = os.path.join(file_path,'frames')
        print(source_dataset_folder,target_image_dir)
        import_annotations(source_dataset_folder, target_image_dir)

    # keys_path = [i.split('.')[0] for i in keys_path]
        # auto_label.annotate(target_image_dir)

