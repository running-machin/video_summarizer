import os
from tqdm import tqdm
# from ultralytics.utils.plotting import Annotator
from ultralytics import YOLOWorld
from moviepy.editor import VideoFileClip
from pylabel import importer
import torch
import whisperx
import pickle
import pandas as pd
import numpy as np
from dataclasses import dataclass
# from mivolo import Predictor

VIDEO_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"]
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16  # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
silent_mode={"verbose": False}
def labeling(source):
    model = YOLOWorld('yolov8x-world.pt')  # Load pretrain or fine-tune model

    # Process the image
    # source = cv2.imread('/workspace/detect-anotation/frames_test/frame23.jpg')
    results = model.predict(source,**silent_mode)

    return results

def yolo_format(output_dir, results):
    os.makedirs(output_dir, exist_ok=True)

    for result in tqdm(results, desc='Saving labels'):
        path = result.path
        file_name = os.path.basename(path).split('.')[0]
        result.save_txt(f'{output_dir}/{file_name}.txt', save_conf=True)

def clean_label(output_dir, threshold=0.5):
    # Iterate over each text file in the output directory
    for text_file in tqdm([file for file in os.listdir(output_dir) if file.endswith('.txt')], desc='Cleaning low accurate labels'):
        text_file = os.path.join(output_dir, text_file)
        with open(text_file, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
            filtered_lines = []
            # Iterate over each line in the text file
            for line in lines:
                line_split = line.strip().split()
                # Check if the line has more than 5 elements and the confidence score is above the threshold
                if len(line_split) == 6 and float(line_split[-1]) >= threshold: 
                    filtered_lines.append(line_split[:-1])
                elif  len(line_split) == 5: #already filtered 
                    filtered_lines.append(line_split)
            # If there are no filtered lines, remove the text file
            if not filtered_lines :
                os.remove(text_file)
            else:
                if any(len(line) != 5 for line in filtered_lines):
                    raise ValueError(f"Line contains more or less than 5 items for {text_file}: {filtered_lines}")
                cleaned_lines = [' '.join(line) for line in filtered_lines]
                # Write the cleaned lines back to the text file
                with open(text_file, 'w') as file:
                    file.write('\n'.join(cleaned_lines))

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
    return dataset.df
@dataclass
class Cfg:
    detector_weights: str
    checkpoint: str
    device: str = "cuda"  
    with_persons: bool = True
    disable_faces: bool = False
    draw: bool = False


def load_model():
    # Update the paths to the locations where you saved the files
    detector_path = 'yolov8x_person_face.pt'
    age_gender_path = 'model_imdb_cross_person_4.22_99.46.pth.tar'

    predictor_cfg = Cfg(detector_path, age_gender_path)
    predictor = Predictor(predictor_cfg)

    return predictor

def detect_details_batch(images, score_threshold, iou_threshold, mode, predictor):
    predictor.detector.detector_kwargs['conf'] = score_threshold
    predictor.detector.detector_kwargs['iou'] = iou_threshold

    if mode == "Use persons and faces":
        use_persons = True
        disable_faces = False
    elif mode == "Use persons only":
        use_persons = True
        disable_faces = True
    elif mode == "Use faces only":
        use_persons = False
        disable_faces = False

    predictor.age_gender_model.meta.use_persons = use_persons
    predictor.age_gender_model.meta.disable_faces = disable_faces

    batch_face_details = []
    for image in images:
        image = image[:, :, ::-1]  # RGB -> BGR
        detected_objects, out_im = predictor.recognize(image)
        details = []
        for ind, det in enumerate(detected_objects.yolo_results.boxes):
            bbox = detected_objects.get_bbox_by_ind(ind).cpu().numpy()
            confidence = float(det.conf) if det.conf is not None else None
            age = detected_objects.ages[ind]
            gender = detected_objects.genders[ind]
            class_name = detected_objects.yolo_results.names[int(det.cls)]

            details.append({
                'bbox': bbox,
                'confidence': confidence,
                'age': age,
                'gender': gender,
                'class': class_name
            })
        batch_face_details.append(details)

    return batch_face_details

def detect_details(image: np.ndarray, score_threshold: float, iou_threshold: float, mode: str, predictor: Predictor) -> dict:
    # input is rgb image, output must be rgb too

    predictor.detector.detector_kwargs['conf'] = score_threshold
    predictor.detector.detector_kwargs['iou'] = iou_threshold

    if mode == "Use persons and faces":
        use_persons = True
        disable_faces = False
    elif mode == "Use persons only":
        use_persons = True
        disable_faces = True
    elif mode == "Use faces only":
        use_persons = False
        disable_faces = False

    predictor.age_gender_model.meta.use_persons = use_persons
    predictor.age_gender_model.meta.disable_faces = disable_faces
    image = image[:, :, ::-1]  # RGB -> BGR
    detected_objects, out_im = predictor.recognize(image)
    details = []
    # Iterate over all objects
    for ind, det in enumerate(detected_objects.yolo_results.boxes):
        bbox = detected_objects.get_bbox_by_ind(ind).cpu().numpy()
        confidence = float(det.conf) if det.conf is not None else None
        age = detected_objects.ages[ind]
        gender = detected_objects.genders[ind]
        class_name = detected_objects.yolo_results.names[int(det.cls)]
        path = detected_objects.yolo_results.paths[int(det.cls)]
        details.append({
            'bbox': bbox,
            'confidence': confidence,
            'age': age,
            'gender': gender,
            'class' : class_name
        })
    return details
def gender_detect(source):
    if not os.path.exists('yolov8x-world_gender.pt'):        
        model = YOLOWorld('yolov8x-world.pt')
        model.set_classes(['man','woman'])
        model.save('yolov8x-world_gender.pt')
    model = YOLOWorld('yolov8x-world_gender.pt')
    results = model.predict(source,**silent_mode)
    return results
def check_gender(dict_list):
    genders = (0, 0)
    for d in dict_list:
        if 'name' in d:
            # if 'woman' in d['name'].lower() and d['confidence'] > 0.6:
            #     genders = (genders[0], genders[1] + 1)
            # elif 'man' in d['name'].lower() and d['confidence'] > 0.6:
            #     genders = (genders[0] + 1, genders[1])
            if 'woman' in d['name'].lower():
                genders = (genders[0], genders[1] + 1)
            elif 'man' in d['name'].lower():
                genders = (genders[0] + 1, genders[1])
    return genders


def gender_data(results,df):
    attribute = {}
    file_name = os.path.basename(results.path)
    # id = [img['id'] for img in coco.dataset['images'] if img['file_name'] == file_name][0]
    id = df[df['img_filename'] == file_name]['img_id'].values[0]
    # gender = 1 if results[0].summary()[0]['name'] == 'man' else 0
    attribute[id] = check_gender(results.summary())
    return attribute

def clean_coco(df):
    # Select columns: 'img_folder', 'img_filename' and 2 other columns
    df = df.loc[:, ['img_folder', 'img_filename', 'img_path', 'img_id']]
    # Join 'img_folder' and 'img_filename' to create 'img_path'
    df['img_path'] = df.apply(lambda row: os.path.join(row['img_folder'], row['img_filename']), axis=1)
    
    return df

def gender_data_export(data:list,label_path):
    with open(os.path.join(label_path,'gender.data'),'wb') as f:
        pickle.dump(data,f)

def extract_audio(video):
    video = VideoFileClip(video)
    audio = video.audio
    audio_file = os.path.join(os.getcwd(), "__temp__.mp4")
    audio.write_audiofile(audio_file)
    return audio_file

def audio_transcript(audio):
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio)
    transcript = model.transcribe(audio, batch_size=batch_size)
    return transcript



def annotate(videos_path:str):
    '''
    Annotate the videos in the given
    input: videos_path: str : path to the video, should contain the frames folder in it
    '''
    # videos_path = '/mnt/g/Github/video_summarizer/downloaded_videos'
    # videos = [os.path.join(videos_path, file) for file in os.listdir(videos_path) if file.endswith(tuple(VIDEO_EXTENSIONS))]
    videos = [videos_path]
    batch_size = 100
    predictor = load_model()
    for video in videos:

        label_path = os.path.join(os.path.dirname(video),os.path.basename(video).split('.')[0],'labels')
        # # frames_path = os.path.join(os.path.dirname(video),os.path.basename(video).split('.')[0],'frames')
        # frames_path = os.path.join(os.path.dirname(video),os.path.basename(video).split('.')[0])
        # # gender_path = os.path.join(label_path,'gender_labels')
        # images = [os.path.join(frames_path,img) for img in os.listdir(frames_path) if img.endswith('.jpg')]
        # # remove the images that already have txt file in the label_path
        # old_labels = list(os.listdir(label_path))
        # labels = [os.path.join(frames_path,label.split('.')[0]+'.jpg') for label in old_labels if label.endswith('.txt')]
        # images = list(set(images).difference(set(labels)))
        # print(len(images), frames_path)
        # # load every 100 images to labelling to reduce the RAM usage
        # for i in range(0, len(images), batch_size):
        #     print(f"Processing Batch of {i+batch_size} out of {len(images)}")
        #     labels = labeling(images[i:i+batch_size])
        #     yolo_format(label_path, labels)
        # clean_label(label_path)
        # coco_df = yolo_to_coco(label_path, frames_path)
        coco_df = importer.ImportCoco(os.path.join(label_path,'coco128.json')).df
        coco_df = clean_coco(coco_df) 
        # # filtering already existing gender.data, in case of 're-'object detection
        # if os.path.exists(os.path.join(label_path,'gender.data')):
        #     with open(os.path.join(label_path, 'gender.data'),'rb') as f:
        #         data = pickle.load(f)
        #         # get all the img-id from list of dictionaries
        #         already_img = [int(list(d.keys())[0]) for d in data]
        #         print(already_img)
        #         already_img = [coco_df[coco_df['img_id'] == img_id]['img_path'].values[0] for img_id in already_img if img_id in coco_df['img_id'].values]
        #         images = list(set(coco_df['img_path'].values).difference(set(already_img)))
        # else:
        data = []
        images = list(set(coco_df['img_path'].values))
        for i in range(0, len(images), batch_size):
            print(f"Processing Gender Batch of {i+batch_size} out of {len(images)}")
            # results = gender_detect(images[i:i+batch_size])
            results = detect_details_batch(images, score_threshold=0.4, iou_threshold=0.7, mode="Use persons", predictor=predictor)
            print(results)
            break
            data.extend([gender_data(result,coco_df) for result in tqdm(results,desc='Processing gender data')])
        break
        #     yolo_format(gender_path,results)

        # clean_label(gender_path)
        # gender_df = yolo_to_coco(gender_path, frames_path)
        # gender_df = importer.ImportCoco(os.path.join(label_path,'coco128.json')).df
        # gender_df = clean_coco(gender_df)
        
        data = [d for d in data if (0,0) not in d.values()]
        gender_data_export(data ,label_path)
        
        # gender_data_export(data ,gender_path)

if __name__ == "__main__":
    """
    USE THE ONE IN REVISE_TOOLS BEACUSE OF DEPENDENCIES ISSUES
    """
    # video_paths = ["/mnt/g/Github/video_summarizer/logs/1707472329_SumGANAttTrainer/summe_splits.json_preds.h5",
    #               "/mnt/g/Github/video_summarizer/logs/1706525703_SumGANTrainer/summe_splits.json_preds.h5",
    #               "/mnt/g/Github/video_summarizer/logs/2024-03-19_14-51-35_TransformerTrainer/summe_splits.json_preds.h5",
    #               "/mnt/g/Github/video_summarizer/logs/2024-03-20_08-55-09_VASNetTrainer/summe_splits.json_preds.h5",
    #               "/mnt/g/Github/video_summarizer/logs/1707231795_DSNTrainer/summe_splits.json_preds.h5"]
                
            
    # for video in video_paths:
    #     annotate(video)
    datasets = [#"/mnt/g/Github/video_summarizer/datasets/video",
                "/mnt/g/Github/video_summarizer/datasets/videos"]
    for dataset in datasets:
        frames_folder = os.path.join(dataset, 'frames')
        videos_folder = os.listdir(frames_folder)
        for video in videos_folder:
            annotate(os.path.join(frames_folder,video))