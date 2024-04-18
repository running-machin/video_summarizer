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

VIDEO_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"]
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16  # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

def labeling(source):
    model = YOLOWorld('yolov8x-world.pt')  # Load pretrain or fine-tune model

    # Process the image
    # source = cv2.imread('/workspace/detect-anotation/frames_test/frame23.jpg')
    results = model.predict(source)

    return results

def yolo_format(output_dir, results):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i in tqdm(range(len(results)), desc='Saving labels'):
        path = results[i].path
        file_name = os.path.basename(path).split('.')[0]
        results[i].save_txt(f'{output_dir}/{file_name}.txt', save_conf=True)

def clean_label(output_dir, threshold = 0.6):
    for text_file in tqdm(os.listdir(output_dir),desc='Cleaning low accurate labels'):
        text_file = os.path.join(output_dir, text_file)
        # print(text_file)
        with open(text_file, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
            # print(lines)
            # Filter lines based on the condition
            filtered_lines = [line for line in lines if float(line.strip().split()[-1]) >= threshold]
            # print(filtered_lines)
            # Remove the last element (last value) from each line
            cleaned_lines = [line.strip().rsplit(' ', 1)[0] for line in filtered_lines]
            # print(cleaned_lines)
            # Check if the file is empty after filtering
        if not cleaned_lines:
            # File is empty, so delete it
            os.remove(text_file)
        else:
            # raise error if the cleaned lines are more than 5 items
            cleaned_lines_check = [line.split() for line in cleaned_lines]
            all_len_5 = all(len(sublist) == 5 for sublist in cleaned_lines_check)
            if not all_len_5:
                raise ValueError(f"Too many or too few items in the cleaned lines for {text_file}\n{filtered_lines}\n{cleaned_lines}")
            # Open the file for writing and save the cleaned lines
            with open(text_file, 'w') as file:
                file.write('\n'.join(cleaned_lines))

def yolo_to_coco(yolo_labels, frames_dir):
    yoloclasses = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

    dataset = importer.ImportYoloV5(path=yolo_labels, path_to_images=frames_dir, cat_names=yoloclasses, img_ext="jpg", name="coco128")

    dataset.export.ExportToCoco(cat_id_index=1)
    return dataset.df

def gender_detect(source):
    if not os.path.exists('yolov8x-world_gender.pt'):        
        model = YOLOWorld('yolov8x-world.pt')
        model.set_classes(['man','woman'])
        model.save('yolov8x-world_gender.pt')
    model = YOLOWorld('yolov8x-world_gender.pt')
    results = model.predict(source)
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


if __name__ == "__main__":

    videos_path = '/mnt/g/Github/video_summarizer/downloaded_videos'
    videos = [os.path.join(videos_path, file) for file in os.listdir(videos_path) if file.endswith(tuple(VIDEO_EXTENSIONS))]
    batch_size = 100
    for video in videos:
        label_path = os.path.join(os.path.dirname(video),os.path.basename(video).split('.')[0],'labels')
        frames_path = os.path.join(os.path.dirname(video),os.path.basename(video).split('.')[0],'frames')
        images = [os.path.join(frames_path,img) for img in os.listdir(frames_path) if img.endswith('.jpg')]
        # load every 1000 images to labelling to reduce the RAM usage
        for i in range(0, len(images), batch_size):
            print(f"Processing Batch of {i+batch_size} out of {len(images)}")
            labels = labeling(images[i:i+batch_size])
            yolo_format(label_path, labels)
        clean_label(label_path)
        coco_df = yolo_to_coco(label_path, frames_path)
        coco_df = importer.ImportCoco(os.path.join(label_path,'coco128.json')).df
        coco_df = clean_coco(coco_df)
        data= []
        images = list(set(coco_df['img_path'].values))
        for i in range(0, len(images), batch_size):
            print(f"Processing Batch of {i+batch_size} out of {len(images)}")
            results = gender_detect(images[i:i+batch_size])
            data.extend([gender_data(result,coco_df) for result in tqdm(results,desc='Processing gender data')])
        data = [d for d in data if (0,0) not in d.values()]
        gender_data_export(data ,label_path)

        