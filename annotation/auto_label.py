import os
from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO,YOLOWorld
from pylabel import importer
import cv2
import whisperx

model = whisper.load_model("NbAiLab/whisper-large-v2-nob", device="cuda")
dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = "temp_audio.wav"
result = whisper.transcribe(model, sample, language="en")

def labeling(source):
    model = YOLOWorld('yolov8x-world.pt')  # Load pretrain or fine-tune model

    # Process the image
    # source = cv2.imread('/workspace/detect-anotation/frames_test/frame23.jpg')
    results = model(source)

    return results

def yolo_format(output_dir, results):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i in range(len(results)):
        path = results[i].path
        file_name = os.path.basename(path).split('.')[0]
        results[i].save_txt(f'/workspace/detect-anotation/frames_test_label/{file_name}.txt', save_conf=True)

def clean_label(output_dir, threshold = 0.6):
    for text_file in os.listdir(output_dir):
        text_file = os.path.join(output_dir, text_file)
    print(text_file)
    with open(text_file, 'r') as file:
        lines = file.readlines()
    # Filter lines based on the condition
    filtered_lines = [line for line in lines if float(line.strip().split()[-1]) >= threshold]

    # Remove the last element (last value) from each line
    cleaned_lines = [line.strip().rsplit(' ', 1)[0] for line in filtered_lines]
        # Check if the file is empty after filtering
    if not cleaned_lines:
        # File is empty, so delete it
        os.remove(text_file)
    else:
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

    dataset = importer.ImportYoloV5(path=path_to_annotations, path_to_images=path_to_images, cat_names=yoloclasses,
    img_ext="jpg", name="coco128")

    dataset.export.ExportToCoco(cat_id_index=1)
    return dataset.df

def audio_transcript:
    pass
