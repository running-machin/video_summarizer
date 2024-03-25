import requests
import pandas as pd
import os
import sys
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
# tqdm.pandas()
from multiprocessing import Pool

def extract_captions(file_path):
    """
    Simple function to extract captions from an image file.
    Args:
        file_path (str): Path to the image file.
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")
    raw_image = Image.open(file_path).convert('RGB')
    # conditional image captioning
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def extract_captions_batched(file_paths, processor, model):
    """Processes images in batches and returns captions with image names

    Args:
        file_paths (list): List of image file paths.
        processor (BlipProcessor): Preprocessor for the model.
        model (BlipForConditionalGeneration): Blip model for caption generation.

    Returns:
        list: List of captions with corresponding image names.
    """

    batch_size = 50  # Adjust based on GPU memory (experiment for optimal size)
    captions_with_names = []
    for i in tqdm(range(0, len(file_paths), batch_size)):
        batch_images = []
        batch_image_names = []

        for file_path in file_paths[i:i+batch_size]:
            raw_image = Image.open(file_path).convert('RGB')
            batch_images.append(raw_image)
            batch_image_names.append(file_path.split("/")[-1])  # Extract filename

        # Preprocess the batch
        text = ["a photography of"] * len(batch_images)
        inputs = processor(batch_images,text, return_tensors="pt").to("cuda")

        # Unconditional image captioning
        out = model.generate(**inputs)
        captions = processor.batch_decode(out, skip_special_tokens=True)

        # Combine captions with image names
        for image_name, caption in zip(batch_image_names, captions):
            captions_with_names.append((image_name, caption))

    return captions_with_names

def caption_dataset(frames_path):
    # intialize the dataframe
    df = pd.DataFrame(columns=["frame_name", "caption"])
    # path to the frames
    # frames = os.listdir(frames_path)
    frames = [os.path.join(frames_path, f) for f in os.listdir(frames_path) if f.endswith('.jpg')]
    # def process_frame(frame):
    #     # print(f"Processing frame: {os.path.join(frames_path, frame)}")
    #     caption = extract_captions(frame)
    #     return {"frame_name": frame, "caption": caption}    
    # df["frame_path"] = df["frame_name"].apply(lambda img: os.path.join(frames_path, img))
    tqdm.pandas()
    # df = df.progress_apply(process_frame, axis=1)
    captions = extract_captions_batched(frames, processor, model)
    # remove the image_path column
    df = pd.DataFrame(captions, columns=['frame_name', 'caption'])


    return df

if __name__ == "__main__":
    frames_path = "/mnt/g/Github/video_summarizer/logs/2024-03-20_08-55-09_VASNetTrainer/summaries/frames"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large") 
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")
    attributes = caption_dataset(frames_path)
    # save the dataframe
    attributes.to_csv(os.path.join(frames_path, "attributes.csv"), index=False)
