import argparse
import h5py
from summary import summarise  # Import the summarise function from summary.py

if __name__ == "__main__":
    file_path = "/mnt/g/Github/video_summarizer/logs/2024-03-20_08-55-09_VASNetTrainer/summe_splits.json_preds.h5"
    f = h5py.File(file_path, "r")
    for video in f['summarizer_dataset_summe_google_pool5.h5'].keys():
        print(video)
        args = argparse.Namespace(fps=30, height=240, width=320,save_type =False)
        args.video = video
        args.path = file_path
        args.save_type = False
        # args.fps = 'your_fps'
        # args.width = 'your_width'
        # args.height = 'your_height'
        args.frames = f'datasets/videos/frames/{video}'
        args.dataset = list(f.keys())[0]

        summarise(args)  # Call the main function with args as the argument