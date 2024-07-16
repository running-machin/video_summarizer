import argparse
import os
import h5py
from summary import summarise  # Import the summarise function from summary.py
from annotation import auto_label
if __name__ == "__main__":
    # file_path = 
    # file_paths = ["/mnt/g/Github/video_summarizer/logs/2024-03-20_08-55-54_DSNTrainer/tvsum_splits.json_preds.h5",
    #               "/mnt/g/Github/video_summarizer/logs/2024-03-20_08-21-35_SumGANAttTrainer/tvsum_splits.json_preds.h5",
    #               "/mnt/g/Github/video_summarizer/logs/2024-03-18_22-02-22_TransformerTrainer/tvsum_splits.json_preds.h5",
    #               "/mnt/g/Github/video_summarizer/logs/2024-03-20_08-17-47_VASNetTrainer/tvsum_splits.json_preds.h5"]
    # file_paths = ["/mnt/g/Github/video_summarizer/logs/1707472329_SumGANAttTrainer/summe_splits.json_preds.h5",
    #               "/mnt/g/Github/video_summarizer/logs/1706525703_SumGANTrainer/summe_splits.json_preds.h5",
    #               "/mnt/g/Github/video_summarizer/logs/2024-03-19_14-51-35_TransformerTrainer/summe_splits.json_preds.h5",
    #               "/mnt/g/Github/video_summarizer/logs/2024-03-20_08-55-09_VASNetTrainer/summe_splits.json_preds.h5",
    #               "/mnt/g/Github/video_summarizer/logs/1707231795_DSNTrainer/summe_splits.json_preds.h5"]
                



    for file_path in file_paths:
        with h5py.File(file_path, 'r') as f:
            data =f
            keys_path = list(f.keys())
            for video in f[keys_path[0]].keys():
                print(video)
                args = argparse.Namespace(fps=30, height=240, width=320,save_type =False)
                args.video = video
                args.path = file_path
                args.save_type = False
                if keys_path[0].split('_')[2] == 'tvsum':
                    args.frames = f'datasets/video/frames/{video}'
                if keys_path[0].split('_')[2] == 'summe':
                    args.frames = f'datasets/videos/frames/{video}'
                # # args.fps = 'your_fps'
                # # args.width = 'your_width'
                # # args.height = 'your_height'
                
                args.dataset = list(f.keys())[0]

                summarise(args)  # Call the main function with args as the argument
    # keys_path = [i.split('.')[0] for i in keys_path]
    # auto_label.annotate(os.path.join(os.path.dirname(file_path),'summaries',keys_path[0]))

# for file_path in file_paths:
#     with h5py.File(file_path, 'r') as f:
#         keys_path = list(f.keys())
#         # keys_path = [i+'h5' for i in keys_path]
#         print(keys_path)
#     print(os.path.join(os.path.dirname(file_path),'summaries',keys_path[0]))
    