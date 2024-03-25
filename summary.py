import argparse
import h5py
import cv2
from tqdm import tqdm
import os
import os.path as osp

"""
Generate a video summary from the predictions and the frames.
"""

def frm2video(frm_dir, summary, vid_writer, width, height):
    for idx, val in tqdm(enumerate(summary), total=len(summary), ncols=80):
        if val == 1:
            # here frame name starts with "000001.jpg"
            frm_name = str(idx+1).zfill(5) + ".jpg"
            frm_path = osp.join(frm_dir, frm_name)
            frm = cv2.imread(frm_path)
            frm = cv2.resize(frm, (width, height))
            vid_writer.write(frm)

def framesfromsummary(frm_dir,video, save_path, summary, width, height):
    # Create the directory if it doesn't exist
    if not osp.exists(save_path):
        os.makedirs(save_path)
    frames = []
    for idx, val in tqdm(enumerate(summary), total=len(summary), ncols=80):
        if val == 1:
            # here frame name starts with "000001.jpg"
            frm_name = str(idx+1).zfill(5) + ".jpg"
            frm_path = osp.join(frm_dir, frm_name)
            frm = cv2.imread(frm_path)
            frm = cv2.resize(frm, (width, height))
            # save the frame in the directiory as jpg
            cv2.imwrite(osp.join(save_path, f'{video}_{frm_name}'), frm)
            # check if the frame is using os.path
            # if not osp.exists(osp.join(save_path, f'{video}resized_{frm_name}')):
            #     print(f"Failed to save frame : {frm_name} at {save_path}")

            frames.append(frm)
    return frames

def summarise(args):
    # model name from the args.path
    # model_name = osp.basename(osp.dirname(args.path)).split("_")[-1]
    # summary_file = f"summary_{model_name}_{args.video}.mp4"
    summary_file = f"summary_{args.video}.mp4"
    if args.save_type:
        summary_path = osp.join(osp.dirname(args.path),'summaries', summary_file)
        vid_writer = cv2.VideoWriter(
            summary_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (args.width, args.height)
        )
        h5_preds = h5py.File(args.path, "r")
        dataset = h5_preds[args.dataset]
        summary = dataset[args.video]["machine_summary"][...]
        h5_preds.close()
        frm2video(args.frames, summary, vid_writer, args.width, args.height)
        vid_writer.release()
    else:
        # save as frames
        summary_path = osp.join(osp.dirname(args.path),'summaries','frames')
        h5_preds = h5py.File(args.path, "r")
        dataset = h5_preds[args.dataset]
        summary = dataset[args.video]["machine_summary"][...]
        framesfromsummary(args.frames, args.video, summary_path, summary, args.width, args.height)

    print(f"Summary saved at {summary_path}")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to hdfs5 predictions file")
    parser.add_argument("-f", "--frames", type=str, required=True, help="Path to frame directory")
    parser.add_argument("-d", "--dataset", type=str, help="Dataset hdfs5 filename")
    parser.add_argument("-s", "--save_type", type=bool, default=True, help="If save as videos,else save as frames")
    parser.add_argument("-v", "--video", type=str, help="Which video key to choose")
    parser.add_argument("--fps", type=int, default=25, help="frames per second")
    parser.add_argument("--width", type=int, default=1280, help="frame width")
    parser.add_argument("--height", type=int, default=720, help="frame height")
    args = parser.parse_args()
    summarise(args)
