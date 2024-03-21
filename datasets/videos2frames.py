import glob
import os
import sys
import subprocess

# print(os.curdir)

# get max threads from the machine
max_threads = os.cpu_count()
def process_videos_in_directory(directory, max_threads):
    # Change the current directory to the provided directory
    os.chdir(directory)
    print(os.curdir)
    # Iterate over all .mp4 files in the current directory
    for f in glob.glob("*.mp4"):
        print(f"Processing {f} file...")
        basename = os.path.basename(f).replace('.mp4', '')
        name = basename.split('.')[0]
        print(f)
        output_dir = os.path.join("frames", name)
        os.makedirs(output_dir, exist_ok=True)
        ffmpeg_cmd = [
            "ffmpeg",
            "-threads", str(max_threads),
            "-i", f,
            "-pix_fmt", "yuv420p",
            "-color_range", "1",
            os.path.join(output_dir, "%05d.jpg")
        ]
        print(subprocess.run(ffmpeg_cmd))
    # print("Processing complete.")

# Call the function with the directory you want
process_videos_in_directory('/mnt/g/Github/video_summarizer/datasets/video/', max_threads)
process_videos_in_directory('/mnt/g/Github/video_summarizer/datasets/videos/', max_threads)
