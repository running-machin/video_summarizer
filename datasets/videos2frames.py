import glob
import os
import sys
import subprocess

# print(os.curdir)

# get max threads from the machine
max_threads = os.cpu_count()
def process_videos_in_directory(directory, max_threads):
    # get current directory
    default_dir = os.getcwd()
    # Change the current directory to the provided directory
    os.chdir(directory)
    print(os.curdir)
    files = sorted(glob.glob("*.mp4"))
    # Iterate over all .mp4 files in the current directory
    for i,f in enumerate(files):
        print(f"Processing {f} file...")
        basename = os.path.basename(f).replace('.mp4', '')
        name = basename.split('.')[0]
        # print(f"{name}_{f.index}")
        output_dir = os.path.join("frames", f"video_{i+1}") # giving the index of the video as per the h5 file
        os.makedirs(output_dir, exist_ok=True)
        ffmpeg_cmd = [
            "ffmpeg",
            "-threads", str(max_threads),
            "-i", f,
            # "-pix_fmt", "yuv420p",
            "-color_range", "1",
            os.path.join(output_dir, "%05d.jpg")
        ]
        print(subprocess.run(ffmpeg_cmd))
    # print("Processing complete.")
    # Change the current directory back to the original directory
    os.chdir(default_dir)

if __name__ == '__main__':
    print(os.getcwd)
# Call the function with the directory you want
process_videos_in_directory('~./datasets/video/', max_threads)
process_videos_in_directory('~./datasets/videos/', max_threads)
