import glob
import os
import sys
import subprocess
from subprocess import Popen,PIPE
import h5py

# print(os.curdir)

# get max threads from the machine
max_threads = os.cpu_count()
def extract_frames_multithreaded(video_path, output_dir, num_threads=4):
  """
  Extracts frames from a video using ffmpeg with multithreading on CPU cores.

  Args:
      video_path: Path to the input video file.
      output_dir: Directory to store the extracted frames.
      num_threads: Number of threads to use for parallel processing (default: 4).
  """

  # Get original video resolution
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  # command = ["ffprobe", "-v", "error", "-show_format", "-show_streams", video_path]
  # probe_process = Popen(command, stdout=PIPE, stderr=PIPE)
  # output, _ = probe_process.communicate()
  # output_str = output.decode('utf-8')

  # # Find video width and height from ffprobe output
  # width = None
  # height = None
  # for line in output_str.splitlines():
  #   if line.startswith("width="):
  #     width = int(line.split("=")[1])
  #   elif line.startswith("height="):
  #     height = int(line.split("=")[1])
  #   elif line.startswith("bit_rate="):
  #     bit_rate = int(line.split("=")[1])
  # if not width or not height:
  #   raise ValueError("Failed to determine video resolution using ffprobe.")

  # Construct ffmpeg command with multithreading
  command = [
      "ffmpeg",
      "-i", video_path,
      "-threads", str(num_threads),
      # "-vf", f"scale={width}:{height}",  # Scale to maintain original width
      # "-b:v", f"{bit_rate}",
      f"{output_dir}/frame_%05d.jpg"  # Output format with padding
  ]
  print(" ".join([w for w in command]))
  process = Popen(command)
  process.wait()


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
    process_videos_in_directory('/mnt/g/Github/video_summarizer/datasets/video/', max_threads)
    process_videos_in_directory('/mnt/g/Github/video_summarizer/datasets/videos/', max_threads)
# filename = 'datasets/summarizer_dataset_tvsum_google_pool5.h5'
# with h5py.File(filename,'r') as file:
#         # print(list(file['video_1'].keys()))
#         video_name = file['video_1']['video_name'][()]
# extract_frames_multithreaded(f'datasets/video/{video_name}.mp4', 'datasets/video/frames/video_1', num_threads=4)

