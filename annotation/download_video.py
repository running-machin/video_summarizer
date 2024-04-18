from pytube import YouTube, Stream
import os
from subprocess import Popen, PIPE

class Downloader:
    def __init__(self, url):
        """
        Class to handle downloading a video from youtube

        Args:
            url (str): URL of the youtube video
        """
        self.url = url
        """ URL of the youtube video """
        self.file_name = None
        """ Filename of the downloaded video """
        self.video  = YouTube(self.url)
        """ pytube.YouTube object of the video """
        self.output_dir = os.path.join(os.getcwd(), 'downloaded_videos')
        """ Path where the video is downloaded to """

    def get_file_name(self):
        """
        Returns:
            str: File name of the video of the YouTube object (self.video).
                The file name is derived from the title of the YouTube video.
                Spaces in the title are replaced with underscores and the file extension is '.mp4'
        """
        file_name = self.video.title.replace(' ', '_')+'.mp4'
        if not file_name:
            raise ValueError('Video title is empty. Cannot download video')
        return file_name

    # def get_video_resolution(self, video):
    #     video = video.streams.filter(progressive=True, file_extension='mp4')

    
    def download(self, resolution, output_dir):
        """
        Downloads the video to the specified output directory with the specified resolution

        Args:
            resolution (str): The resolution of the video to be downloaded.
                Possible values are 'high' and 'low'
            output_dir (str): The path to the directory where the video is to be downloaded.

        Returns:
            str: The path to the downloaded video file
        """
        file_name = self.get_file_name()
        if resolution == 'high':
            # Download the video with the highest resolution
            try:
                downloaded_video = self.video.streams.get_highest_resolution().download(
                    output_path=output_dir, filename=file_name)
            except:
                raise ValueError('Could not download video with highest resolution')
        elif resolution == 'low':
            # Download the video with the lowest resolution
            try:
                downloaded_video = self.video.streams.get_lowest_resolution().download(
                    output_path=output_dir, filename=file_name)
            except:
                raise ValueError('Could not download video with lowest resolution')
        else:
            raise ValueError('Invalid resolution')
        return downloaded_video


class vid2frames:
    """
    Class for extracting frames from video files using ffmpeg.

    Attributes:
        video_path (str): Path to input video file.
        file_name (str): Base name of the video file.
        output_dir (str): Directory to store extracted frames.
        num_threads (int): Number of threads to use for parallel processing
            (default: number of CPU cores).
    """
    def __init__(self, video_path: str):
        """
        Initializes the video path and output directory.

        Args:
            video_path: Path to the input video file.
        """
        self.video_path = video_path
        
        self.output_dir = os.path.join(os.getcwd(), 'frames')
        self.num_threads = 16 if os.cpu_count() > 16 else os.cpu_count()
        self.VIDEO_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"]

    def extract_frames_multithreaded(self, video_path: str, output_dir: str, num_threads: int = None):
        """
        Extracts frames from a video using ffmpeg with multithreading on CPU cores.

        Note that the frames will be saved in a directory with the same name as the
        video file, e.g. `input_video.mp4` will result in a `input_video` directory
        containing the extracted frames.

        Args:
            video_path: Path to the input video file.
            output_dir: Directory to store the extracted frames.
            num_threads: Number of threads to use for parallel processing (default:
                number of CPU cores).

        Raises:
            ValueError: If the `output_dir` does not exist and cannot be created.
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.num_threads = num_threads if num_threads else self.num_threads
        self.file_name = os.path.basename(self.video_path).split('.')[0]
        print("file name",self.file_name)
        if not num_threads:
            num_threads = self.num_threads

        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                raise ValueError(f"Could not create output directory {output_dir}: {e}")

        # Get original video resolution
        command = ["ffprobe", "-v", "error", "-show_format", "-show_streams", video_path]
        probe_process = Popen(command, stdout=PIPE, stderr=PIPE)
        output, _ = probe_process.communicate()
        output_str = output.decode('utf-8')

        # Find video width and height from ffprobe output
        width = None
        height = None
        for line in output_str.splitlines():
            if line.startswith("width="):
                width = int(line.split("=")[1])
            elif line.startswith("height="):
                height = int(line.split("=")[1])
            elif line.startswith("bit_rate="):
                bit_rate = int(line.split("=")[1])
        if not width or not height:
            raise ValueError("Failed to determine video resolution using ffprobe.")
        # Construct ffmpeg command with multithreading
        command = [
            "ffmpeg",
            "-i", video_path,
            "-threads", str(num_threads),
             "-loglevel", "error",  # Suppress all output except for error messages #Did not test this one
            # "-vf", f"scale={width}:{height}",  # Scale to maintain original width
            # "-b:v", f"{bit_rate}",
            f"{os.path.join(output_dir,self.file_name,'frames')}/%05d.jpg"  # Output format with padding
        ]
        # print(" ".join([w for w in command]))
        process = Popen(command)
        process.wait()

        print(f"Finished extracting frames with {num_threads} threads. Frames saved to {output_dir}")

    def extract_batches(self, folder_path: str):
        """
        Takes a folder of videos and extracts frames from each video in the folder.

        Args:
            folder_path: Path to directory containing videos to extract frames from.

        Raises:
            ValueError: If any of the videos in the `folder_path` cannot be extracted
                frames from.
        """
        videos = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(tuple(self.VIDEO_EXTENSIONS))]

        for video in videos:
            try:
                self.extract_frames_multithreaded(video_path = video,output_dir= folder_path)
            except ValueError as e:
                raise ValueError(f"{video} could not be extracted. {e}") from None



if __name__ == "__main__": 
    url = 'https://youtu.be/T5oxMXX2pPY?si=TqdX-NSxObrA-upl'
    downloader = Downloader(url)
    # downloader.download('high', downloader.output_dir)
    # print(f"Downloaded Successfully to {downloader.output_dir}")
    extractor = vid2frames(downloader.output_dir)
    extractor.extract_batches(downloader.output_dir)
    extractor.extract_frames_multithreaded(downloader.output_dir, downloader.output_dir)

    