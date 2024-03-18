import youtube_dl

def download_video(url, download_path):
    ydl_opts = {
        'outtmpl': download_path + '/%(title)s.%(ext)s',
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# take current directory as download path
download_path = './'

# Use the function to download a video
download_video('https://www.youtube.com/watch?v=3AQ7yC5jQ28',download_path=download_path)