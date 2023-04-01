import os

def download(video_link):
    vid = video_link.split('?v=')[1]
    down_video = " ".join([
        "yt-dlp",
        '-f', 'mp4',
        video_link,
        '--output',
        './data/videos/' + vid + '.mp4'
    ])
    status = os.system(down_video)

video_list = open(os.path.join('data', 'video_links.txt')).read().strip().split('\n')
os.makedirs(os.path.join('data', 'videos'), exist_ok=True)
for video in video_list:
    download(video)
