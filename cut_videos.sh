
input="videos/10 Extreme Moments Caught on Security Camera.mp4"
ffmpeg -i "$input" -ss 00:10:25 -to 00:11:30 -c copy videos/3_segment.mp4

