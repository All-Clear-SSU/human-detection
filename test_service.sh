#!/bin/bash

# Run Server First
curl -X POST "http://0.0.0.0:8383/predict?conf_threshold=0.5" \
  -F "file=@human_cctv.png"


curl -X POST "http://0.0.0.0:8383/predict_image" \
  -F "file=@human_cctv.png" \
  --output output.jpg

curl -X POST "http://0.0.0.0:8383/predict_with_pose?conf_threshold=0.5&pose_conf_threshold=0.3" \
 -F "file=@human_cctv.png"


curl -X POST "http://0.0.0.0:8383/predict_image_with_pose" \
 -F "file=@human_cctv.png" \
 --output output_with_pose.jpg
