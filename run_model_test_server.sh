#!/bin/bash

# 0.0.0.0 port 8383
nohup uvicorn src.service:app --host 0.0.0.0 --port 8383 > server.log 2>&1 &

