#!/bin/bash
# Start API server with reload, but exclude demo files and temporal_example.py
# This prevents the server from restarting when you edit demo/test files

uvicorn app.main:app --reload \
  --reload-exclude 'demo/*' \
  --reload-exclude 'temporal_example.py' \
  --reload-exclude 'tests/*'
