# Distributed Systems - Final Project

## Running the project
```
docker run -it -v $(pwd):/mnt --network=host pytorch/pytorch:latest
torchrun --standalone main.py
```