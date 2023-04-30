# Distributed Systems - Final Project

## Running the project
```
docker run -it -v $(pwd):/mnt --network=host pytorchlightning/pytorch_lightning
python main_lightning.py fit --traineraccelerator cpu --trainer.max_epochs 5 --trainer.strategy ddp --trainer.deterministic True --trainer.num_nodes 2
```