# LLM guided music evolution

This code was used for the paper "Using Large Language Models as Fitness Functions in Evolutionary Algorithms for Music Generation" (accepted for AIMC'25).

The code was tested on Ubuntu 20.04, 22.04, and 24.04 and Debian 11, and 12.\
For the most stable python environment, we recommend cloning ours by using conda as shown below.

## Recommended setup and test run (under linux with conda)

Setup the project:
```
bash setup.sh
conda env create -f environment.yml
```

Test the program:
```
bash run.sh --help
bash run.sh --test-mode
```

## Pre-trained musical LLM models

In the current stage of development, this project uses the CLaMP model (v1) as fitness function for the evolutionary algorithm:\
<https://github.com/microsoft/muzic/tree/main/clamp>
