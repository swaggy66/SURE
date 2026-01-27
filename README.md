# SURE

This repository is the implementation for the paper at ICASSP 2026: **SURE: SYNERGISTIC UNCERTAINTY-AWARE REASONING FOR MULTIMODAL EMOTION RECOGNITION IN CONVERSATIONS** 

## Install

Clone this git repository and change the directory to this repository:

```
pip install -r requirements.txt
```

## Quick Start

1. The datasets used in our experiments have been uploaded to (https://github.com/butterfliesss/SDT). You can download them and put them into data/. If you wish to use a different dataset, please ensure that it is processed into a similar format.

2. Run the model on the IEMOCAP/MELD dataset as follows:

```
bash run.sh
```

Once the above commands are executed, the evaluation results for IEMOCAP and MELD will be saved as sure_iemocap.txt/sure_meld.txt.
