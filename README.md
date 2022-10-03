
# Audio-based gender recognition

This was a collage project. 

We extract features for every record and create dataset in the schema 
```audio-based features --> label M or F```. 
No Deep Learning.

The term *gender* recognition is unfortunately inaccurate. Every speaker in Librispeech 
dataset has `M` or `F` label assigned. `M` voice is considered to have _male_ 
characteristics and `F` voice has _female_ characteristics. 
We follow this notation focusing only on technical aspects of classification. 
The voice timbre may be very misleading and its subjective 
perception is not the area of focus of this project. 

### Installation

Simple deps:

`pip install -r requirements.txt`


### Running

Download and unpack Librispeech data directly in `data/raw/` folder so that the data with `SPEAKERS.txt`, `BOOKS.txt` etc. files is in 
`data/raw/LibriSpeech/`.

Project is based only on clean files (train-100, train-360, dev and test).

Run:

```python extract_features```


### Results

Trained on signals from `train-clean-100` set, tested on signals from `dev-clean`.

Dev f1 score: `0.8861251457442676`
