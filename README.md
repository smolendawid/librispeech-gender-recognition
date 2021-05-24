
# Audio-based gender recognition

A collage project. We extract features for every record and create dataset features --> label. 

No Deep Larning.


### Installation

Simple deps:

`pip install -r requirements.txt`


### Running

Download and unpack Librispeech data directly in data/raw/ folder so that the data with SPEAKERS.txt BOOKS.txt etc is in 
data/raw/LibriSpeech/.

Project is based only on clean files (train-100, train-360, dev and test)

Run
`python extract_features`


### Results

Trained on signals from train-clean-100 set, tested on signals from dev-clean

Dev f1 score: `0.8861251457442676`
