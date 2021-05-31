# ScoringSumm
Code for our paper:
"Improving Abstractive Summarization with Summary Scoring"

## 1. How to Install

### Requirements
- `python3`
- `conda create --name env --file spec-file.txt`
- `pip3 install -r requirements.txt`

### Description of Codes
- `main.py` -> training and evaluation procedure
- `model.py` -> models
- `data_utils.py` -> dataloader
- `utils.py` -> utility functions
- `preprocess.py` -> data preprocessing

### Workspace
Following directories should be created for our experiments.
- `./cache` -> storing model checkpoints
- `./result` -> storing evaluation results

## 2. Preprocessing
We use the following datasets for our experiments.

- CNN/DailyMail -> https://github.com/abisee/cnn-dailymail
- XSum -> https://github.com/EdinburghNLP/XSum

For data preprocessing, please run
```
python preprocess.py --src_dir [path of the raw data] --tgt_dir [output path] --split [train/val/test] --cand_num [number of candidate summaries]
```
`src_dir` should contain the following files (using test split as an example):
- `test.source`
- `test.source.tokenized`
- `test.target`
- `test.target.tokenized`
- `test.out`
- `test.out.tokenized`

Each line of these files should contain a sample. In particular, you should put the candidate summaries for one data sample at neighboring lines in `test.out` and `test.out.tokenized`.

The preprocessing precedure will store the processed data as seperate json files in `tgt_dir`.

We have provided an example file in `./example`.

## 3. How to Run

### Hyper-parameter Setting
You may specify the hyper-parameters in `main.py`.
### Train
```
python main.py --cuda --gpuid [list of gpuid] -l
```
### Fine-tune
```
python main.py --cuda --gpuid [list of gpuid] -l --model_pt [model path]
```
### Evaluate
```
python main.py --cuda --gpuid [single gpu] -e --model_pt [model path]
```


## 4. Results

### CNNDM
|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| BART     | 44.39   | 21.21   | 41.28   |
| Ours     | 46.67   | 22.15   | 43.54   |

### XSum
|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| Pegasus  | 47.10   | 24.53   | 39.23   |
| Ours     | 47.61   | 24.57   | 39.44   |

Our model outputs on these datasets can be found in `./output`.
