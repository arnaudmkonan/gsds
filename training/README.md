# README Training

The general strategy is to get a feel for our model accuracies and the variation we expect of this accuracy on new data.
To do this we will explore the following

1. Dependence on cleaned vs original labels
2. Dependence on seed for model initialization and mini-batching
3. Dependence on train-validation split
4. Dependence on data used in training (tested via cross validation)


## To run training

Remember to first activate environment (see Install.md in repository base)

```
source activate gsv
cd training
```

### Training with one validation set

Now start training:
```
python lib/train.py
```

See args in train.py to change default values or type:

```bash
python lib/train.py --help

optional arguments:
  -h, --help            show this help message and exit
  --bsize BSIZE         mini batch size, lower if have memory issues
  --learning_rate LEARNING_RATE
                        learning rate
  --lrs LRS             learning rate step decay, ie how many epochs to weight
                        before decaying rate
  --lrsg LRSG           learning rate step decay factor,gamma decay rate
  --L2 L2               L2 weight decay
  --num_epochs NUM_EPOCHS
                        number of epochs
  --random_seed RANDOM_SEED
                        use random seed, use 0 for false, 1 for generate, and
                        more than 2 to seed
  --model_type MODEL_TYPE
                        retrain or finetune
  --train_dir TRAIN_DIR
                        train directory in data root
  --model_dir MODEL_DIR
                        model directory
  --val_dir VAL_DIR     validation directory in data root
  --data_dir DATA_DIR   data directory
```


To train with a different learning rate and training data (note the train3 and val3 directories need to
be prepared according to DataPrep.md):

```
python lib/train.py --learning_rate 0.002 --train_dir 'train3' --val_dir 'val3'
```


### Cross Validation 


To use default parameters
```
python lib/train_cv.py
```

```bash
optional arguments:
  -h, --help            show this help message and exit
  --bsize BSIZE         mini batch size, lower if have memory issues
  --learning_rate LEARNING_RATE
                        learning rate
  --lrs LRS             learning rate step decay, ie how many epochs to weight
                        before decaying rate
  --lrsg LRSG           learning rate step decay factor,gamma decay rate
  --L2 L2               L2 weight decay
  --num_epochs NUM_EPOCHS
                        number of epochs
  --num_folds NUM_FOLDS
                        number of CV folds
  --label_file LABEL_FILE
                        csv file for labels
  --random_seed RANDOM_SEED
                        use random seed, use 0 for false, 1 for generate, and
                        more than 2 to seed
  --model_type MODEL_TYPE
                        retrain or finetune
  --model_dir MODEL_DIR
                        model directory
  --data_dir DATA_DIR   data directory
```

To do a 12 fold cross validation:
```
python lib/train_cv.py --num_folds 12
```





