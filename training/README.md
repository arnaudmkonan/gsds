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
```

Now start training:
```
python lib/train.py
```

See args in train.py to change default values



