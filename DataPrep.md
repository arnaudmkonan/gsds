# Data Preparation

The general idea is to not create any new files. We separate out files to their respective classes. A test set is created
and then from there all holdout sets are done via links to the original files. Training and validation holdouts are created
via the following bash scripts. 

Cross-validation is done directly in python.

Installation requirements:

* dos2unix
* shuf

```
mkdir data
mv practical_set.zip data
```

1. First create the test holdout, with 50 examples from each class removed from overall train data:
```
bash utils/create_testset.sh 50
```
This will remove 100 files for a test set. The importance of this size is discussed in the data exploration jupyter notebook.

2. Create corrected labels for training (if want to run cross validation)
```
bash utils/create_training_labels.sh 
```

3. To create 80:20 train-validation sets:

```
bash utils/create_train_validation_links.sh 5
```

4. To create a 1/3  validation holdout:
```
bash utils/create_train_validation_links.sh 3
```

Note can also use `utils/create_train_validation.sh` script if want to copy data and not make links


Data exploration see directory `data_exploration`

