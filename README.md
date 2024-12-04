# CS6350_Project
CS 5350/6350 Machine Learning, Fall 2024 - Projects

Three different methods are used to test the performance.

Different hyperparameter settings are used to find the best model (in [ ]). 

## rt_predict.py

```
Number of trees: n_estimators [300]
max depth: max_depth [20]
Minimum samples to split: min_samples_split [10]
Number of features for best split: max_features ['sqrt']
```

The prediction file is saved at `'./data/submission_predictions_rt.csv'`.

The Best Validation Auc: 0.9126.

## boost.py
```
Number of trees: n_estimators [400]
Step size: learning_rate [0.05]
Maximum depth: max_depth [5]
Subsample ratio : subsample [1.0]
Subsample ratio of columns: colsample_bytree [0.6]
Minimum loss reduction: gamma [0.1]
L1 regularization weights: reg_alpha [0.01]
L2 regularization weights: reg_lambda [1.0]
```

The prediction file is saved at `'./data/submission_predictions_boost.csv'`.

The Best Validation Auc: 0.9263.

## nn.py
```
hidden layer size: hidden_sizes_list [64, 32]
dropout rate: dropout_rates [0.1]
learning rate: learning_rates [0.005]
batch size: batch_sizes [128]
```

You may need to wait for a few hours until the model find the best parameters. 

The prediction file is saved at `'./data/submission_predictions_nn.csv'`.

The Best Validation Auc: 0.9118.