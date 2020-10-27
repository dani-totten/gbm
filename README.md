# Predicting Housing Prices with Gradient Boosting Machines

- Data from Kaggle of 19,933 housing prices in Seattle area
- 3/4 training, 1/4 test split
- Grew 5000 trees based on 14 predictors
- Interaction depth of 1, all trees are stumps and no variable interactions are allowed
- Learning rates of 1, 0.1, 0.01 to look for overfitting
- Best predictions came from lambda = 0.1


# Gradient Boosting Machines
GBMs are both interpretable and have strong predictive power. They achieve this by using adaptive boosting and negative gradient loss optimization to create an ensemble of weak learners that improve on the predictive power of decision trees. I looked at the impact of one particular parameter - lambda. This parameter is a multiplier of the adjustment that the boosting process takes. In addition, lambda is also be referred to as the “learning rate” because it dictates the pace at which the model will descend to either the global or local minimum of the loss function. Lambda varies from 0 to 1 with higher values indicating "faster" learning through large adjustments and lower values indicating "slower" learning through smaller adjustments. Lower shrinkage rates are generally preferable, but come at a cost of high computation times.

# Data
Data came from Kaggle with an initial 39 variables and 49k+ observations. I took out unnecessary variables, duplicate variables and non-standardized character descriptions. I removed data from before 2014 to cut down on the size of the data file and houses costing over $5 million to reduce skew. I wound up with 10 continuous variables, 3 binary variables and one multinomial factor. These covariates predicted price, a continuous variable. There was correlation between some of the variables but, since GBMs are non-parametric and don't need to satisfy assumptions around the correlation structure of the predictor space, I kept them all in. Any correlation between observations was ignored, even though neighborhood and adjacent home prices are likely correlated to a home's price.

![](https://github.com/dani-totten/gbm/blob/main/corr_gbm.png)

There was high skew in housing prices, but as a non-parametric model there are no assumptions around the distribution of the data, so I didn't do a transformation.

![](https://github.com/dani-totten/gbm/blob/main/price_density.png)

# Model
I ran three models with 5k trees each at learning rates of 1, 0.1, and 0.01 on all 14 predictors. I used RSS for loss, error and negative gradient. Since the model is greedy and optimizes reduction in MSE, the variable importance can be thought of as the proportion of MSE reduction caused by splits at a given variable.

![](https://github.com/dani-totten/gbm/blob/main/relative_influence.png)

# Lambda
The faster learning, lambda = 1, showed a fast reduction in MSE and the GBM reached its minimum MSE at around 200 trees before beginning to increase again. The medium learner, lambda = 0.1, had the lowest MSE and reached its minimum at 1700 trees. The slowest learner, lambda = 0.01, didn't reach its minumum in 5k trees and was still decreasing. The MSE by number of trees for each learning is shown below.

![](https://github.com/dani-totten/gbm/blob/main/mse_red_learning_rate.png)

Looking at higher numbers of trees, you can more clearly see the differences in the learning rates. The MSE for the model with the high learning rate is jumpy and increasing, the middle learning rate stays steady and the slowest learner asymptotically decreasing.

![](https://github.com/dani-totten/gbm/blob/main/MSE_higher_trees.png)
