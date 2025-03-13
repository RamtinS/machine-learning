# Regression Models  

## What is Regression?  

Regression is used to model the relationship between input features and a continuous target variable. It helps in predicting numerical values based on patterns in the data.  

Common types of regression:  
- **Linear Regression**: Models the relationship between variables with a straight line.  
- **Non-Linear Regression**: Captures more complex relationships using curved functions.  

## Task Description  

The datasets contain observations of newborns. The goal is to create regression models to predict various attributes.  

### Subtasks:  
1. **Linear Regression in 2D**  
   - Predict weight based on length using `length_weight.csv`.  
   - Visualize the model with observations and report the loss.  

2. **Linear Regression in 3D**  
   - Predict age (in days) based on length and weight using `day_length_weight.csv`.  
   - Use 3D plotting to visualize the model and report the loss.  

3. **Non-Linear Regression in 2D**  
   - Predict head circumference based on age (in days) using `day_head_circumference.csv`.  
   - Use the model: `f(x) = 20σ(xW + b) + 31`, where `σ` is the sigmoid function.  
   - Visualize the model with observations and report the loss.  
