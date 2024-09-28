# Yelp Rating Predictions - Hybrid Recommendation System via PySpark
This project implements a hybrid recommendation system utilizing a combination of user and business metadata from Yelp datasets. By switching from XGBoost to CatBoost, I improved the model's RMSE from 0.9824 to 0.9791. The system employs collaborative filtering and regression techniques to enhance prediction accuracy efficiently. 

> This repository contains the implementation of a hybrid recommendation system developed for a Yelp dataset competition for my course DSCI 553. The project aims to improve the prediction accuracy and efficiency of a recommendation system using a combination of collaborative filtering and model-based approaches.

## Introduction
In this project, I developed a hybrid recommendation system that combines item-based collaborative filtering with a machine learning model (CatBoost) to predict ratings for user-business pairs. The system is built to handle a large-scale dataset from Yelp and leverages both user and business metadata for better recommendations.

## Features
The recommendation system uses the following features extracted from the dataset to train the model:
1.**user.json:** review_count, average_stars, useful, funny, cool, fans, compliment_hot, compliment_more, compliment_profile, compliment_cute, compliment_list, compliment_note, compliment_plain, compliment_cool, compliment_funny, compliment_writer, compliment_photos, yelping_since
2. **business.json:** review_count, stars, is_open, RestaurantsPriceRange2, latitude, longitude
3. **review_train.json:** useful, funny, cool, date
4. **checkin.json:** time values summed
5. **tip.json:** count, date
6. **photo.json:** count

## Data Sources
The dataset consists of several files with various aspects of user and business information from Yelp:
1. **user.json:** Contains metadata about users, such as review count, average rating, useful/funny/cool votes, and compliments.
2. **business.json:** Contains metadata about businesses, such as review count, average rating, price range, and location (latitude/longitude).
3. **review_train.json:** Contains user reviews with useful/funny/cool votes and review dates.
4. **checkin.json:** Contains the check-in times per business.
5. **photo.json:** Contains the count of photos associated with a business.
6. **tip.json:** Contains tips left by users and their dates.

## Model Architecture
The project implements a hybrid recommendation system with two key components:
1. **Collaborative Filtering (Item-Item):** We use a co-rated item to item collaborative filtering technique based on the Pearson correlation coefficient. This method predicts a rating based on the similarity between items rated by the same user.
2. **Model-Based Approach (CatBoost):** A machine learning model (CatBoostRegressor) is trained on user, business, and review features. CatBoost was chosen due to its high efficiency and ability to handle categorical features, providing better accuracy and lower RMSE than traditional models like XGBoost.

## Hybrid Approach
The system improves prediction accuracy by blending item-based collaborative filtering and model-based predictions. The collaborative filtering component predicts ratings based on item similarity using the Pearson correlation coefficient. The model-based approach, trained with CatBoost, uses user and business metadata for prediction. The final rating is a weighted combination of both predictions:

> Final Prediction = α × Item-Based Prediction + (1 − α) × Model-Based Prediction
_Where α is a weighting factor (set to 0.06 in this project)._

  ### Why Hybrid?
  1. Item-Based Filtering captures user preferences for specific item similarities.
  2. Model-Based Approach leverages more global user and business features.
  3. Combining the two enhances the system's capability to recommend more personalized and accurate ratings.

## Evaluation Metrics
The system's performance is evaluated using two key metrics:
1. Root Mean Squared Error (RMSE): Measures the difference between predicted ratings and actual ratings.
$`\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\text{Pred}_i - \text{Rate}_i)^2}`$


1. Error Distribution: Shows the absolute difference between predicted and actual ratings, divided into 5 bins:
```
>= 0 and < 1
>= 1 and < 2
>= 2 and < 3
>= 3 and < 4
>= 4
```

## Performance
```
Validation RMSE: 0.9791
Test RMSE: 0.9773
Error Distribution:
>=0 and <1: 102,058 predictions
>=1 and <2: 33,045 predictions
>=2 and <3: 6,158 predictions
>=3 and <4: 780 predictions
>=4: 3 predictions
```

## Setup and Usage
### Prerequisites
To run the project, you'll need the following:
```
Python 3.6
Spark 3.1.2
Scala 2.12
JDK 1.8
Required Python Libraries:
pyspark
catboost
scikit-learn
numpy
```

## Results and Insights
### Key Findings:
1. Incorporating business and user metadata improved prediction accuracy.
2. Transitioning from XGBoost to CatBoost resulted in a slight yet meaningful reduction in RMSE.
3. The hybrid model performs well on large datasets, making it suitable for real-world recommendation systems with millions of data points.
   
### Future Work:
1. Integrate additional features from checkin.json and tip.json to further enhance predictions.
2. Experiment with deep learning models for feature extraction, such as autoencoders, to improve collaborative filtering.
