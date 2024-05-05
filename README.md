# Head Pose Estimation Project




https://github.com/Ibrahimmustafa1/HeadPoseEstimation/assets/85252957/371cc723-1c30-4080-978f-edb5cbab3a5f




## Overview

This project focuses on head pose estimation using the AFLW2000 dataset. The aim is to predict the yaw, pitch, and roll angles of a person's head from images or videos. I leverage the Mediapipe library for extracting facial landmarks and utilize machine learning models to predict head poses.

## Dataset

I have used the AFLW2000 dataset, which contains images of faces along with annotations for head poses. This dataset is widely used for head pose estimation tasks and provides a diverse range of facial orientations and expressions.

## Methodology

1. **Data Preprocessing**: I preprocess the AFLW2000 dataset to extract relevant features and annotations for training my models.

2. **Feature Extraction**: Facial landmarks are extracted using the Mediapipe library, which provides robust and accurate landmarks detection.

3. **Model Training**: I employ machine learning models to predict the yaw, pitch, and roll angles based on the extracted facial landmarks. I explore various models such as Support Vector Regression (SVR), Random Forest, and Gradient Boosting.

4. **Hyperparameter Tuning**: Using techniques like grid search or randomized search, I optimize the hyperparameters of my models to improve performance.

5. **Model Evaluation**: I evaluate the trained models using metrics such as Root Mean Squared Error (RMSE) to assess their accuracy in predicting head poses.

6. **Model Selection**: After rigorous evaluation, I identify the SVR model as the best-performing model for head pose estimation.

7. **Integration with Flask API**: The selected SVR model is integrated into a Flask API, allowing users to interact with the model through HTTP requests.

8. **Angular Web Interface**: I provide a user-friendly web interface built with Angular where users can upload videos and visualize the predicted head poses overlaid on the video.
