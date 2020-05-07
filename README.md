# Disaster-Response-Pipeline

This project is a Udacity Nanodegree project for Data Science. It is to classify disaster response messages through machine learning.

# Content

In this repository I have three main folders and two screenshots.

The Folders are :

Data

Apps

Models

**Data:**     
In this folder I have the data set used for this project,the database and also my python script.

     processor.py : reads in the data, cleans and stores it in a SQL database. 
  
     disaster_categories.csv and disaster_messages.csv : The datasets that were used in the project.
     
     DisasterResponse.db: The created database from transformed and cleaned data.

**Apps:**
In this folder I have a python script for running the web app and web templates.

     run.py : Flask app and the user interface used to predict results and display them.
     
     templates: folder containing the html templates
     
     
 **Models:** 
 In this folder I have the python script I used to build the model and also the classifier.
 
    train_classifier.py: includes the code necessary to load data, transform it using natural language processing, run a machine learning model using GridSearchCV and train it. 
    
    classifier.pkl: This is the saved model that can be reused and also deployed.
    
    
**Screenshots of the Web App:**

![picture alt](http://via.placeholder.com/200x150 "Title is optional")
  
  
  
  
