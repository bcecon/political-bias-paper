# Predicting Political Bias in News Media

The overall goal of this project was to determine whether a neural network could predict either liberal or conservative bias in a news article. Two news sources were chosen as proxies or representatives of liberal and conservative bias, the Huffington Post and Fox News respectively. The text of these articles were then converted into TF-IDF vectors for use as input for a multilayer perceptron, as well as a support vector machine to serve as a comparison model. Cross-validation was used to obtain performance estimates as well as tune their hyperparameters. The differences between their performance estimates was then tested for statistical significance.

The final paper submitted as a writing sample for my PhD applications can be found [here](http://briannadardin.com/Predicting-Political-Bias-in-News-Media.pdf). It goes into detail about the methods used and the decisions made in the process, as well as analyzes the results. This repository stores all the code I used in the development of this paper. 

## File Descriptions

**Create Database.sql:** This simple SQL file creates the database used to store all the data.

**Create Tables.sql:** This SQL file creates the tables representing the articles, the n-grams, and the combinations of articles and n-grams. 

**web-scraping.py:** This Python file scrapes the Huffington Post and Fox News websites for political news articles and adds them to the database.

**data-processing.py:** This Python file calculates all the TF-IDF scores and creates the term-document matrix used to train the models.

**model-training.py:** This Python file executes the cross-validation and grid search procedure and conducts the paired difference _t_ test.