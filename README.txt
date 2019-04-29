I implemented several ensemble classifiers using Python’s Sklearn library and applied it to a Blood Transfusion dataset from the UC Irvine Machine Learning Repository. The dataset contains 748 donor data and has 4 features: R (Recency - months since last donation), F (Frequency - total number of donation), M (Monetary - total blood donated in c.c.), T (Time - months since first donation), and a binary variable representing whether he/she donated blood in March 2007 (1 stand for donating blood; 0 stands for not donating blood).

The program ensemble_learning.py must be run with python3 (it was made with and tested with Python 3.5.2 on ubuntu). It covers all three tasks of Lab4. It uses all CPU cores available and takes about 1.5 minutes to run.

Install dependencies: pip install -r requirements.txt 

usage:type into terminal: python3 ensemble_learning.py lab4-train.csv lab4-test.csv


the csv files that contain the data are provided in the submission folder
