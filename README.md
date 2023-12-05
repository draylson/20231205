# Movie Genre Pipeline Project

## Overview

### Summary:

This project work on a data set containing real movie descriptions and genres. A machine learning pipeline was created to categorize movies by genre based on their descriptions so that you can better categorize your movies or check if your description for them really conveys to readers what kind of movie they are. The project includes a web app where users can input a movie description and get classification results in several genres.

### Instructions:

1. Optionally, run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### File Descriptions:

* `process_data.py` - Script to load, clean and save disaster response data.
* `train_classifier.py` - Script to build and train diaster message classifier.
* `run.py` - Script to run a web app which classifies users messages. 
* `imdb_raw.csv` - File containing raw movies data.
* `master.html` - Web app main page.
* `go.html` - Web app page to display model results.
* `IMDB.db` - Database generated by `process_data.py` script. 
* `classifier.pkl` - Model generated by `train_classifier.py` script.

### Installation

This code runs with Python 3.* and requires some libraries, to install these libraries you will need to execute:

    pip install -r requirements.txt

### Libraries used

* flask==2.2.2
* joblib==1.2.0
* nltk==3.8.1
* numpy==1.24.3
* pandas==2.0.3
* plotly==5.9.0
* scikit-learn==1.3.0
* sqlalchemy==1.4.39

### Licensing, Authors, Acknowledgements

This project uses as a base the code resources provided for the `Project: Disaster Response Pipeline` module of the `Data Scientist` course at Udacity. So, we must give credit to Udacity for the code template. The licensing information for the Udacity's Educational Content by clicking on the following link: [Udacity's Terms of Use](https://www.udacity.com/en-US/legal/terms-of-use).

Must also give credit to IMDB for the data. The licensing information for the IMDB can be found by clicking on the following link: [Can I use IMDb data in my software?](https://help.imdb.com/article/imdb/general-information/can-i-use-imdb-data-in-my-software/G5JTRESSHJBBHTGX?pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=1ed1aea6-d2ad-4705-95fd-ba13f1b5014f&pf_rd_r=XRE3QWF2G5YWTD2SGT0V&pf_rd_s=center-1&pf_rd_t=60601&pf_rd_i=interfaces&ref_=fea_mn_lk1#).

Otherwise, feel free to use the code and data here as you would like!

## Project Definition, Analysis, and Conclusion

This project was designed to meet the requirements of Udacity's `Data Scientist` course. In this section, the information required in the `Data Scientist Capstone` project rubric is provided.

### Project Definition

The project definition is provided below.

#### Project Overview

This project work on a data set containing real movie synopsis and genres. A machine learning pipeline was created to categorize movies by genre based on their descriptions so that you can better categorize your movies or check if your description for them really conveys to readers what kind of movie they are. The project includes a web app where users can input a movie description and get classification results in several genres. The data was obtained through web scraping pages from the IMDB website in a previous project carried out for a Udacity course.

#### Problem Statement

Determine the genre of a movie from its synopsis. The problem in question fits into a [classification problem](https://www.kdnuggets.com/2022/09/top-5-machine-learning-practices-recommended-experts.html). Thus, the strategy used was to train machine learning to classify the synopses between the various movie genres.

To do this, we use [`scikit-learn`](https://scikit-learn.org/stable/index.html), a Python module for machine learning. The [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) method was used to find an algorithm (and respective parameters) that provided better results.

Since a movie can be classified into more than one genre and, for each genre, it can only assume two values (belongs to that genre or doesn't belong), this is a [multilabel classification problem](https://scikit-learn.org/stable/modules/multiclass.html#multioutputclassifier). Although `scikit-learn` provides nine algorithms that support multilabel classification, to reduce the computational cost involved, those that also support multiclass-multioutput classification problems were selected: `DecisionTreeClassifier`, `ExtraTreeClassifier`, `ExtraTreesClassifier`, `KNeighborsClassifier`, `RadiusNeighborsClassifier`, `RandomForestClassifier`.

Thus, the classifier presenting the best score is used in the implementation of a web app, where users are able to enter a synopsis of a story and the app provides the movie genres that best fit the synopsis provided.

#### Metrics

The main metric used was the recall score, as it is intuitively the ability of the classifier to find all the positive samples.

F1-score and precision score were also considered. But trying to be accurate, the classifiers tended not to provide genres for the input texts, as they avoided false positives.

Thus, the recall score metric proved to be more appropriate. Although the classifier may suggest a not very suitable genre, it tends to provide more true positives.

### Analysis

The data analysis was performed through the `data/analysis.ipynb` file. A summary is described below.

#### Data Exploration

The dataset has a total of 2467 movie samples and their respective data. Among these, we highlight the description (synopsis) and the genres in which they were classified.

Some of these samples contained `null` values for description and/or gender. Thus, 2335 samples remain.

Since a movie can be classified into more than one genre. Genre data consists of a list in JSON format, which is processed and transformed into columns with binary values.

The distribution of movies by genre was calculated. Through this distribution it was possible to realize that the dataset presents a large difference in the number of samples for each movie genre. For example, out of 2334 samples, we only have one movie classified as `Western`.

The genre with the largest number of samples is `Drama` (1065 samples), which has almost twice as many samples as Comedy (553 samples), the second genre with the largest number of samples.

#### Data Visualization

Two visualizations are plotted in the `data/analysis.ipynb` file. A version of one of them is also present on the web app's home page.

### Methodology

The methodology is described below.

#### Data Preprocessing

A preview of data preprocessing can be found in the `data/analysis.ipynb` file. However, the final preprocessing is implemented and documented in the `data/process_data.py` file.

#### Implementation

The `models/train_classifier.py` script builds a pipeline that processes text and then performs multi-output classification on the movie genres in the dataset. `GridSearchCV` is used to find the best parameters for the model. The final classifier is found in the `models/classifier.pkl` file.

In the final version, the DecisionTreeClassifier, ExtraTreeClassifier, KNeighborsClassifier, RadiusNeighborsClassifier, ExtraTreesClassifier, RandomForestClassifier estimators are considered. Three different values are considered for the parameters `n_neighbor`, `radius` and `n_estimators`.

Scores are calculated with the `recall score` metric, creating a global average by counting the total true positives, false negatives and false positives. Divisions by zero are handled by assigning the value `0` as the result.

Furthermore, the cross-validation splitting strategy used was StratifiedKFold, considering a number of 3 folds.

Among the complications that occurred during the coding process, I can mention the computational cost to perform cross-validation with multiple values in different parameters. Identifying the parameters that were really relevant was a challenge.

Identifying suitable values for various parameters also required efforts. This includes choosing metrics and their parameters, radius values for RadiusNeighborsClassifier, number of folds for StratifiedKFold, among others.

#### Refinement

Initially, I ran GridSearchCV with RandomForestClassifier, configuring three values for `n_estimators`, `criterion` and `max_features`, generating a classifier and a classification report.

Next, I ran GridSearchCV individually with the other estimators: DecisionTreeClassifier, ExtraTreeClassifier, KNeighborsClassifier, RadiusNeighborsClassifier, ExtraTreesClassifier. In a similar way, configuring three values for `n_estimators`, `criterion`, `max_features`, `n_neghbours`, `radius`, `weights` e `algorithms`, also generating a classifier and a classification report.

Noticing irrelevant parameters, that is, the default values were the best, these were disregarded. Others needed to have their values adjusted to mitigate the occurrence of errors. Thus, arriving at an ideal set of parameters and values for each estimator.

A GridSearchCV was configured including the six estimators with the parameters and values identified as appropriate and I began to observe the score metrics more closely.

After carrying out executions considering different score metrics (F1-score, precision, recall) and parameter values for them, it was possible to arrive at the most appropriate final version. The final configuration can be found in the file `models/train_classifier.py`.

### Results

The results are described below.

#### Model Evaluation and Validation

The table below presents the recall score for each combination of estimator and parameter values.

|                            | N_NEIGHBORS | RADIUS | N_ESTIMATORS |  FOLD | RECALL_SCORE | AVERAGE FOLD SCORE |
|----------------------------|:-----------:|:------:|:------------:|:-----:|:------------:|:------------------:|
| DecisionTreeClassifier     |      -      |    -   |       -      |   1   |    0.302     |       0.311        |
| **DecisionTreeClassifier** |    **-**    |  **-** |     **-**    | **2** |   **0.327**  |      **0.311**     |
| DecisionTreeClassifier     |      -      |    -   |       -      |   3   |    0.305     |       0.311        |
| ExtraTreeClassifier        |      -      |    -   |       -      |   1   |    0.223     |       0.240        |
| ExtraTreeClassifier        |      -      |    -   |       -      |   2   |    0.259     |       0.240        |
| ExtraTreeClassifier        |      -      |    -   |       -      |   3   |    0.238     |       0.240        |
| KNeighborsClassifier       |      1      |    -   |       -      |   1   |    0.291     |       0.296        |
| KNeighborsClassifier       |      1      |    -   |       -      |   2   |    0.313     |       0.296        |
| KNeighborsClassifier       |      1      |    -   |       -      |   3   |    0.284     |       0.296        |
| KNeighborsClassifier       |      5      |    -   |       -      |   1   |    0.208     |       0.205        |
| KNeighborsClassifier       |      5      |    -   |       -      |   2   |    0.205     |       0.205        |
| KNeighborsClassifier       |      5      |    -   |       -      |   3   |    0.202     |       0.205        |
| KNeighborsClassifier       |      10     |    -   |       -      |   1   |    0.147     |       0.138        |
| KNeighborsClassifier       |      10     |    -   |       -      |   2   |    0.126     |       0.138        |
| KNeighborsClassifier       |      10     |    -   |       -      |   3   |    0.142     |       0.138        |
| RadiusNeighborsClassifier  |      -      |    5   |       -      |   1   |    0.000     |       0.000        |
| RadiusNeighborsClassifier  |      -      |    5   |       -      |   2   |    0.000     |       0.000        |
| RadiusNeighborsClassifier  |      -      |    5   |       -      |   3   |    0.000     |       0.000        |
| RadiusNeighborsClassifier  |      -      |   10   |       -      |   1   |    0.000     |       0.000        |
| RadiusNeighborsClassifier  |      -      |   10   |       -      |   2   |    0.000     |       0.000        |
| RadiusNeighborsClassifier  |      -      |   10   |       -      |   3   |    0.000     |       0.000        |
| RadiusNeighborsClassifier  |      -      |   20   |       -      |   1   |    0.000     |       0.000        |
| RadiusNeighborsClassifier  |      -      |   20   |       -      |   2   |    0.000     |       0.000        |
| RadiusNeighborsClassifier  |      -      |   20   |       -      |   3   |    0.000     |       0.000        |
| ExtraTreesClassifier       |      -      |    -   |      50      |   1   |    0.136     |       0.141        |
| ExtraTreesClassifier       |      -      |    -   |      50      |   2   |    0.141     |       0.141        |
| ExtraTreesClassifier       |      -      |    -   |      50      |   3   |    0.145     |       0.141        |
| ExtraTreesClassifier       |      -      |    -   |      100     |   1   |    0.132     |       0.139        |
| ExtraTreesClassifier       |      -      |    -   |      100     |   2   |    0.141     |       0.139        |
| ExtraTreesClassifier       |      -      |    -   |      100     |   3   |    0.145     |       0.139        |
| ExtraTreesClassifier       |      -      |    -   |      200     |   1   |    0.143     |       0.145        |
| ExtraTreesClassifier       |      -      |    -   |      200     |   2   |    0.145     |       0.145        |
| ExtraTreesClassifier       |      -      |    -   |      200     |   3   |    0.147     |       0.145        |
| RandomForestClassifier     |      -      |    -   |      50      |   1   |    0.126     |       0.131        |
| RandomForestClassifier     |      -      |    -   |      50      |   2   |    0.124     |       0.131        |
| RandomForestClassifier     |      -      |    -   |      50      |   3   |    0.143     |       0.131        |
| RandomForestClassifier     |      -      |    -   |      100     |   1   |    0.140     |       0.144        |
| RandomForestClassifier     |      -      |    -   |      100     |   2   |    0.142     |       0.144        |
| RandomForestClassifier     |      -      |    -   |      100     |   3   |    0.151     |       0.144        |
| RandomForestClassifier     |      -      |    -   |      200     |   1   |    0.143     |       0.142        |
| RandomForestClassifier     |      -      |    -   |      200     |   2   |    0.138     |       0.142        |
| RandomForestClassifier     |      -      |    -   |      200     |   3   |    0.146     |       0.142        |

#### Justification

The best estimator was the `DecisionTreeClassifier` with default values for all its parameters, presenting the best score of `0.311`. However, the `KNeighborsClassifier` with `N_NEIGHBOR = 1` also presented scores close to the `DecisionTreeClassifier`. Thirdly, we have the `ExtraTreeClassifier` with default values for all its parameters.

The resulting scores are also stored in the `models/training_output.txt` and the `DecisionTreeClassifier` is stored in the `models/classifier.pkl` file. In de `app` folder, you can find the web app which uses the generated classifier to provide movie genre to the texts the users provide.

### Conclusion

Next, we conclude our achievements.

#### Reflection

In this project, a machine learning was trained for determine the genre of a movie from its synopsis. A `DecisionTreeClassifier` proved to be the best estimator for the problem, resulting in a better recall score. As a result, a web app is provided where users enter a synopsis of a story and the app provides the movie genres that best fit the synopsis provided.

Choosing metrics and parameter values to train a classifier has proven to be a challenge, as well as the computational cost to execute all possibilities. In addition to the parameters presented, others were also considered, such as: `criterion`, `max_features`, `weights` and `algorithm`, when applicable to each estimator. At least three values for each parameter were considered. After long hours of running `GridSearchCV`, it was possible to realize that the default values were the most appropriate for these parameters.

#### Improvement

The resulting classifier, in specific cases, can associate no genre for a synopsis. The use of `ClassifierChain`s with `One-Vs-The-Rest` estimators can be explored to mitigate this limitation.

Furthermore, the dataset has a small number of samples, which in turn are not balanced. Using a larger dataset to train the classifiers could provide better assistance to the classifiers.