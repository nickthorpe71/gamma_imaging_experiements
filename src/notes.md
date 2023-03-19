# Gamma Hadron Classification

Using the magic04 dataset we are going to train a model to be able to classify whether a particle is a gamma or hadron particle given the selected features.

## Notes
- since the data is labeled this is an example of supervised learning
- this is an example of a classification problem

### Features
AKA column headers

#### Qualitative Data
- qualitiative data is categorical meaning it has a finite number of categories or groups
- within qualitative data there is nominal data and ordinal data
- nominal data means there is no inherent order
- ordinal data means they have an inherent order

#### Quantitative Data
- quantitative data is numerical and could be discrete or continuous
- discrete meaning data that only takes certain values
- continuous meaning data that have an order

## Classification 
- classification means we are trying to predict discrete classes
- generally there is binary and multi-class classification
- binary ex: sentiment(pos/neg),
- multi-class ex: types of animals(dog/cat/boar/shark/etc.)

## Regression
- used to predict continuous values (a number on some sort of scale)
- ex: stock market, housing prices

## Data Prep

### Oversampling
- means to generate more of an underrepresended class of the data
    - ex: if there are 100 instances of the X class and 200 of the Y class then the oversampler will generate more data for X


## K-Nearest Neighbors
- the concept is: given a set of labeled data points, you can plot a new data point and try to guess it's label using the labels of surrounding (neighbor) data points
- we use a distance function (ex: eudlidean distance) 
    - euclidean distance is basically drawing a straight line from one plot point to another
    - euclidian distance formula: distance = √((x1 - x2)^2 + (y1 - y2)^2)
        - the distance function can be extended beyond 2 dimensions depending on the number of features
- the K is simply how many of the closest neighbors to consider

## Naive Bayes
- bayes' rule = the probability that a happened given b happened
    - notation: p(A|B) = p(B|A) * p(A) / p(B)
- naive bayes expands bayes' rule and applies it to classification
- the naive part comes from the assumption that all the features are independant 