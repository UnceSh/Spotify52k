# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 21:21:47 2023

@author: Unce Shahid
"""

# 0.) Preprocessing / Initialization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from scipy.special import expit
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import random

# Seeding for replicability
random.seed(14715908)

# Read in the data file
file = pd.read_csv('spotify52kData.csv')

# Cleaning the data: 
    
    # The pruning method used below is row-wise removal of duplicate track_names, indices are reset to account for dropped data
noDupes = file.drop_duplicates(subset = "track_name").reset_index(drop = "True")

    # Now I select the columns of analysis (exclude the song number, artists, album, and track names)  
df = noDupes.iloc[:, 4:]

    # Effective Data Analysis (EDA): Obtaining a general idea of how the data looks
EDA = df.describe()
dfCor = df.corr(numeric_only = "True") # None of the predictors have even a slightly high correlation with popularity

# Question  1
    # Consider the 10 song features duration, danceability, energy, loudness, speechiness, acousticness, instrumentalness, 
    # liveness, valence and tempo. Are any of these features reasonably distributed normally? If so, which one?

# Plot each of the 10 features' distribution

# Plot 1: Duration 
plt.subplot(2, 5, 1)
plt.hist(df['duration'], bins=500, range=(15000, 800000)) # I use many bins for duration because the range of values is very large
    # I cut the range at 800,000 to make the graph more easily visible 
plt.xticks([]) # I get rid of the x and y labels to unclutter the plots
plt.yticks([])
plt.title('Duration')

# Plot 2: Danceability
plt.subplot(2, 5, 2)
plt.hist(df['danceability'], bins=100)
plt.xticks([])
plt.yticks([])
plt.title('Danceability')

# Plot 3: Energy
plt.subplot(2, 5, 3)
plt.hist(df['energy'], bins=100)  
plt.xticks([])
plt.yticks([])
plt.title('Energy')

# Plot 4: Loudness
plt.subplot(2, 5, 4)
plt.hist(df['loudness'], bins=100, range = (-40, 2))  
    # I cut the range at -40 and at 2 to make the graph more visible
plt.xticks([])
plt.yticks([])
plt.title('Loudness')

# Plot 5: Speechiness
plt.subplot(2, 5, 5)
plt.hist(df['speechiness'], bins=100) 
    # I was considering changing range to make the graph bigger, but I didn't want to exclude the many datapoints at ~.9  
plt.xticks([])
plt.yticks([])
plt.title('Speechiness')

# Plot 6: Acousticness
plt.subplot(2, 5, 6)
plt.hist(df['loudness'], bins=100) 
plt.xticks([])
plt.yticks([])
plt.title('Acousticness', fontsize='9')

# Plot 7: Insutrmentalness
plt.subplot(2, 5, 7)
plt.hist(df['instrumentalness'], bins=100, range=(-.1, 1))
plt.xticks([])
plt.yticks([])
plt.title('Instrumentalness', fontsize='9')

# Plot 8: Liveness
plt.subplot(2, 5, 8)
plt.hist(df['loudness'], bins=100)  
plt.xticks([])
plt.yticks([])
plt.title('Liveness',)

# Plot 9: Valence
plt.subplot(2, 5, 9)
plt.hist(df['valence'], bins=100)  
plt.xticks([])
plt.yticks([])
plt.title('Valence')

# Plot 10: Tempo
plt.subplot(2, 5, 10)
plt.hist(df['tempo'], bins=100)  
plt.xticks([])
plt.yticks([])
plt.title('Tempo')

plt.show()


# Question 2
  # Is there a relationship between song length and popularity of a song? If so, if the relationship positive or negative?

# For this question I find the correlation between song length and popularity and also plot a scatter plot of length vs pop.

dpCorr = df['duration'].corr(df['popularity']) # = -0.097

plt.scatter(df['duration'], df['popularity'])
plt.xlabel('Duration')
plt.ylabel('Popularity')
plt.title("Song Length vs Popularity")
plt.show()


# Question 3
  # Are explicitly rated songs more popular than songs that are not explicit? 

# Look at the popularity distribution to figure out what type of signficance test to use

plt.hist(df['popularity'], bins=100)
plt.show() # Since there are so many 0s, I decided to use a non-parametric test

# Mann-Whitney U test

notEx = df['popularity'].where(df['explicit']==False).dropna() # We filter for non-explicit popularity
Ex = df['popularity'].where(df['explicit']==True).dropna() # We filter for explicit popularity

u3, p3 = stats.mannwhitneyu(notEx, Ex) # Obtain test values
print("p3:", p3)
print("Not-explicit median:", notEx.median())
print("Explicit median:", Ex.median())

# I compared the medians to see which one was larger. Explicit songs have a median of 34, while non-explicit songs have a 
# median of 33. The Mann-Whitney U test resulted in a statistically significant p-value, so it is very likely that explicit
# songs are rated higher than non-explicit songs.


# Question 4
  # Are songs in major key more popular than songs in minor key? 

# Model the popularity distribution for major key and for minor key replicants
major = df['popularity'].where(df['mode']==1).dropna() 
    # Mode of 1 = major, mode of 0 = minor
minor = df['popularity'].where(df['mode']==0).dropna() 

# Plot 1: Major Key 
plt.subplot(2, 5, 1)
plt.hist(major, bins=100) 
plt.xticks([0, 50, 100])
plt.yticks([])
plt.title('Major-Key')

# Plot 2: Danceability
plt.subplot(2, 5, 2)
plt.hist(minor, bins=100)
plt.xticks([0, 50, 100])
plt.yticks([])
plt.title('Minor-Key')
plt.show()

# Now performing the signficance test, like question 3
u4, p4 = stats.mannwhitneyu(major, minor) # Obtain test values
print("p4:", p4)
print("Major median:", major.median())
print("Minor median:", minor.median())


# Question 5
  # Energy is believed to largely reflect the “loudness” of a song. Can you substantiate (or refute) that this is the case?
  
# Find the correlation between energy and loudness to see how well associated they are

elCorr = df['energy'].corr(df['loudness']) # = 0.7755

plt.scatter(df['energy'], df['loudness'])
plt.xlabel('Energy')
plt.ylabel('Loudness')
plt.title("Energy vs Loudness")
plt.show()


# Question 6
  # Which of the 10 song features in question 1 predicts popularity best? How good is this model?

# Create a dataframe of the 10 features and popularity
d11 = df.iloc[:, 0:2]
d11['danceability'] = df['danceability']
d11['energy'] = df['energy']
d11['loudness'] = df['loudness']
d11['speechiness'] = df['speechiness']
d11['acousticness'] = df['acousticness']
d11['instrumentalness'] = df['instrumentalness']
d11['liveness'] = df['liveness']
d11['valence'] = df['valence']
d11['tempo'] = df['tempo']

# Create a correlation matrix
d11corr = d11.corr()

# Create an R-squared matrix
d11r2 = d11corr.iloc[:, 0]
for i in range (len(d11r2)):
    d11r2[i] = round((d11r2[i]**2), 5)

# Create linear regression model
x = d11['instrumentalness'].values.reshape(-1, 1)
y = d11['popularity'].values.reshape(-1, 1)
lr6 = LinearRegression().fit(x, y)
print("R2:", lr6.score(x, y)) # The R2 of the model reflects the calculated R2 in the R2 matrix

# Create a scatter plot of the instrumentalness model
ypred = lr6.predict(x)
plt.scatter(x, y)
plt.plot(x, ypred, color='red')
plt.xlabel('Instrumentalness')
plt.ylabel('Popularity')
plt.title("Instrumentalness vs Popularity")
plt.show()


# Question 7.
  # Building a model that uses *all* of the song features in question 1, how well can you predict popularity? How much 
  # (if at all) is this model improved compared to the model in question 7). How do you account for this?

# Create a multiple regression model with all 10 factors as predictors
X = d11.iloc[:, 1:]
Y = d11.iloc[:, 0]
mr7 = LinearRegression().fit(X, Y)
print("R2:", mr7.score(X, Y)) #.096

# Run OLS to double check my results 
X2 = sm.add_constant(X)
model = sm.OLS(Y,X2).fit()
print(model.summary()) # R2 of .096


# Question 8
  # When considering the 10 song features above, how many meaningful principal components can you extract? What 
  # proportion of the variance do these principal components account for? Using the principal components, how many
  # clusters can you identify? 

# Look at the correlation matrix
d10 = d11.iloc[:, 1:]
corrMatrix = np.corrcoef(d10, rowvar=False)
plt.imshow(corrMatrix)
plt.xlabel('Feature')
plt.ylabel("Feature")
plt.colorbar()
plt.show()

# Standardize the data (10 characteristics)
zscoredData = stats.zscore(d10)

# Create PCA object, get eigenvalues and eigenvectors
pca = PCA().fit(zscoredData)
eigVals = pca.explained_variance_
varExplained = eigVals/sum(eigVals)*100
loadings = pca.components_

# Rotate the data
rotatedData = pca.fit_transform(zscoredData)

# Scree plot
nfeatures = d10.shape[1]
x8 = np.linspace(1, nfeatures, nfeatures)
plt.bar(x8, eigVals)
plt.plot([0, nfeatures], [1, 1], color = 'orange') # Orange line showing kaiser criterion
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.show()

# Kaiser criterion
kaiserThreshold = 1
nFactors = np.count_nonzero(eigVals > kaiserThreshold)
print(nFactors)

# Graphing variance explained
eigSum = np.cumsum(varExplained)
plt.plot(x8, eigSum)
plt.axvline(x = 3, color='red') # 3 PCs
plt.plot(3, 58, 'go') #Intersection point
plt.title('Variance Explained with n Principal Components')
plt.xlabel('n Principal Components')
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.ylabel('Total Variance Explained')
plt.show()

# Principal Component Interpretation
whichPrincipalComponent = 3 # Select and look at one factor at a time
plt.bar(x8, loadings[whichPrincipalComponent, :]*(-1))
plt.xlabel('Feature')
plt.ylabel('Loading')
plt.show()

# PC1 consists of heavy positive loadings of features 3 and 4, and a heavy negative loading of feature 6
    # Energy, loudness, acousticness
# PC2 consists of heavy positive loadings of features 2 and 9, and a heavy negative loading of feature 7
    # Danceability, valence, instrumentalness
# PC3 consists of heavy negative loadings of feature 5 and 8
    # Speechiness, liveness
# PC4 consists of a positive loading of feature 10 and a negative loading of feature 1
    # Duration, tempo


# Question 9
  # Can you predict whether a song is in major or minor key from valence? If so, how good is this prediction? If not, is 
  # there a better predictor?
  
# Major and minor key is a binary outcome, so I'm going to implement logistic regression

# Start with a simple EDA
y9 = df['mode'] # Outcome
x9 = df['valence'] # Predictor
x9 = np.array(x9).reshape(len(x9), 1) # Reshaped it
plt.scatter(x9, y9)
plt.xlabel('Valence')
plt.ylabel('Key Mode')
plt.title('Valence vs Key Mode')
plt.yticks([0, 1])
plt.show() # Looks like valence might not be a good predictor, every valence has both modes

# Run Logistic Regressions
logmodel = LogisticRegression().fit(x9, y9)

# Plot the model
x9a = np.linspace(0, 1, 38512)
y9a = x9a * logmodel.coef_ + logmodel.intercept_
sigmoid = expit(y9a)

plt.plot(x9a, sigmoid.ravel(), color='red', linewidth=3)
plt.scatter(x9, y9)
plt.axhline(y=.5, color='gray', linestyle='dotted')
plt.xlabel('Valence')
plt.ylabel('Key Mode')
plt.yticks([0, 1])
plt.show()

# Make predictions
valenceScore = .7
probMode = sigmoid[0, np.abs(x9a-valenceScore).argmin()].round(3)
print("Probability:", probMode)

# Repeat steps above while changing valence for other predictors, to see if there is a better predictor
y9b = df['mode'] # Outcome
x9b = df['acousticness'] # Predictor
x9b = np.array(x9b).reshape(len(x9b), 1) # Reshaped it

logmodel2 = LogisticRegression().fit(x9b, y9b)
x9c = np.linspace(0, 1, 38512)
y9c = x9c * logmodel2.coef_ + logmodel2.intercept_
sigmoid2 = expit(y9c)

plt.plot(x9c, sigmoid2.ravel(), color='red', linewidth=3)
plt.scatter(x9, y9)
plt.axhline(y=.5, color='gray', linestyle='dotted')
plt.xlabel('Acousticness')
plt.ylabel('Key Mode')
plt.yticks([0, 1])
plt.show()

probMode = sigmoid2[0, np.abs(x9c-valenceScore).argmin()].round(3)
print("Probability:", probMode)

# Predictors 
    # Bad: Popularity, duration, explicit, danceability, energy, key, loudness, speechiness, acousticness, instrumentalness
         # liveness, tempo, time_signature, track_genre
# Every predictor was bad, all of them suffer the same problems, however acousticness was the best predictor


# Question 10
  # Can you predict the genre, either from the 10 song features from question 1 directly or the principal components you 
  # extracted in question 8?

# Build the classification tree data from the meaningful principal components
x10 = np.column_stack((rotatedData[:, 0], rotatedData[:, 1], rotatedData[:, 2]))

# Convert genre to numeric labels
genres = df['track_genre'].unique()
genDict = dict(zip(genres, range(0, len(genres))))
y10 = df['track_genre']
y10 = y10.map(genDict, na_action='ignore') # All genres have been changed to numeric integers from 0 to 51

# Train Test Split 70/30 
xtrain, xtest, ytrain, ytest = train_test_split(x10, y10, test_size=0.3)

# Decison tree
clf = DecisionTreeClassifier(criterion='gini').fit(xtrain, ytrain)
y10p = clf.predict(xtest)

# Metrics
print(metrics.classification_report(ytest, y10p))

# Plot tree: too computationally intensive for my computer
#plt.figure(figsize=(15, 7.5))
#plot_tree(clf, filled=True, rounded=True, class_names=(genres.tolist()))
#plt.show() 

# Create a confusion matrix of every 
cm = metrics.multilabel_confusion_matrix(ytest, y10p)

# Plot the confusion matrix
f, axes = plt.subplots(11, 5, figsize=(12, 16))
axes=axes.ravel()
for i in range (52):
    display = metrics.ConfusionMatrixDisplay(cm[i], display_labels=[]).plot(ax=axes[i]) # Each confusion matrix
    display.ax_.set_title(genres[i]) # Title of matrix = genre
    if i<50:
        display.ax_.set_xlabel('') # Formatting
    if i%5!=0:
        display.ax_.set_ylabel('') # Formatting
    display.im_.colorbar.remove() 

# Formatting the other 3 bars
axes[52].set_axis_off()
axes[53].set_axis_off()
axes[54].set_axis_off()


# More formatting
plt.subplots_adjust(wspace=.1, hspace=.35)
f.colorbar(display.im_, ax=axes)
plt.show()

