import json
import os
from pprint import pprint
import pandas as pd
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
from sklearn.preprocessing import normalize
import cPickle as pickle


dirPath = "../dataset/"

datasetDir = "USDA/data/"

directory = dirPath + datasetDir


tasteLabels =   {"salt": "salty","sea":"salty","tomato":"umami",\
                "sweet": "sweet","sugar": "sweet",\
                "honey": "sweet","syrup": "sweet",\
                "chicken": "umami","mushroom":"umami",\
                "fish": "umami","chocolate":"sweet","cereal":"sweet",\
                "sour":"sour","pickle":"sour","lime":"sour",\
                "bitter":"bitter","coffee":"bitter","tea":"bitter",\
                "hot":"spicy","spicy":"spicy","chili":"spicy","pepper":"spicy"}

# tasteLabels = {"sour":"sour","sweet":"sweet","salt":"salty","fish":"umami","coffee":"bitter","tea":"bitter","hot":"spicy","spicy":"spicy","lime":"sour","lemon":"sour","pickle":"sour"}

labels = {"salty":0,"sweet":1,"umami":2,"sour":3,"spicy":4,"bitter":5}

idx = 0

foodInfo = []

for filename in os.listdir(directory): # 1 food ingredient at a time
    if filename.endswith(".txt"):
        filePath = directory + filename
        idx = idx + 1
        with open(filePath, 'r') as f:
            USDAdict = json.load(f)

        name = USDAdict['report']['food']['name'] # Storing the food name
        # foodGroup = USDAdict['report']['food']['fg'] # Storing the food group
        fullName = name.split(",")
        foodName = ""

        for length in range(len(fullName)):
            foodName += fullName[length] + " "

        # foodName = fname.title() + " " + lname

        nutrientInfo = {} # Storing the nutrient information

        """
        Assigning taste labels to food name
        """
        foodLabel = []
        count = 0
        for item in tasteLabels:
            if item in foodName.lower():
                count +=1
                foodLabel.append(tasteLabels[item])

        if foodLabel != []:
            currentLabel = []
            for i in range(len(foodLabel)):
                currentLabel.append(labels[foodLabel[i]])
            d = {'idx':idx,'foodName':foodName,'foodLabel':currentLabel}
        else:
            d = {'idx':idx,'foodName':foodName,'foodLabel':[]}

        foodInfo.append(d.copy())

"""
Word2Vec model
"""

foodItemCount = len(foodInfo)
print "foodItemCount",foodItemCount
sentences = []
sentenceInfo = []

# Generating the sentences vector for Word2Vec input
for i in range(foodItemCount):
    currentIngredient = foodInfo[i]
    foodName = currentIngredient['foodName'].encode("ascii").split()
    info = {"name":foodName,"idx":currentIngredient['idx'],"label":currentIngredient['foodLabel']}
    sentenceInfo.append(info)
    sentences.append(foodName)

size = 100 # dimension size
window = 3 # maximum distance between a target word and words around the target word
min_count = 5 # minimum count of words to consider when training the model
workers = 8
sg = 1

model = Word2Vec(sentences, min_count=min_count, window = window, size = size, workers = workers, sg = sg)
model.save('model.bin')

words = list(model.wv.vocab) # Unique word vocabulary
wordCount = len(words)

labelVec = []

"""
Extracting the 100-dimensional vector for each word in the vocabulary
"""
for i in range(wordCount):
    currentWord = words[i]

    """
    Extracting the label of the word
    """
    for j in range(len(sentenceInfo)):
        currentSentence = sentenceInfo[j]
        if currentWord in currentSentence['name']:
            currentLabel = currentSentence['label']
            break

    wordnlabel = {'word':currentWord,'label':currentLabel,'vector':model.wv[currentWord]}

    labelVec.append(wordnlabel)

"""
Creating the numpy array
"""

inVec = []

for i in range(len(labelVec)):
    inVec.append(labelVec[i]['vector'])

X = np.array(inVec)

"""
Gaussian Mixture Model
"""

dpgmm = mixture.BayesianGaussianMixture(n_components=6,covariance_type='full',max_iter=1000).fit(X)
gmmPredLabels = dpgmm.predict(X)
gmmPredScores = dpgmm.predict_proba(X)
uniqueLabels =np.unique(gmmPredLabels)

"""
Post-hoc assigning labels to clusters
"""

trueClusterLabel = np.zeros([6,6]) # row - gmmPredLabels , col - tasteLabels
finalClusterLabel = []


for i in range(gmmPredLabels.shape[0]):
    tasteLabel = labelVec[i]['label']
    if tasteLabel != []:
        predLabel = gmmPredLabels[i]
        for j in range(len(tasteLabel)):
            currentTasteLabel = tasteLabel[j]
            trueClusterLabel[predLabel][currentTasteLabel] = trueClusterLabel[predLabel][currentTasteLabel] + 1


for i in range(6):
    score = trueClusterLabel[i]
    norm = np.array([float(i)/sum(score) for i in score])
    finalClusterLabel.append(norm)


"""
Get taste information for all ingredients
"""

tasteInfo = []

for i in range(foodItemCount):

    food = foodInfo[i]['foodName'].encode("ascii")
    foodName = food.split()
    foodWordCount = len(foodName)
    scores = np.zeros(6)
    for word in range(foodWordCount):
        currentWord = foodName[word]
        for j in range(len(labelVec)):
            if currentWord in labelVec[j]['word']:
                for k in range(6):
                    scores += gmmPredScores[j][k]*finalClusterLabel[k]

    scores = scores/foodWordCount

    taste = {'foodName':food,'scores':scores}
    tasteInfo.append(taste)

print tasteInfo

with open('tasteInfo.p', 'wb') as fp:
    pickle.dump(tasteInfo, fp)
# print tasteInfo
