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

tasteLabels =   {"salt": "salty","sea":"salty","tomato":"umami",\
                "sweet": "sweet","sugar": "sweet",\
                "honey": "sweet","syrup": "sweet",\
                "chicken": "umami","mushroom":"umami",\
                "fish": "umami","chocolate":"sweet","cereal":"sweet",\
                "sour":"sour","pickle":"sour","lime":"sour",\
                "bitter":"bitter","coffee":"bitter","tea":"bitter",\
                "hot":"spicy","spicy":"spicy","chili":"spicy","pepper":"spicy"}

labels = {"salty":0,"sweet":1,"umami":2,"sour":3,"spicy":4,"bitter":5}

dirPath = "../dataset/"

datasetDir = "USDA/data/"

directory = dirPath + datasetDir

fileName = "ingredientslist.json"
filePath = dirPath + fileName

def USDAParser(directory,tasteLabels,labels):
    """
    Parser for USDA
    Generates an output list of id, foodname and tasteLabel
    """
    idx = 0

    foodInfo = []

    for filename in os.listdir(directory): # 1 food ingredient at a time
        if filename.endswith(".txt"):
            filePath = directory + filename
            idx = idx + 1
            with open(filePath, 'r') as f:
                USDAdict = json.load(f)

            name = USDAdict['report']['food']['name'].encode('ascii','ignore') # Storing the food name
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

    return foodInfo

def ingredientParser(filePath,tasteLabels,labels):
    """
    Parser for USDA
    Generates an output list of id, foodname and tasteLabel
    """
    idx = 0

    foodInfo = []

    with open(filePath,'r') as f:
        ingDict = json.load(f)

    itemCount = len(ingDict)

    for i in range(itemCount):
        word = ingDict[i]

        name = ingDict[i].encode('ascii','ignore')
        fullName = name.split(",")
        foodName = ""

        for length in range(len(fullName)):
            foodName += fullName[length] + " "

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

    return foodInfo

def buildSentences(foodInfo):
    """
    Generate sentences for Word2Vec model using foodInfo dict {idx,foodName,tasteLabel/foodLabel}
    returns :sentences - list for Word2Vec with {name.split()}
    returns :sentenceInfo - list with {name.split(),idx,foodLabel}
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

    return sentences,sentenceInfo

def trainWord2VecModel(sentences):
    """
    Use sentences list to train a word2vec model
    returns : model, words - list of vocabulary
    """
    size = 100 # dimension size
    window = 3 # maximum distance between a target word and words around the target word
    min_count = 5 # minimum count of words to consider when training the model
    workers = 8
    sg = 1

    model = Word2Vec(sentences, min_count=min_count, window = window, size = size, workers = workers, sg = sg)
    model.save('model.bin')
    words = list(model.wv.vocab) # Unique word vocabulary

    return model,words

def generateVectorsVocab(model,sentenceInfo,words):
    """
    returns: labelVec - {word,foodLabel,vector}
    returns: X - numpy array of vector
    """
    labelVec = []
    wordCount = len(words)

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

    return labelVec,X

def trainGMM(X):
    """
    returns : gmmPredLabels, gmmPredScores
    """
    dpgmm = mixture.BayesianGaussianMixture(n_components=6,covariance_type='full',max_iter=1000).fit(X)
    gmmPredLabels = dpgmm.predict(X)
    gmmPredScores = dpgmm.predict_proba(X)

    with open('gmmModel.p', 'wb') as fp:
        pickle.dump(dpgmm, fp)

    return gmmPredLabels,gmmPredScores

def clusterLabeling(gmmPredLabels,labelVec):
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

    with open('finalClusterLabel.p', 'wb') as fp:
        pickle.dump(finalClusterLabel, fp)

    return finalClusterLabel

def generateTasteInfo(foodInfo,labelVec,gmmPredScores,finalClusterLabel):
    """
    Get taste information for all ingredients
    returns - tasteInfo - {foodName,scores}
    """

    tasteInfo = []
    foodItemCount = len(foodInfo)
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

    return tasteInfo

def main():
    foodInfo1 = USDAParser(directory,tasteLabels,labels)
    foodInfo2 = ingredientParser(filePath,tasteLabels,labels)
    foodInfo = foodInfo1 + foodInfo2
    sentences,sentenceInfo = buildSentences(foodInfo)
    model,words = trainWord2VecModel(sentences)
    labelVec,X = generateVectorsVocab(model=model,sentenceInfo=sentenceInfo,words=words)
    gmmPredLabels,gmmPredScores = trainGMM(X)
    finalClusterLabel = clusterLabeling(gmmPredLabels,labelVec)
    tasteInfo = generateTasteInfo(foodInfo,labelVec,gmmPredScores,finalClusterLabel)
    print tasteInfo
    with open('tasteInfo.p', 'wb') as fp:
        pickle.dump(tasteInfo, fp)

if __name__ == "__main__":
    main()
