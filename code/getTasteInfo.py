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
import re
import random

def loadWord2VecModel(filePath):
    """
    Load the trained Word2Vec model
    """
    model = Word2Vec.load(filePath)
    return model

def getVector(model,ingredientName):
    """
    Generate the Word2Vec vector output from the model
    """

    vocabWords = list(model.wv.vocab)
    wordCount = len(vocabWords)

    vector = []
    for i in range(wordCount):
        currentWord = vocabWords[i]
        if currentWord in ingredientName:
            v= model[currentWord]
            vector.append(np.array(v))

    return vector

def loadClusters(gmmfilepath,finalClusterLabelfilepath):
    """
    Load the GMM Clusters
    """
    dpgmm = pickle.load(open(gmmfilepath,"rb"))
    finalClusterLabel = pickle.load(open(finalClusterLabelfilepath,"rb"))
    return dpgmm,finalClusterLabel

def getTasteInfoIngredient(gmmmodel,vector,finalClusterLabel):

    """
    Generate the taste info vector for an ingredient
    """

    tasteInfoScores = np.zeros(6)

    for i in range(len(vector)):
        x = vector[i].reshape(1,100)
        scores = gmmmodel.predict_proba(x)
        for k in range(6):
            tasteInfoScores+=scores[0][k]*finalClusterLabel[k]

    return tasteInfoScores

def loadCuisine(cuisineName,cuisinePath):
    """
    Load ingredients from the given cuisine
    """
    with open(cuisinePath,"r") as f:
        cuisineDict = json.load(f)

    return cuisineDict[cuisineName]

def loadCuisines(cuisineNames,cuisinePath):
    """
    Load ingredients from the given cuisine
    """
    with open(cuisinePath,"r") as f:
        cuisineDict = json.load(f)

    noCuisines = len(cuisineNames)

    loadedCuisine = []

    for i in range(noCuisines):
        loadedCuisine+=cuisineDict[cuisineNames[i]]

    return loadedCuisine

def getTasteInfoCuisine(cuisine,model,gmmModel,finalClusterLabel):
    """
    Generate the taste info vector for a cuisine
    """
    ingCount = len(cuisine)

    tasteInfo = np.zeros(6)

    for i in range(ingCount):
        currentIng = cuisine[i].encode('ascii','ignore')
        v = getVector(model,currentIng)
        currentIngScore = getTasteInfoIngredient(gmmModel,v,finalClusterLabel)
        tasteInfo+=currentIngScore


    cuisineInfo = tasteInfo/float(ingCount)
    return cuisineInfo

def getIngredientsForRecipe(cuisineList,recipeList,recipeFlag=False):
    """
    Given recipe list generate the ingredient list
    Input : Prior Cuisine List, Prior Recipe list
    Output : Ingredient List
    """

    if recipeFlag==True:
        recipeFile = open("recipe.json")
        recipeStr = recipeFile.read()

        recipeDataJSON = json.loads(recipeStr) # recipes with ingredients

        cuisineCount = len(cuisineList)
        recipeCount = len(recipeList)

        ingredientList = []
        finalIngList = []
        # search for all cuisines
        for i in range(cuisineCount):
        # search for all recipes provided
            for j in range(recipeCount):
            # search for all recipes in current cuisine
                for recipe in recipeDataJSON[cuisineList[i]]:
                # search for current user-provided recipe in all recipes for current cuisines
                    if re.search(recipeList[j],recipe,re.IGNORECASE):
                    # get ingredient list for found recipe
                        currentIngList = recipeDataJSON[cuisineList[i]][recipe]
                        ingredientList += currentIngList

        for i in range(len(ingredientList)):
            finalIngList.append(ingredientList[i].encode('ascii','ignore'))

        return finalIngList

def main():
    """
    Generate Ingredients given prior cuisine, target cuisine
    """
    filePath = "./model.bin"
    gmmfilepath  = "./gmmModel.p"
    finalClusterLabelfilepath = "./finalClusterLabel.p"
    cuisinePath = "./fav_ing_dict.json"
    model = loadWord2VecModel(filePath)
    ingredient = "Cucumber"
    v = getVector(model,ingredient)
    dpgmm,finalClusterLabel=loadClusters(gmmfilepath,finalClusterLabelfilepath)
    tasteInfo = getTasteInfoIngredient(dpgmm,v,finalClusterLabel)
    cuisine = loadCuisine('greek',cuisinePath=cuisinePath)
    cuisineInfo = getTasteInfoCuisine(cuisine,model,dpgmm,finalClusterLabel)

if __name__ == "__main__":
    main()
