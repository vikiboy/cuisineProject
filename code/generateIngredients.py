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
from getTasteInfo import loadWord2VecModel,loadClusters,getVector,getTasteInfoCuisine,\
                            getTasteInfoIngredient,loadCuisines,loadCuisine,getIngredientsForRecipe

def getIngredientPoolTasteInfo(ingredientPool,model,gmmModel,finalClusterLabel):
    """
    Generate the taste info vector for the ingredient pool
    """
    ingTasteInfo = getTasteInfoCuisine(ingredientPool,model,gmmModel,finalClusterLabel)
    return ingTasteInfo

def getMatchScore(ingTasteInfo,cuisineTasteInfo):
    """
    Get similarity score for taste between a set of ingredients from the target cuisine and the prior cuisine
    """
    return np.dot(ingTasteInfo,cuisineTasteInfo)

def addIngredientToPool(cuisine,ingredientPool):
    """
    Add the next ingredient from the target cuisine to the set of ingredients
    """
    idx = np.random.randint(len(cuisine)-1)
    ingredientPool.append(cuisine[idx])
    cuisine.remove(cuisine[idx])
    return ingredientPool,cuisine

def removeIngredientFromPool(ingredientPool):
    """
    Remove the last ingredient from the ingeedient pool
    """
    ingredientPool = ingredientPool[:-1]
    return ingredientPool

def generateIngredientSet(targetCuisine,priorCuisine,model,gmmModel,finalClusterLabel):
    """
    Generate the final set of ingredients from the target cuisine
    """
    ingredientPool = []

    priorTasteInfo = getTasteInfoCuisine(priorCuisine,model,gmmModel,finalClusterLabel)

    ingCount = 1
    prevScore = 0.0
    iterCount = 0

    updatedTargetCuisine = targetCuisine

    while (len(ingredientPool)<=10) and (len(updatedTargetCuisine)-1>0):
        iterCount +=1

        # Add ingredient to the pool from the target cuisine
        ingredientPool,updatedTargetCuisine = addIngredientToPool(updatedTargetCuisine,ingredientPool)

        # Get the taste score of the ingredient pool
        ingTasteInfo = getIngredientPoolTasteInfo(ingredientPool,model,gmmModel,finalClusterLabel)
        # Find the match score
        currentScore = getMatchScore(ingTasteInfo,priorTasteInfo)

        if currentScore>=prevScore:
            ingCount+=1
            prevScore = currentScore
        else:
            ingCount-=1
            ingredientPool=removeIngredientFromPool(ingredientPool)

    return ingredientPool

def saveOutputIngredients(ingredientPool,outputFilePath):
    """
    Save to a txt file
    """
    with open(outputFilePath,"wb") as output:
        for i in range(len(ingredientPool)):
            output.write(ingredientPool[i]+"\n")

    print "Written set of ingredients to ",outputFilePath

def main(targetCuisine,priorCuisine,recipeList,recipeFlag=False):
    """
    Generate Ingredients given prior cuisine, target cuisine
    """
    filePath = "./model.bin"
    gmmfilepath  = "./gmmModel.p"
    finalClusterLabelfilepath = "./finalClusterLabel.p"
    cuisinePath = "./fav_ing_dict_3.json"

    # Loading the word2vec model, gmm model, and the final cluster labels

    model = loadWord2VecModel(filePath)
    gmmModel,finalClusterLabel=loadClusters(gmmfilepath,finalClusterLabelfilepath)

    # Loading the dicts for prior and target cuisines

    targetCuisine = loadCuisine(targetCuisine,cuisinePath)
    if recipeFlag==False:
        priorCuisine = loadCuisines(priorCuisine,cuisinePath)
    else:
        priorCuisine = getIngredientsForRecipe(priorCuisine,recipeList,recipeFlag)

    # Generate ingredient pool
    # print priorCuisine

    ingredientPool = generateIngredientSet(targetCuisine,priorCuisine,model,gmmModel,finalClusterLabel)

    outputFilePath = "TEST.txt"
    saveOutputIngredients(ingredientPool,outputFilePath)
    print ingredientPool

if __name__ == "__main__":
    main('greek',['mexican','chinese'],["Taco Lasagna","Beef Fajitas","Shrimp Wontons"],True)
