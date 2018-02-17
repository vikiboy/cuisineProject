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

filePath = "recipe.json"

trainedNames = ['mexican','chinese','american','greek','indian','italian','thai','french']

def generateAllIngredients(filePath):

    """
    Given recipe list generate the ingredient list
    Input : Prior Cuisine List, Prior Recipe list
    Output : Ingredient List
    """


    cuisineFile = open(filePath)
    cuisineStr = cuisineFile.read()

    cuisineDataJSON = json.loads(cuisineStr)

    inputLoadCuisineDict = {}
    ingredientList = []

    for cuisine in trainedNames:
        currentCuisine = cuisineDataJSON[cuisine]
        currentCuisineIngList = []
        for recipe in currentCuisine:
            currentIngredientList = currentCuisine[recipe]
            for i in range(len(currentIngredientList)):
                currentCuisineIngList.append(currentIngredientList[i].encode('ascii','ignore'))

        inputLoadCuisineDict[cuisine] = currentCuisineIngList
        ingredientList+=currentCuisineIngList


    with open('fav_ing_dict_3.json', 'w') as fp:
        json.dump(inputLoadCuisineDict, fp)

    with open('ingredientList.json','w') as fpp:
        json.dump(ingredientList,fpp)

def main():
    generateAllIngredients(filePath=filePath)

if __name__ == '__main__':
    main()
