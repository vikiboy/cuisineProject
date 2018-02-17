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

def returnRandomRecipes(priorCuisineList):
    """
    Takes in cuisine list, and returns 9 random recipes
    """


    cuisineFile = open(filePath)
    cuisineStr = cuisineFile.read()
    cuisineDataJSON = json.loads(cuisineStr)

    recipeList = []
    for i in range(len(priorCuisineList)):
        for recipe in cuisineDataJSON[priorCuisineList[i]]:
            recipeList.append(recipe.split(" ", 1)[0].encode('ascii','ignore'))


    randomList = np.random.choice(recipeList,9)

    return randomList

def main():
    returnRandomRecipes(['mexican','chinese'])

if __name__ == '__main__':
    main()
