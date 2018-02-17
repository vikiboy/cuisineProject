"""
Created on Thu Dec  7 21:59:58 2017

@author: Raphael
"""
import json
import numpy as np



data = json.load(open('../dataset/train.json'))

#Preprosessing: cusine norminalization
cuisine_cnt_vector      = {'greek': 0, 'southern_us': 0, 'indian': 0,
                           'italian': 0, 'mexican': 0, 'chinese': 0,
                           'thai': 0, 'french': 0,
                           }
cuisine_norm = cuisine_cnt_vector.copy()

for d in data:
    cuisine = d['cuisine']
    if(cuisine in cuisine_norm):
        cuisine_norm[cuisine] += 1

for c in cuisine_norm:
    cuisine_norm[c] /= len(data)


#Parsing
cookbook = dict()
id_cnt = 0

for d in range(len(data)):
    recipe = data[d]
    act_cuisine = recipe['cuisine']
    if(act_cuisine in cuisine_cnt_vector):
        for ing in recipe['ingredients']:
            if(ing not in cookbook):
                cookbook[ing] = {'id': id_cnt, 'cuisine_vec': cuisine_cnt_vector.copy()}
                id_cnt += 1

            cookbook[ing]['cuisine_vec'][act_cuisine] += 1


#Postprocessing

keylist = list(cookbook.keys())

for ing in cookbook:
    vec = cookbook[ing]['cuisine_vec']

    appearence_cross_cuisines = 0
    for val in vec:
        appearence_cross_cuisines += vec[val]

    for val in vec:
        vec[val] = (vec[val]/appearence_cross_cuisines)*cuisine_norm[val]

num_fav_ings = 100
cuisine_histo_list = []
for c in cookbook:
    cuisine_vec = cookbook[c]['cuisine_vec']
    cuisine_histo = []
    for ent in cuisine_vec:
        cuisine_histo.append(cuisine_vec[ent])
    cuisine_histo_list.append(cuisine_histo)

cuisine_histo_list = np.array(cuisine_histo_list)

fav_ing_dict = dict()
for x in range(8):
    fav_ings = []
    # print(list(cuisine_cnt_vector.keys())[x])
    t = cuisine_histo_list[:,x]

    indices = np.argsort(t)[-num_fav_ings:]
    for i in range(num_fav_ings):
        fav_ings.append(keylist[indices[i]])

    # print(fav_ings)
    fav_ing_dict[list(cuisine_cnt_vector.keys())[x]] = fav_ings
    print fav_ing_dict

    with open('fav_ing_dict_2.json', 'w') as fp:
        json.dump(fav_ing_dict, fp)
