import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch & format data
data = fetch_movielens(min_rating=4.0)

#print training and testing data

print(repr(data['train']))
print(repr(data['test']))

#create model
model = LightFM(loss='warp')

#train model
model.fit(data['train'], epochs=30, num_threads=2)


def sample_recommendation(model, data, user_ids):

    #number of users and movies in training data
    n_users, n_items = data['train'].shape

    #generate recommendations for each user input
    for user_id in user_ids:

        #movies already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #movies model predicts to like
        scores = model.predict(user_id, np.arange(n_items))
        
        #rank best to worst
        top_items = data['item_labels'][np.argsort(-scores)]

        #print results
        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)
            
sample_recommendation(model, data, [4, 26, 69])