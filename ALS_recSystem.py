import pandas as pd
import scipy.sparse as sparse
import numpy as np
import random
import implicit
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

def make_train(ratings, pct_test = 0.2):
    test_set = ratings.copy()
    test_set[test_set != 0] = 1
    training_set = ratings.copy()
    
    nonzero_inds = training_set.nonzero()
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))
    
    random.seed(0)
    
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs)))
    
    samples = random.sample(nonzero_pairs, num_samples)
    content_inds = [index[0] for index in samples] # Get the item row indices
    person_inds = [index[1] for index in samples] # Get the user column indices

    
    training_set[content_inds, person_inds] = 0
    training_set.eliminate_zeros()
    
    return training_set, test_set, list(set(person_inds))

def recommend(grouped_df,person_id, sparse_person_content, person_vecs, content_vecs, num_contents = 10):
    # Get the interactions scores from the sparse person content matrix
    person_interactions = sparse_person_content[person_id,:].toarray()
    # Add 1 to everything, so that articles with no interaction yet become equal to 1
    person_interactions = person_interactions.reshape(-1) + 1
    # Make articles already interacted zero
    person_interactions[person_interactions > 1] = 0
    # Get dot product of person vector and all content vectors
    rec_vector = person_vecs[person_id,:].dot(content_vecs.T).toarray()
    
    # Scale this recommendation vector between 0 and 1
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
    # Content already interacted have their recommendation multiplied by zero
    recommend_vector = person_interactions * rec_vector_scaled
    # Sort the indices of the content into order of best recommendations
    content_idx = np.argsort(recommend_vector)[::-1][:num_contents]
    
    # Start empty list to store titles and scores
    titles = []
    scores = []

    for idx in content_idx:
        # Append titles and scores to the list
        titles.append(grouped_df.title.loc[grouped_df.content_id == idx].iloc[0])
        scores.append(recommend_vector[idx])

    recommendations = pd.DataFrame({'title': titles, 'score': scores})

    return recommendations

def auc_score(predictions, test):
    # fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)


def calc_mean_auc(training_set, altered_persons, predictions, test_set):
    store_auc = [] # An empty list to store the AUC for each user that had an item removed from the training set

    content_vecs = predictions[1]
    for person in altered_persons: # Iterate through each user that had an item altered
        training_column = training_set[:,person].toarray().reshape(-1) # Get the training set column
        zero_inds = np.where(training_column == 0) # Find where the interaction had not yet occurred
        
        # Get the predicted values based on our user/item vectors
        person_vec = predictions[0][person,:]
        pred = person_vec.dot(content_vecs).toarray()[0,zero_inds].reshape(-1)
        
        # Get only the items that were originally zero
        # Select all ratings from the MF prediction for this user that originally had no iteraction
        actual = test_set[:,person].toarray()[zero_inds,0].reshape(-1)
                
        store_auc.append(auc_score(pred, actual)) # Calculate AUC for the given user and store
        
    # End users iteration
    
    return float('%.3f'%np.mean(store_auc))


def als(sparse_content_person, sparse_person_content):
    model_2 = implicit.als.AlternatingLeastSquares(factors = 20, regularization = 0.10, iterations = 100)
    
    alpha = 15
    data = (sparse_content_person * alpha).astype('double')
    
    model_2.fit(data)
    
    person_vecs = sparse.csr_matrix(model_2.user_factors)
    content_vecs = sparse.csr_matrix(model_2.item_factors)
   
    content_train, content_test, content_persons_altered = make_train(sparse_content_person, pct_test = 0.2)
    
    return(calc_mean_auc(content_train, content_persons_altered, [person_vecs, content_vecs.T], content_test))
    
 
def main():
    articles_df = pd.read_csv("D://Proj_RecommendationSystem//Article Recommendation//shared_articles.csv")
    interactions_df = pd.read_csv("D://Proj_RecommendationSystem//Article Recommendation//users_interactions.csv")
    
    articles_df.drop(['authorUserAgent', 'authorRegion', 'authorCountry'], axis = 1, inplace = True)
    interactions_df.drop(['userAgent', 'userRegion', 'userCountry'], axis = 1, inplace = True)
    
    print(articles_df.head())
    print(articles_df['eventType'].value_counts())
    
    articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
    articles_df.drop('eventType', axis = 1, inplace = True)
    
    print(articles_df.info())
    print(interactions_df.info())
    
    df = pd.merge(interactions_df[['contentId', 'personId', 'eventType']], articles_df[['contentId', 'title']], how = 'inner', on ='contentId')
    print(df.head())
    print(df.info())
    print(df['eventType'].value_counts())
    
    event_type_strength = {'VIEW' : 1.0, 'LIKE' : 2.0, 'BOOKMARK' : 3.0, 'FOLLOW' : 4.0, 'COMMENT CREATED' : 5.0}
    
    df['eventStrength'] = df['eventType'].apply(lambda x: event_type_strength[x])
    print(df.head())
    
    df.drop_duplicates()
    grouped_df = df.groupby(['personId', 'contentId', 'title']).sum().reset_index()
    
    print(grouped_df.sample())
    print(grouped_df.dtypes)
    
    grouped_df['title'] = grouped_df['title'].astype("category")
    grouped_df['personId'] = grouped_df['personId'].astype("category")
    grouped_df['contentId'] = grouped_df['contentId'].astype("category")
    grouped_df['person_id'] = grouped_df['personId'].cat.codes
    grouped_df['content_id'] = grouped_df['contentId'].cat.codes
    
    print(grouped_df.head())
    sparse_content_person = sparse.csr_matrix((grouped_df['eventStrength'].astype(float), (grouped_df['content_id'], grouped_df['person_id'])))
    sparse_person_content = sparse.csr_matrix((grouped_df['eventStrength'].astype(float), (grouped_df['person_id'], grouped_df['content_id'])))
    
    scores = []
    # scores.append(bpr(sparse_content_person,sparse_person_content))
    scores.append(als(sparse_content_person, sparse_person_content))
    # scores.append(lmf(sparse_content_person,sparse_person_content))
     
    print(scores)
    
main()
