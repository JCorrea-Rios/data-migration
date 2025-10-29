import numpy as np
import pandas as pd

# def char_bigrams(text):
#     ## Creates bigrams from strings
#     return [text[i:i+2] for i in range(len(text) - 1)]

def char_ngrams(text, n):
    ## Creates n-grams from strings
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def tokenize_4_TFIDF(company_name):
    ## Creates tokens appropriate for our TFIDF similarity function
    company_name = company_name.lower()
    tokens = char_ngrams(company_name, n=3)
    return tokens


########## Add a function for creating blocks based on Country

##### Update Similarity Matrix so it only compares names within the same block

###

def produceSimilarityMatrix(df, similarity_function_class, Name_Column="Name"):
    """
    Takes in a dataframe and a py_stringmatching similarity function class
    Compares all rows against each other
    Returns a similarity matrix
    """
    import numpy as np
    from py_stringmatching import similarity_measure as sm

    similarity_matrix = np.zeros((df.shape[0], df.shape[0]))

    df.reset_index(drop=True, inplace=True)

    index_range = df.index.tolist()
    # Compute similarity for each pair of rows


    # FOR OUR BAG BASED SIMILARITY FUNCTIONS
    if isinstance(similarity_function_class, sm.tfidf.TfIdf):
        for i in index_range:
            for j in index_range:
                similarity_matrix[i, j] = similarity_function_class.get_raw_score(tokenize_4_TFIDF(df[Name_Column][i]), tokenize_4_TFIDF(df[Name_Column][j]))
    
    # FOR OUR EDIT DISTANCE AND ALIGNMENT BASED FUNCTIONS
    else:
        for i in index_range:
            for j in index_range:
                similarity_matrix[i, j] = similarity_function_class.get_raw_score(df[Name_Column][i], df[Name_Column][j])

    np.fill_diagonal(similarity_matrix, -np.inf) ### fills the diagonals with neg infinity as we don't care about an item being similar to itself.

    return similarity_matrix









def addTopN_SimilaritiesToDf(df=None, similarity_matrix=None, N=None, similarity_name="Top_N_Similar"):
    import numpy as np
    # Get the top N similar entries for each row
    if N == None: 
        N = 2
    top_N_indices = np.argsort(similarity_matrix, axis=1)[:, -N:]
    #print(top_N_indices)

    # Translate indices back to original strings
    top_N_similar = []
    top_N_similar_scores = []
    for row_index, indices in enumerate(top_N_indices):
        top_similarities_for_row = [df.iloc[i, 0] for i in indices]
        top_similarities_scores_for_row = [similarity_matrix[row_index][i] for i in indices]
        top_N_similar.append(top_similarities_for_row)
        top_N_similar_scores.append(top_similarities_scores_for_row)

    # Add the new columns to the DataFrame
    df[similarity_name] = top_N_similar
    df[similarity_name + "_Scores"] = top_N_similar_scores

    return df