from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from collections import Counter
import numpy as np
import pickle
import os
import traceback

def main():
    #create database connection
    engine = create_engine('mssql://BRIANNA\\SQLEXPRESS/article_bias?trusted_connection=yes&driver=SQL+Server')
    
    #map the database structure onto an object oriented model for simpler processing
    Base = automap_base()
    Base.prepare(engine, reflect=True)
    
    #This simply creates easier ways to reference the newly created classes
    Article = Base.classes.article
    Ngram = Base.classes.ngram
    Article_Ngram = Base.classes.article_ngram
    
    #using this object relational model, create a session with the database
    session = Session(engine)
    
    allArticles = session.query(Article).all()

    #Save the total number of articles to use in the IDF calculations
    articleTotal = len(allArticles)
    
    for i, art in enumerate(allArticles):
        print("Processing article #",i+1)
        #Counter produces the array of ngrams along with their frequencies
        ngrams = Counter(art.processed_text.split(" "))
        print("Number of n-grams to process:",len(ngrams))
        #Save the frequency of the most common ngram to use in the augmented TF calculations
        mostCommon = ngrams.most_common(1)[0][1]
        a = 0.5
        
        for n in ngrams:
            #Query the database to see if the ngram already exists in the ngram table
            #one_or_none ensures that if there are no results, it will return as None
            #then calculate the augmented term frequency for this article/ngram combination
            sqlN = session.query(Ngram).filter_by(ngram=n).one_or_none()
            tf = a + a*ngrams[n]/mostCommon
            
            if sqlN is None:
                #if the ngram doesn't exist in the table, then we create a new record
                #IDF initializes as the log10 of the articleTotal since the ngram has
                #only appeared in 1 article so far
                newNGram = Ngram(ngram=n, article_count=1, inv_doc_freq=np.log10(articleTotal))
                session.add(newNGram)
                session.commit()
                
                #after saving the record to the database, we query for it to grab its ID
                #then we can create the article_ngram record
                addedNGram = session.query(Ngram).filter_by(ngram=n).one_or_none()
                artNGram = Article_Ngram(article_id=art.article_id, ngram_id=addedNGram.ngram_id, term_freq=tf)
                session.add(artNGram)
            else:
                #if the ngram is already in the ngram table, then we update the article_count
                #and then recalculate the IDF with the new count
                sqlN.article_count = sqlN.article_count + 1
                sqlN.inv_doc_freq = np.log10(articleTotal/sqlN.article_count)
                
                #then we also create a new article_ngram record in the same way as above
                artNGram = Article_Ngram(article_id=art.article_id, ngram_id=sqlN.ngram_id, term_freq=tf)
                session.add(artNGram)
    
    #if there are any leftover records in the session that weren't committed when the ngrams
    #were committed, this will insert them all
    session.commit()
        
    #now that all the ngrams have their final IDF scores, we are ready to calculate the TF-IDF scores
    #this query grabs all the article_ngram records and their associated ngram record
    artNGrams = session.query(Ngram, Article_Ngram).join(Article_Ngram, Ngram.ngram_id == Article_Ngram.ngram_id).all()
    
    #here we simply iterate over all the article_ngram records and calculate its TF-IDF
    #the query returns result objects which are tuples of the article_ngram and ngram records
    #which is why the referencing here is unintuitive
    for ang in artNGrams:
        ang[1].tf_idf = ang[1].term_freq * ang[0].inv_doc_freq
    
    #all of the changes above are saved by the session so they can be committed all at once
    session.commit()
    
    # query all the ngram ids to be used in creating the term-document matrix
    # the id will correspond to the column or dimension
    ngram_ids = []
    for val in session.query(Ngram.ngram_id).filter(Ngram.article_count>16).distinct():
         ngram_ids.append(val[0])
    
    # make sure the ids are in order for consistency between articles     
    ngram_ids.sort()
    
    #Query all the articles
    allArticles = session.query(Article).all()
    
    #Prepare X and Y arrays
    X = []
    Y = []
    
    for art in allArticles:
        #converts the source name into either 0 or 1. 
        #The choice of HuffPo as 1 here was completely arbitrary
        Y_class = int(art.source_name == 'huffpo')
        
        #Query all the article_ngram records associated with this article
        artNGrams = session.query(Article_Ngram) \
                    .filter_by(article_id=art.article_id) \
                    .join(Ngram, Ngram.ngram_id == Article_Ngram.ngram_id) \
                    .filter(Ngram.article_count>16).all()
        
        #initialize a zero vector with ngram_ids dimensions
        X_vector = [0] * len(ngram_ids)
        
        #use the ngram_id to update the corresponding index in X_vector with its TF-IDF score
        for ang in artNGrams:
            idx = ngram_ids.index(ang.ngram_id)
            X_vector[idx] = ang.tf_idf
        
        #update the appropriate arrays    
        Y.append(Y_class)
        X.append(X_vector)
    
    # convert to numpy arrays so they can be used as inputs for the models
    X = np.array(X)
    Y = np.array(Y)
    
    save_path = r'C:\Users\bdardin\Documents\Political Bias Project'
    
    # pickle or store the data for easy retrieval    
    with open(os.path.join(save_path,"x.txt"), "wb") as fp:
        pickle.dump(X, fp)
        
    with open(os.path.join(save_path,"y.txt"), "wb") as fp:
        pickle.dump(Y, fp)

if __name__ == "__main__":
    # wrap the program in a try/except block in case there are errors
    try:
        main()
    except Exception as e:
        print(e)
        print(traceback.format_exc())