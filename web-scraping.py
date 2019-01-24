from selenium import webdriver
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
import random
import numpy as np
import re
import traceback

def main():
    #create database connection
    engine = create_engine('mssql://BRIANNA\\SQLEXPRESS/article_bias?trusted_connection=yes&driver=SQL+Server')
    
    #map the database structure onto an object oriented model for simpler processing
    Base = automap_base()
    Base.prepare(engine, reflect=True)
    
    #This simply creates easier ways to reference the newly created classes
    #Here we only need the Article class
    Article = Base.classes.article
    
    #using this object relational model, create a session with the database
    session = Session(engine)
    
    #query for existing articles in the database and save their URLs in a list
    #this ensures that duplicate articles aren't added to the corpus
    allArticles = session.query(Article).all()
    old_urls = []
    
    if len(allArticles) > 0:
        for art in allArticles:
            old_urls.append(art.article_url)
    
    # create the window-less Chrome browser so we can click "load more" buttons
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--window-size=1920,1080')
    driver = webdriver.Chrome(options=options)
    
    # create a dictionary to store the URLs and later the article objects for each site
    site_dict = {"fox": {"urls": [],
                        "articles": [] },
                 "huffpo": {"urls": [],
                            "articles": [] }
                }
    
    # since HuffPo's politics section is paginated, do a while loop and
    # increment the page number at the end of each loop
    huffpo_base = 'https://www.huffingtonpost.com'
    huffpo_page = 1
    
    # 23 was set as the max page as it was the furthest back available
    while huffpo_page < 24:
        # request the URL and retrieve the HTML
        huffpo_url = huffpo_base + '/section/politics?page=' + str(huffpo_page)
        driver.get(huffpo_url)
        html = driver.page_source
        soup = BeautifulSoup(html, "lxml")
        
        # each article on the page is displayed in its own "card"
        page_articles = soup.find_all("div", {"class": "card__details"})
        print("# of articles on page",huffpo_page,":",len(page_articles))
              
        if len(page_articles) > 0:
            for p in page_articles:
                # this makes sure advertisements aren't saved as articles
                byline = p.find("div", {"class": "card__byline"})
                if byline == None or "Advertisement" in byline.get_text():
                    continue
                else:
                    # save the article URL to the list if it wasn't already added
                    url = huffpo_base + p.find("a").get("href")
                    if url not in old_urls and url not in site_dict["huffpo"]["urls"]:
                        site_dict["huffpo"]["urls"].append(url)
                    else:
                        continue
        
        huffpo_page += 1
    
    # fox has multiple political sections and load more buttons instead of pagination
    # so instead the sections are looped over
    fox_base = 'https://www.foxnews.com'    
    fox_categories = ['/politics',
                      '/category/politics/executive',
                      '/category/politics/senate',
                      '/category/politics/judiciary',
                      '/category/politics/foreign-policy',
                      '/category/politics/elections']

    for c in fox_categories:
        print(c)
        driver.get(fox_base+c)
        
        # keep track of how many articles are being displayed on the page
        previous_length = 0
        while True:
            # get the HTML
            html = driver.page_source
            soup = BeautifulSoup(html, "lxml")
            
            # find the list of articles
            main_list = soup.find("section", {"class": "has-load-more"})
            main_articles = main_list.find_all("article")
            
            # iterate only through the articles that were newly added
            # instead of processing previously seen articles
            for m in main_articles[previous_length:]:
                links = m.find_all("a")
                for l in links:
                    # save the article URL to the list if it wasn't already added
                    url = fox_base + l.get("href")
                    if url != None and url not in old_urls and url not in site_dict["fox"]["urls"]:
                        site_dict["fox"]["urls"].append(url)
            
            # update the previous length so the next loop knows where to start
            previous_length = len(main_articles)
            print("new previous length",previous_length)
            
            # find the load more button
            button = driver.find_element_by_css_selector("div.load-more")
            
            # if it is possible to click on the button, keep the loop going
            # and add as many URLs as possible
            # if clicking the button fails, break and move to the next section
            try:
                button.click()
                print("click successful")
            except:
                print("click failed")
                break
    
    # now that we have the article URLs, we can process them and add to the database
    random.seed(1)
    for site in site_dict.keys():
        for i, u in enumerate(site_dict[site]["urls"]):
            # get the HTHML for the article
            driver.get(u)
            art_html = driver.page_source
            soup = BeautifulSoup(art_html, "lxml")
            
            # first get the raw text of the article then process the text
            # these were placed in separate functions mainly to improve readability
            raw_text = extract_text(site, soup)
            processed = process_text(raw_text)
            
            # create an Article object that will commit the data to the database
            new_article = Article(source_name=site,article_url=u,raw_text=raw_text,processed_text=processed)
            site_dict[site]["articles"].append(new_article)
            print("Processed",site,"article #",i+1)
        
        # initially the idea was to use a train/test split so each article was randomly assigned 
        # to be either training or test by creating a list of indices corresponding to the 
        # number of articles and then randomly shuffling them. 
        # ultimately chose to use cross-validation instead but this split is still saved in the DB
        randomList = np.arange(0,len(site_dict[site]["articles"]))
        random.shuffle(randomList)
        
        # the article's index in the original list is used to grab the number randomly assigned 
        # to that index, if that number was greater than 80% of the list's size 
        # it is a test article, otherwise a training one
        for i, a in enumerate(site_dict[site]["articles"]):
            train = 1
            if(randomList[i] >= round(len(site_dict[site]["articles"])*.8)):
                train = 0
                
            a.is_training = train
            session.add(a)
    
    # once all articles are added to the session, commit the changes to the database    
    session.commit()

def extract_text(site, soup):
    # each site has a different way of representing the text of the article
    # so this ensures all the HTML tags holding the text are found
    if "fox" in site:
        article_div = soup.find("div", {"class": "article-body"})
        text_divs = article_div.find_all("p")
    else:
        article_div = soup.find("div", {"class": "entry__text"})
        text_divs = article_div.find_all("div", {"class": "text"})
    
    # once extracted, extracting the text from the HTML works the same for both
    article_text = ''
    for t in text_divs:
        article_text += t.get_text() + " "
        
    return article_text    
    
def process_text(proc):
    #in case the text has this character, make sure it's replaced with a space 
    #so word boundaries are respected
    proc = proc.replace("\xa0"," ")
    
    #some articles have 2 letter abbreviations separated with periods (a.m., U.S.)
    #this finds them and removes the periods so they can be considered as n-grams
    #there is likely a better way of doing this but I tried
    match = re.findall(r'[A-Za-z]\.[A-Za-z]\.',proc)
    if(len(match)>0):
        for m in match:
            proc = proc.replace(m, m.replace('.',''))
            
    #This removes periods from acronyms longer than 2 characters
    proc = re.sub(r'(?<!\w)([A-Z])\.', '',proc)
            
    #This removes numbers and special characters except periods, colons, question marks and spaces
    proc = re.sub('[^A-Za-z.?: ]+', '', proc)
    
    #There is probably a more elegant way of doing this, but this ensures that sentence
    #boundaries are respected by putting in a space instead (otherwise the words combine)
    #However this operation can introduce extra whitespace which needs to be removed as well
    proc = re.sub(r'[.?:]+', " ",proc)
    proc = re.sub("\s\s+" , " ", proc)
    proc = proc.strip()
    
    #now we convert all the words or ngrams to lower case, unless the ngram is an acronym
    #then we know it is a different ngram and should be stored in upper case
    newProc = []
    for s in proc.split(" "):
        if(s == s.upper() and len(s) > 1 and len(s) < 6):
            newProc.append(s)
        else:
            newProc.append(s.lower())
    proc = " ".join(newProc)
    
    return proc

if __name__ == "__main__":
    # wrap the program in a try/except block in case there are errors
    try:
        main()
    except Exception as e:
        print(e)
        print(traceback.format_exc())