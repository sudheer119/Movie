import requests
from bs4 import BeautifulSoup
from nltk.tokenize.toktok import ToktokTokenizer
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
def getMovieDetails(url):
    data = {}
    r = requests.get(url=url)
    soup = BeautifulSoup(r.text, 'html.parser')
    title = soup.find('title')
    data["title"] = title.string
    ratingValue = soup.find("span", {"itemprop" : "ratingValue"})
    data["ratingValue"] = ratingValue.string
    ratingCount = soup.find("span", {"itemprop" : "ratingCount"})
    data["ratingCount"] = ratingCount.string
    titleName = soup.find("div",{'class':'titleBar'}).find("h1")
    data["name"] = titleName.contents[0].replace(u'\xa0', u'')
    subtext = soup.find("div",{'class':'subtext'})
    data["subtext"] = ""
    for i in subtext.contents:
        data["subtext"] += i.string.strip()
    summary_text = soup.find("div",{'class':'summary_text'})
    data["summary_text"] = summary_text.string.strip()
    credit_summary_item = soup.find_all("div",{'class':'credit_summary_item'})
    data["credits"] = {}
    for i in credit_summary_item:
        item = i.find("h4")
        names = i.find_all("a")
        data["credits"][item.string] = []
        for i in names:
            data["credits"][item.string].append({
                "link": i["href"],
                "name": i.string
            })
    return data

def idratingAMFAge(imdbinputtitle) :
    imdbtitlelist=imdbinputtitle.split()
    if len(imdbtitlelist)==1 :
        imdbtitletag=imdbtitlelist[0]
    else :
        imdbtitletag='+'.join(imdbtitlelist)
    imdbsearchtitle = 'https://www.imdb.com/find?s=tt&q='+imdbtitletag+'&ref_=nv_sr_sm'
    imbdr = requests.get(url=imdbsearchtitle)
    imdbmovies_soup = BeautifulSoup(imbdr.text, 'html.parser')
    imdbmovie_tags = imdbmovies_soup.find_all('a', attrs={'class': None},limit=10)
    for i,link in enumerate(imdbmovie_tags):
        if link.get_text()==imdbinputtitle:
            titleandid=link.get('href')
            break
    imbdmovielink = "https://www.imdb.com"+titleandid

    allratlink = 'https://www.imdb.com'+titleandid+'ratings?demo=imdb_users'
    maleratlink = 'https://www.imdb.com'+titleandid+'ratings?demo=males'
    femaleratlink = 'https://www.imdb.com'+titleandid+'ratings?demo=females'

    allrat_r = requests.get(url=allratlink)
    allrat_soup = BeautifulSoup(allrat_r.text, 'html.parser')
    allrat_tags = allrat_soup.find_all('div', attrs={'class': 'allText'},limit=1)
    listAll=[]

    malerat_r = requests.get(url=maleratlink)
    malerat_soup = BeautifulSoup(malerat_r.text, 'html.parser')
    malerat_tags = malerat_soup.find_all('div', attrs={'class': 'allText'},limit=1)
    listMale = []

    femalerat_r = requests.get(url=femaleratlink)
    femalerat_soup = BeautifulSoup(femalerat_r.text, 'html.parser')
    femalerat_tags = femalerat_soup.find_all('div', attrs={'class': 'allText'},limit=1)
    listFemale = []
    for i,link in enumerate(allrat_tags):
        listAll.append(link.get_text().strip().split())

    for i,link in enumerate(malerat_tags):
        listMale.append(link.get_text().strip().split())

    for i,link in enumerate(femalerat_tags):
        listFemale.append(link.get_text().strip().split())

    Alldf= pd.DataFrame({'rating':[],'percentage':[],'votes':[]})
    Maledf = pd.DataFrame({'rating': [], 'percentage': [], 'votes': []})
    Femaledf = pd.DataFrame({'rating': [], 'percentage': [], 'votes': []})
    for i in range(15,45,3) :
        Alldf.loc[len(Alldf.index)] = [listAll[0][i],listAll[0][i+1],listAll[0][i+2]]
    for i in range(15,45,3) :
        Maledf.loc[len(Maledf.index)] = [listMale[0][i],listMale[0][i+1],listMale[0][i+2]]
    for i in range(15,45,3) :
        Femaledf.loc[len(Femaledf.index)] = [listFemale[0][i],listFemale[0][i+1],listFemale[0][i+2]]


    AlldfDemo = pd.DataFrame({'All Ages':[],'<18':[],'18-29':[],'30-44':[],'45+':[]})
    for i in range(62,93,11) :
        AlldfDemo.loc[len(AlldfDemo.index)] = [[listAll[0][i],listAll[0][i+1]],[listAll[0][i+2],listAll[0][i+3]],
                                               [listAll[0][i+4],listAll[0][i+5]],[listAll[0][i+6],listAll[0][i+7]],[listAll[0][i+8],listAll[0][i+9]]]

    return [titleandid,imbdmovielink,Alldf,Maledf,Femaledf,AlldfDemo]




def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

def remove_stopwords(text, is_lower_case=False):
    stopword_list = nltk.corpus.stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def numto_table(report):
    report = report.splitlines()
    res = []
    res.append(['']+report[0].split())
    for row in report[2:-2]:
       res.append(row.split())
    lr = report[-1].split()
    res.append([' '.join(lr[:3])]+lr[3:])
    return np.array(res)

def imdbreviews(imdbtitletag) :
    imdbreviewdf = pd.DataFrame({'review': [], 'sentiment': []})
    for i in range(1,11) :
        reviewurl='https://www.imdb.com'+imdbtitletag+'reviews?sort=helpfulnessScore&dir=desc&ratingFilter='+str(i)
        reviewr = requests.get(url=reviewurl)
        review_soup = BeautifulSoup(reviewr.text, 'html.parser')
        review_tags = review_soup.find_all('div', attrs={'class': 'text show-more__control'})
        for j in review_tags :
            if i<=5 :
                tem='negative'
            else :
                tem='positive'
            imdbreviewdf.loc[len(imdbreviewdf.index)] = [j.get_text(),tem]
    imdbreviewdf['review'] = imdbreviewdf['review'].apply(remove_stopwords)
    lb = LabelBinarizer()
    imdbreviewdf['sentiment'] = lb.fit_transform(imdbreviewdf['sentiment'])
    X_train, X_test, y_train, y_test = train_test_split(imdbreviewdf['review'], imdbreviewdf['sentiment'],
                                                       test_size=0.33, random_state=42)
    cv = CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1, 3))
    cv_train_reviews = cv.fit_transform(X_train)
    cv_test_reviews = cv.transform(X_test)
    train_sentiments = y_train
    test_sentiments = y_test
    lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
    lr_bow = lr.fit(cv_train_reviews, train_sentiments)
    lr_bow_predict = lr.predict(cv_test_reviews)
    lr_bow_score = accuracy_score(test_sentiments, lr_bow_predict)
    lr_bow_report = classification_report(test_sentiments, lr_bow_predict, target_names=['Positive', 'Negative'])
    return numto_table(lr_bow_report)






















