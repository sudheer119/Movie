from django.shortcuts import render
from django.http import HttpResponse

import requests
from bs4 import BeautifulSoup
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import io
import urllib, base64
import numpy as np
import json
# Create your views here.

def index_view(request,*args,**kwargs) :
    uri1=''
    uri2=''
    uri3 = ''
    uri4 = ''
    uri5 = ''
    data={}
    imagelink=''
    table1=pd.DataFrame({})
    table2=pd.DataFrame({})
    table3=pd.DataFrame({})
    table4=pd.DataFrame({})
    table5 = pd.DataFrame({})
    lr_bow_report=0

    imdbinputtitle=request.POST.get('q')
    if imdbinputtitle!=None :
        imdbtitlelist = imdbinputtitle.split()
        if len(imdbtitlelist) == 1:
            imdbtitletag = imdbtitlelist[0]
        else:
            imdbtitletag = '+'.join(imdbtitlelist)
        imdbsearchtitle = 'https://www.imdb.com/find?s=tt&q=' + imdbtitletag + '&ref_=nv_sr_sm'
        imbdr = requests.get(url=imdbsearchtitle)
        imdbmovies_soup = BeautifulSoup(imbdr.text, 'html.parser')
        imdbmovie_tags = imdbmovies_soup.find_all('a', attrs={'class': None}, limit=10)
        titleandid=''
        for i, link in enumerate(imdbmovie_tags):
            if link.get_text() == imdbinputtitle:
                titleandid = link.get('href')
                break
        imbdmovielink = "https://www.imdb.com" + titleandid

        allratlink = 'https://www.imdb.com' + titleandid + 'ratings?demo=imdb_users'
        maleratlink = 'https://www.imdb.com' + titleandid + 'ratings?demo=males'
        femaleratlink = 'https://www.imdb.com' + titleandid + 'ratings?demo=females'

        allrat_r = requests.get(url=allratlink)
        allrat_soup = BeautifulSoup(allrat_r.text, 'html.parser')
        allrat_tags = allrat_soup.find_all('div', attrs={'class': 'allText'}, limit=1)
        listAll = []

        malerat_r = requests.get(url=maleratlink)
        malerat_soup = BeautifulSoup(malerat_r.text, 'html.parser')
        malerat_tags = malerat_soup.find_all('div', attrs={'class': 'allText'}, limit=1)
        listMale = []

        femalerat_r = requests.get(url=femaleratlink)
        femalerat_soup = BeautifulSoup(femalerat_r.text, 'html.parser')
        femalerat_tags = femalerat_soup.find_all('div', attrs={'class': 'allText'}, limit=1)
        listFemale = []
        for i, link in enumerate(allrat_tags):
            listAll.append(link.get_text().strip().split())

        for i, link in enumerate(malerat_tags):
            listMale.append(link.get_text().strip().split())

        for i, link in enumerate(femalerat_tags):
            listFemale.append(link.get_text().strip().split())

        Alldf = pd.DataFrame({'rating': [], 'percentage': [], 'votes': []})
        Maledf = pd.DataFrame({'rating': [], 'percentage': [], 'votes': []})
        Femaledf = pd.DataFrame({'rating': [], 'percentage': [], 'votes': []})
        for i in range(15, 45, 3):
            Alldf.loc[len(Alldf.index)] = [listAll[0][i], listAll[0][i + 1], listAll[0][i + 2]]
        for i in range(15, 45, 3):
            Maledf.loc[len(Maledf.index)] = [listMale[0][i], listMale[0][i + 1], listMale[0][i + 2]]
        for i in range(15, 45, 3):
            Femaledf.loc[len(Femaledf.index)] = [listFemale[0][i], listFemale[0][i + 1], listFemale[0][i + 2]]
        chartonepoints = []
        AlldfDemo = pd.DataFrame({'All_Ages': [], 'lessthan18': [], 'between18and29': [], 'between30and44': [], 'above45': []})
        i=62
        k=91
        while i<k:
            tem=[]
            j=0
            q=10
            qwe = []
            while j<q :
                te1=listAll[0][i+j]
                te2=listAll[0][i + j+1]
                if te1=='-' :
                    tem.append([0, 0])
                    j=j+1
                    q=q-2
                    qwe.append(0)
                else :
                    temp = te2
                    while True:
                        temp = temp.replace(',', '')
                        if ',' not in temp:
                            break
                    tem.append([float(te1), int(temp)])
                    j=j+2
                    qwe.append(int(temp))
            chartonepoints.append(qwe)
            AlldfDemo.loc[len(AlldfDemo.index)]=tem
            i=i+11
        data = {}
        r = requests.get(url=imbdmovielink)
        soup = BeautifulSoup(r.text, 'html.parser')
        title = soup.find('title')
        imagetag = soup.find_all('div', attrs={'class': 'poster'}, limit=1)
        imagelink=''
        for i in imagetag:
            imagelink = i.img['src']
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
                    "name": i.string
                })

        imdbreviewdf = pd.DataFrame({'review': [], 'sentiment': []})
        for i in range(1, 11):
            reviewurl = 'https://www.imdb.com' + titleandid + 'reviews?sort=helpfulnessScore&dir=desc&ratingFilter=' + str(i)
            reviewr = requests.get(url=reviewurl)
            review_soup = BeautifulSoup(reviewr.text, 'html.parser')
            review_tags = review_soup.find_all('div', attrs={'class': 'text show-more__control'})
            for j in review_tags:
                if i <= 5:
                    tem = 'negative'
                else:
                    tem = 'positive'
                imdbreviewdf.loc[len(imdbreviewdf.index)] = [j.get_text(), tem]
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
        lr_bow_report = accuracy_score(test_sentiments, lr_bow_predict)
        print(lr_bow_report)

        chartonelabel=list(AlldfDemo.columns)
        width = 0.2
        x = np.arange(5)
        plt.bar(x - 0.2, chartonepoints[0], width, color='red')
        plt.bar(x, chartonepoints[1], width, color='orange')
        plt.bar(x + 0.2, chartonepoints[2], width, color='green')
        plt.xticks(x, chartonelabel)
        plt.xlabel("age group")
        plt.ylabel("rating")
        plt.legend(['all','male','female'])
        fig1 = plt.gcf()
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png')
        buf1.seek(0)
        string1 = base64.b64encode(buf1.read())
        uri1 = urllib.parse.quote(string1)

        y2 = [chartonepoints[1][0], chartonepoints[2][0]]
        mylabels2 = ["male", 'female']
        plt.pie(y2, labels=mylabels2)
        fig2 = plt.gcf()
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png')
        buf2.seek(0)
        string2 = base64.b64encode(buf2.read())
        uri2 = urllib.parse.quote(string2)

        json_records1 = Alldf.reset_index().to_json(orient='records')
        table1 = []
        table1 = json.loads(json_records1)

        json_records2 = Maledf.reset_index().to_json(orient='records')
        table2 = []
        table2 = json.loads(json_records2)

        json_records3 = Femaledf.reset_index().to_json(orient='records')
        table3 = []
        table3 = json.loads(json_records3)

        json_records4 = AlldfDemo.reset_index().to_json(orient='records')
        table4 = []
        table4 = json.loads(json_records4)


        json_records5 = imdbreviewdf.head(5).reset_index().to_json(orient='records')
        table5 = []
        table5 = json.loads(json_records5)
    context={
        'graph1':uri1,
        'data1':data,
        'table1' : table1,
        'table2' : table2,
        'table3' : table3,
        'table4' : table4,
        'table5' : table5,
        'imagelink' :imagelink,
        'lr_bow_report' : lr_bow_report
    }

    print(context)
    return render(request,"index.html",context)

def about_view(request,*args,**kwargs) :
    return render(request,"about.html",{})

def contact_view(request,*args,**kwargs) :
    return render(request,"contact.html",{})

def result_view(request,*args,**kwargs) :
    return render(request,"result.html",{})