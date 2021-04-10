from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import json
import requests
from newsapi.newsapi_client import NewsApiClient
from bs4 import BeautifulSoup
from mysite.allinone import *
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('punkt')
class Object:
    def toJSON(self):
        return json.dumps(self,default = lambda o: o.__dict__,sort_keys=True,indent=4)

my_list = []
json_list = []
@api_view(['GET'])
def topheadlines(request):
    newsapi = NewsApiClient(api_key='e714e075a7534f85b7e0bdfd2330c611')
    top_headlines = newsapi.get_top_headlines(
                                        category='business',
                                        language='en',
                                        country='us')
    #top_headlines = top_headlines['articles']

    return Response(top_headlines)



@api_view(['GET'])
def customtopheadlines(request):
    json_content = []
    newsapi = NewsApiClient(api_key='e714e075a7534f85b7e0bdfd2330c611')
    top_headlines = newsapi.get_top_headlines(
                                        category='business',
                                        language='en',
                                        country='us')
    top_headlines = top_headlines['articles']
    for i in range(1,2):
        news_object = top_headlines[i]
        results,temp,temp_2 = assign_roles(news_object['url'],2.0, 0.25)
        custom_object = {
            "hero" : results[0],
            "victim" : results[1],
            "villian" : results[2],
            "source" : news_object["source"]["name"],
            "author" : news_object["author"],
            "title"  : news_object['title'],
            "urlToImage"  : news_object["urlToImage"],
            "shortdescription" : news_object['description'],
            "url" : news_object['url'],
        }
        json_content.append(custom_object)

    return Response(json_content)


@api_view(['GET'])
def searchresults(request):
    user_query = request.GET['search']
    newsapi = NewsApiClient(api_key='e714e075a7534f85b7e0bdfd2330c611')
    all_articles = newsapi.get_everything(q=user_query,
                                        language='en',
                                        sort_by='relevancy')
        
    return Response(all_articles)


@api_view(['GET'])
def customsearchresults(request):
    user_query = request.GET['search']
    newsapi = NewsApiClient(api_key='e714e075a7534f85b7e0bdfd2330c611')
    all_articles = newsapi.get_everything(q=user_query,
                                        language='en',
                                        sort_by='relevancy')
    all_articles = all_articles['articles']
    json_content = []
    for i in range(len(all_articles)):
        news_object = all_articles[i]
        
        # r1 = requests.get(news_object['url'])
        # text = r1.content
        # soup = BeautifulSoup(text, 'html.parser')
        # paragraph_list = soup.find_all('p')
        # whole_content = ""
        # json_content = []
        # for item in range(len(paragraph_list)):
        #     whole_content = whole_content + " " + paragraph_list[item].get_text()
        
        custom_object = {
            "heroes" : "To be decided",
            "victim" : "To be decided",
            "villian" : "To be decided",
            "source" : news_object["source"]["name"],
            "author" : news_object["author"],
            "title"  : news_object['title'],
            "shortdescription" : news_object['description'],
            "urlToImage":news_object["urlToImage"],
            "url" : news_object["url"],
            
        }
        json_content.append(custom_object)
        
    return Response(json_content)



@api_view(['GET'])
def detailnews(request):
    url = request.GET["url"]
    r1 = requests.get(url)
    text = r1.content
    soup = BeautifulSoup(text, 'html.parser')
    paragraph_list = soup.find_all('p')
    whole_content = ""
    json_content = []
    for item in range(len(paragraph_list)):
        whole_content = whole_content + " " + paragraph_list[item].get_text()
    detail_news = {
        "url" : url,
        "fullcontent" : whole_content
    }
    return Response(detail_news)


