from django.contrib import admin
from django.urls import path,include
from .views import topheadlines,customtopheadlines,searchresults,customsearchresults,detailnews
urlpatterns = [
    path('topheadlines/',topheadlines),
    path('customtopheadlines/',customtopheadlines),
    path('searchresults/',searchresults),
    path('customsearchresults/',customsearchresults),
    path('detailnews/',detailnews),
]
