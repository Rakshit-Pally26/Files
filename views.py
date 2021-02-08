import json
from django.shortcuts import render, redirect
from django.http import HttpResponse
import numpy as np
import re
import pandas as pd
from django.http import HttpResponse, HttpResponseBadRequest
from django.core.files.storage import FileSystemStorage
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from fais import usgsgatherer as usgs
from fais import twittergatherer as tw
from .models import realtime_usgs_data, post_local, us_state,twitter_data, historical_usgs_data, flood_station_list,select_state, tweet_streamer,peak_rate, Data
from django_pandas.io import read_frame
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.urls import reverse
from .forms import USStateForms, PhotoUploadForm
import csv
import datetime
from django import db
import pymongo
import urllib
from math import floor
import math
import statistics
from scipy.stats import t
import matplotlib.pyplot as plt
from google.cloud import vision
import PIL
import requests
import io, os
import base64
from geemap import eefolium as gmap
from bs4 import BeautifulSoup
from django.contrib.staticfiles.storage import staticfiles_storage
from django.contrib.sessions.backends.db import SessionStore as DBStore
from django.contrib.sessions.base_session import AbstractBaseSession
from django.db import models
from django.template.loader import get_template 
from firstSite.utils import render_to_pdf 
import cv2
import tensorflow as tf
import requests
import os
from tqdm import tqdm
from bs4 import BeautifulSoup as bs
from urllib.parse import urlparse
from zipfile import ZipFile
from django.utils.encoding import smart_str
import os.path
from django.conf import settings
from reliability.Fitters import Fit_Everything
from reliability.Fitters import Fit_Weibull_2P
from reliability.Fitters import Fit_Exponential_2P
from reliability.Fitters import Fit_Gamma_2P
from reliability.Fitters import Fit_Lognormal_2P
from reliability.Fitters import Fit_Loglogistic_2P
from reliability.Fitters import Fit_Normal_2P
from reliability.Fitters import Fit_Gumbel_2P
from reliability.Fitters import Fit_Beta_2P
import datetime
from suds.client import Client
from pandas import Series
import matplotlib.dates as mdates

def home(request):
    return render(request, 'faisApp/index.html')

def saveCSVRealTime(request):

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="RealTimeFloodData.csv"'
    writer = csv.writer(response)
    writer.writerow(['Station', 'ID', 'Latitude', 'Longitude', 'Flow', 'Stage', 'Average Gauge height', 'URL'])
    try:
        qs = realtime_usgs_data.objects.all()
        if qs.count() < 1:
            return HttpResponse("<script>alert('There is no data select, please select the state before download to csv file')</script>")
 
        for item in qs:
            writer.writerow([item.station, item.id_num, item.latitude, item.longitude, item.flow, item.stage, item.avg_ht,item.url])
    except:
        return HttpResponse("<script>alert('There is no data select, please select the state before download to csv file')</script>")
    return response

def saveCSVHist(request):
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="Hist.csv"'
    writer = csv.writer(response)
    writer.writerow(['date and time', 'discharge', 'gage height'])
    try:
        qs = historical_usgs_data.objects.all()
        if qs.count() < 1:
            return HttpResponse("<script>alert('There is no data select, please select the state before download to csv file')</script>")
 
        for item in qs:
            writer.writerow([item.date, item.discharge, item.gage_height])
    except:
        return HttpResponse("<script>alert('There is no data select, please select the state before download to csv file')</script>")
    return response

def saveCSVTwitter(request):    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="twitter.csv"'
    writer = csv.writer(response)
    writer.writerow(['Tweets', 'Datetime', 'Source url', 'Image url', 'Sentiment'])
    try:
        qs = twitter_data.objects.all()
        if qs.count() < 1:
            return HttpResponse("<script>alert('There is no data select, please select the state before download to csv file')</script>")
 
        for item in qs:
            writer.writerow([item.tweets, item.date_posted, item.source_url, item.image_url, item.Sentiment])
    except:
        return HttpResponse("<script>alert('There is no data select, please select the state before download to csv file')</script>")
    return response


def saveCSVTwitterStream(request):    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="twitter.csv"'
    writer = csv.writer(response)
    writer.writerow(['Tweets', 'Location', 'User', 'Date'])
    try:
        qs = twitter_data.objects.all()
        if qs.count() < 1:
            return HttpResponse("<script>alert('There is no data select, please select the state before download to csv file')</script>")
 
        for item in qs:
            writer.writerow([item.text, item.geo, item.user, item.date])
    except:
        return HttpResponse("<script>alert('There is no data select, please select the state before download to csv file')</script>")
    return response



def historical(request):
    state_list = us_state.objects.all()
    flood_list = flood_station_list.objects.all()
    qs = ""
    context = {
        'hist': qs,
        'station': flood_list,
        'states_list': state_list,
    }
    return render(request, 'faisApp/past-flood.html',context)

def HydroShare(request):
    state_list = us_state.objects.all()
    flood_list = flood_station_list.objects.all()
    qs = ""
    context = {
        'hist': qs,
        'station': flood_list,
        'states_list': state_list,
    }
    return render(request, 'faisApp/hs-flood.html',context)


def execute(request,command):
    state_list = us_state.objects.all()
    qs = realtime_usgs_data.objects.all()
    current_state = request.GET['value']
    try: 
        model = usgs.get_realtime_flood_dataframe(current_state)
        df =  read_frame(qs)
        df.drop(df.index, inplace=True)
        df, model = [d.reset_index(drop=True) for d in (df, model)]
        df['station'] = model['station_nm']
        df['id_num'] = model['site_no']
        df['latitude'] = model['dec_lat_va']
        df['longitude'] = model['dec_long_va']
        df['stage'] = model['stage']
        df['flow'] = model['flow']
        df['avg_ht'] = model['class']
        df['url'] = model['url']
        realtime_usgs_data.objects.all()._raw_delete('default')
        instances = []
        for index, row in df.iterrows():
            if row is not None:
                try:
                    flow_value = float((row['flow']))
                except ValueError:
                    flow_value = 0.0
                try:
                    stage_value = float(row['stage'])
                except ValueError:
                    stage_value = 0.0
                temp = realtime_usgs_data(station=row['station'],id_num=row['id_num'],latitude=float(row['latitude']),longitude=float(row['longitude']),stage=stage_value,flow=flow_value,avg_ht=row['avg_ht'],url=(row['url']))
                instances.append(temp)
        realtime_usgs_data.objects.bulk_create(instances)
    except:
        return HttpResponse("<script>alert('There is a problem with a USGS Server, please contact the administation for more information')</script>")
    
    qs = realtime_usgs_data.objects.all()
    state_list = us_state.objects.all()
    df =  read_frame(qs)
    avg_lat = df['latitude'].mean()
    avg_log = df['longitude'].mean()
    context = {
        'realtime': qs,
        'states_list': state_list,
        'avg_lat': avg_lat,
        'avg_log': avg_log,
    }
    return render(request, "faisApp/realtime-flood-tablemap.html",context )


def executetweet(request,command):
    username = request.GET['username']
    queary = request.GET['queary']
    if username == '':
        username = None
    if queary == '':
        queary = None
    if username == None and queary == None:
        print("return")
        return HttpResponse("<script>alert('please insert either username or queary')</script>")
    startdate = request.GET['startdate']
    enddate = request.GET['enddate']
    maxtweet = request.GET['maxtweet']
    try :
        maxtweet = int(maxtweet)
    except ValueError:
        maxtweet = 0
    try:
        tweets_criteria = tw.create_twitter_criteria(username,queary,startdate, enddate,maxtweet)
        tweets = tw.get_tweets_dataframe(tweets_criteria)
        instances = []
        for index, row in tweets.iterrows():
            if row is not None:
                image_urls = ""
                tweet = ""
                source = "https://twitter.com/home?lang=en"
                dt = datetime.datetime.now()
                try:
                    sentiment = int((row['sentiment']))
                except ValueError:
                    sentiment = 0
                if row['image_url'] is not None:
                    image_urls = row['image_url']
                if row['source'] is not None:
                    source = row['source']
                if row['Tweets'] is not None:
                    tweet = row['Tweets']
                try:
                    dt = row['date']
                except:
                    dt = datetime.datetime.now()
                temp = twitter_data(tweets=tweet,date_posted=dt,source_url=source,image_url=image_urls,Sentiment=sentiment)

                instances.append(temp)
        db.reset_queries()
        if len(instances) > 0:
            twitter_data.objects.all()._raw_delete('default')
            twitter_data.objects.bulk_create(instances)
        else:
            return HttpResponse("<script>alert('Sorry We cannot retrive the data from Twitter please ensure that you enter the correct information or try again later')</script>")

    except:
        return HttpResponse("<script>alert('Sorry We cannot retrive the data from Twitter please ensure that you enter the correct information or try again later')</script>")
    qs = twitter_data.objects.all()

    context = {
        'twitter': qs,
    }

    return render(request, "faisApp/tweet-table.html",context )

def executehistorical(request,command):
    station = request.GET['station']
    startdate = request.GET['startdate']
    enddate = request.GET['enddate']
    datatype_list = ["00065","00060"]
    current_state = select_state.objects.all()
    criteria = usgs.create_usgs_criteria(region=current_state[0].state_code,station=station, parameters=datatype_list,since=startdate,until=enddate)
    try:
        hist_data = usgs.get_flood_data_dataframe(criteria)
        instances = []
        historical_usgs_data.objects.all()._raw_delete('default')
        for index, row in hist_data.iterrows():
            if row is not None:
                discharge = 0.0
                gageheight = 0.0

                if row['datetime'] is not None:
                    datesearch = row['datetime']
                try:
                    discharge = float(row['Discharge (ft^3/S)'])
                except:
                    discharge = 0.0
                try:
                    gageheight = float(row['Gage height (ft)'])
                except:
                    gageheight = 0.0
                temp = historical_usgs_data(date=datesearch, discharge=discharge,gage_height=gageheight )
                instances.append(temp)
        if len(instances) > 0:
            historical_usgs_data.objects.bulk_create(instances)
    except:
        return HttpResponse("<script>alert('Sorry We cannot retrived the data from USGS please ensure that you enter the correct information or tried again later')</script>")
    qs = historical_usgs_data.objects.all()
    size = int(historical_usgs_data.objects.count())
    indexing =int(floor(size/15))
    print("index " + str(indexing) +  "  size " + str(size))
    graphs_date = []
    graph_discharge = []
    graphs_gage = []
    for i in range(0,15):
        graphs_date.append(qs[i * indexing].date)
        graph_discharge.append(qs[i * indexing].discharge)
        graphs_gage.append(qs[i * indexing].gage_height)
    context = {
        'hist': qs,
        'graph_date':graphs_date,
        'graph_discharge': graph_discharge,
        'graphs_gage': graphs_gage,
    }

    return  render(request, "faisApp/hist-flood-table.html",context )


def executehydroshare(request,command):
    station = request.GET['station']
    startdate = request.GET['startdate']
    enddate = request.GET['enddate']
    wsdlURL = 'http://hydroportal.cuahsi.org/nwisuv/cuahsi_1_1.asmx?WSDL'
    siteCode = 'NWISUV:' + station # Can change to different site if desired.  Format: 'NWISUV:########'
    variableCode = 'NWISUV:00060'
    variableCode1 = 'NWISUV:00065'
    beginDate = startdate
    endDate = enddate  # Can change to less recent date if desired. 
    NWIS = Client(wsdlURL).service
    response = NWIS.GetValuesObject(siteCode, variableCode, beginDate, endDate)
    flow = []  # The values
    dates = []  # The dates
    gage_ht = []
    dates1 = []
    values = response.timeSeries[0].values[0].value
    NWIS1 = Client(wsdlURL).service
    response1 = NWIS1.GetValuesObject(siteCode, variableCode1, beginDate, endDate)
    values1 = response1.timeSeries[0].values[0].value
    for v in values:
        flow.append(float(v.value))
        dates.append(v._dateTime)
    for v1 in values1:
        gage_ht.append(float(v1.value))
        dates1.append(v1._dateTime)
    siteName = response.timeSeries[0].sourceInfo.siteName
    latitude = response.timeSeries[0].sourceInfo.geoLocation[0].latitude
    longitude = response.timeSeries[0].sourceInfo.geoLocation[0].longitude
    mylist = zip(flow , dates, gage_ht)

    ts = Series(flow, index=dates)
    mean_discharge = ts.resample(rule='24H', base=0).mean()
    max_discharge = ts.resample(rule='24H', base=0).max()
    min_discharge = ts.resample(rule='24H', base=0).min()   
    my_stringIObytes = io.BytesIO()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)  # arguments for add_subplot - add_subplot(nrows, ncols, plot_number)

    # Plotting series data
    ts.plot(color='powderblue', linestyle='solid', label='15-minute streamflow values')
    mean_discharge.plot(color='green', linestyle='solid', label='Average streamflow values')
    max_discharge.plot(color='red', linestyle='solid', label='Maximum streamflow values')
    min_discharge.plot(color='blue', linestyle='solid', label='Minimum streamflow values')

    # Formatting Axes
    ax.set_ylabel('Flow, cubic feet per second')
    ax.set_xlabel('Date')
    ax.grid(True)
    ax.set_title(siteName)

    # Adding a legend
    legend = ax.legend(loc='upper left', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.95')
    for label in legend.get_texts():
        label.set_fontsize('small')

    for label in legend.get_lines():
        label.set_linewidth(1.8)
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode('ascii')
    html_img1 =  'data:image/png;base64,{0}'.format(my_base64_jpgData)

    my_stringIObytes1 = io.BytesIO()
    ts = Series(gage_ht, index=dates1)
    hourlyTotDisAvg = ts.resample(rule='1D', base=0).mean()
    hourlyTotDisMax = ts.resample(rule='1D', base=0).max()
    hourlyTotDisMin = ts.resample(rule='1D', base=0).min()

    # Create a figure object and add a subplot
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1, 1, 1)  # arguments for add_subplot - add_subplot(nrows, ncols, plot_number)

    # Call the plot() methods on the series object to plot the data
    ts.plot(color='yellow', linestyle='solid', label='15-minute streamflow values', alpha=0.5, linewidth=1.8)
    hourlyTotDisAvg.plot(color='green', linestyle='solid', label='Daily avg flows',
                        marker = 'o', ms=2, linewidth =0.75)
    hourlyTotDisMax.plot(color='red', linestyle='solid', label='Daily max flows',
                        marker = 'o', ms=2, linewidth =0.75)
    hourlyTotDisMin.plot(color='blue', linestyle='solid', label='Daily min flows',
                        marker = 'o', ms=2, linewidth =0.75)
    # Set some properties of the subplot to make it look nice
    ax.set_ylabel('Gage Height, in feet')
    ax.set_xlabel('Date (YYYY-MM-DD)')
    ax.grid(True)
    ax.set_title('Daily Max, Min, & Avg Gage Height for: ' + siteName + ', ' + siteCode)
    ax.set_xlim(beginDate, endDate) #set limits with date variables
    # Add a legend with some customizations
    legend = ax.legend(loc='upper left', shadow=True)
    fig.autofmt_xdate()  # use auto-formatter to enable accurate date representation with mouse
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=15))  # set ticks interval for every 15 days.

    # Create a frame around the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.95')

    # Set the font size in the legend
    for label in legend.get_texts():
        label.set_fontsize('small')

    for label in legend.get_lines():
        label.set_linewidth(1.8)  # the legend line width
    plt.savefig(my_stringIObytes1, format='jpg')
    my_stringIObytes1.seek(0)
    my_base64_jpgData1 = base64.b64encode(my_stringIObytes1.read()).decode('ascii')
    html_img2 =  'data:image/png;base64,{0}'.format(my_base64_jpgData1)

    context = {
        'data': mylist,
        'siteName' : siteName,
        'latitude' : latitude,
        'longitude' : longitude,
        'StationNumber' : station,
        'graph_dis' : html_img1,
        'graph_gage' : html_img2
    }

    return  render(request, "faisApp/hs-flood-table.html",context)



def stateselecthist(request,command):
    current_state_hist = request.GET['value']
    state_list =  usgs.get_station_list_dataframe(current_state_hist)
    select_state.objects.all()._raw_delete('default')
    select_state(state_code=current_state_hist, state_name="temp").save()
    instances = []
    for index, row in state_list.iterrows():
        if row is not None:
            temp = flood_station_list(station_number=row['site_no'], station_name=row['station_nm'])
            instances.append(temp)
    if len(instances) > 0:
        flood_station_list.objects.all()._raw_delete('default')
        flood_station_list.objects.bulk_create(instances)
    flood_list = flood_station_list.objects.all()

    context = {
        'station': flood_list,
    }
    return  render(request, "faisApp/station-list.html",context )

def readNWISSpeak(site):
    url = "https://nwis.waterdata.usgs.gov/usa/nwis/peak/?site_no=" + site + "&format=rdb"
    text = urllib.request.urlopen(url)
    df = pd.DataFrame()
    line_num = 0
    index = 0
    for line in text:
            line = line.decode('utf-8') 
            if line[0:7] == '#  USGS':
                s = line[17:]
                pass           
            if line[0] == '#':
                pass
            else:
                line_num += 1
                if line_num == 1:
                    temp = line.replace('\t', ',')
                    header = temp.split(',')
                    for col in header:
                        df[col] = None                       
                elif line_num == 2:
                    pass
                else:
                    temp = line.replace('\t', ',')
                    row = temp.split(',')
                    df.loc[index] = row
                    index += 1
    remove_col = ["peak_tm","peak_cd", "gage_ht_cd","year_last_pk","ag_dt", "ag_tm","ag_gage_ht", "ag_gage_ht_cd\r\n"]
    df.drop(columns=remove_col,inplace=True)
    return df,s

def floodDataAnalytics(station):
    annualpeak,s = readNWISSpeak(station)
    q = pd.to_numeric(annualpeak.iloc[:,3])
    n = len(q )
    r = n + 1 - q.rank(method='first') # highest Q has rank r = 1
    T = (n + 1)/r

    Ttick_array = [1.001,1.01,1.1,1.5,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50,60,70,80,90,100]
    Ttick = pd.DataFrame(Ttick_array)
    xtlab = [1.001,1.01,1.1,1.5,2,None,None,5,None,None,None,None,10,None,None,None,None,15,None,None,None,None,20,None,30,None,None,None,50,None,None,None,None,100]
    y = 0.0 - np.log(0.0-(np.log(1 - 1/(T))))
    yTick = 0.0 - np.log(0.0-(np.log(1 - 1/(Ttick))))
    xmin = min(min(y),min(yTick))
    xmax = max(yTick)
    KTtick = 0.0 - (math.sqrt(6) / math.pi) * (0.5772 + np.log(np.log(Ttick / (Ttick-1)))) 
    QTtick = statistics.mean(q)  + (KTtick * statistics.stdev(q))
    se = (statistics.stdev(q) * np.sqrt(1 + 1.14*KTtick + 1.1*(np.square(KTtick)))) / np.sqrt(n)
    LB = QTtick - t.ppf(0.975,  n - 1)*se
    UB = QTtick + t.ppf(0.975,  n - 1)*se

    the_max = np.max(UB)
    Qmax = np.max(QTtick)
    my_stringIObytes = io.BytesIO()
    fig, ax = plt.subplots()
    ax.scatter(y,q)
    ax.set_ylabel("Annual Peak Flow (cfs)")
    ax.set_xlabel("Return Period, T (year)")
    ax.plot(yTick.values, QTtick.values, color="black")
    ax.plot(yTick.values, LB.values, color="blue" )
    ax.plot(yTick.values, UB.values, color="red")
    plt.xticks(ticks=np.arange(len(yTick)), labels= xtlab)
    plt.ylim(bottom=0)
    plt.grid()
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode('ascii')
    return my_base64_jpgData, annualpeak, s


def floodFrequencyAnalysis(request,command):
    station = request.GET['station']
    instances = []
    select_state.objects.all()._raw_delete('default')
    try:
        response,peak,s1 = floodDataAnalytics(station)
        q = pd.to_numeric(peak.iloc[:,3])
        data1 = q.describe()
        html_img =  'data:image/png;base64,{0}'.format(response)
        for index, row in peak.iterrows():
            if row is not None:
                temp = peak_rate(site_no=row['site_no'], peak_dt=row['peak_dt'],peak_va=row['peak_va'],gage_ht=row['gage_ht'])
                instances.append(temp)
        
        if len(instances) > 0:
            peak_rate.objects.all()._raw_delete('default')
            peak_rate.objects.bulk_create(instances)
    except:
        return HttpResponse("<script>alert('Sorry We cannot retrived the data from USGS please ensure that you enter the correct information or tried again later')</script>")
    data = peak_rate.objects.all()
    s=[]
    s.append("Count : " + str(data1[0])) 
    s.append("Mean : " + str(round(data1[1],2)))
    s.append("Standard Deviation : " + str(round(data1[2],2)))
    s.append("Minimum value : " + str(data1[3]))
    s.append("Q1 (25%) : " + str(data1[4]))
    s.append("Median (50%) : " + str(data1[5]))
    s.append("Q3 (75%) : " + str(data1[6]))
    s.append("Maximum value : " + str(data1[7]))
    url = "https://waterdata.usgs.gov/nwis/inventory/?site_no=02147500&agency_cd=USGS"
    text = urllib.request.urlopen(url)
    for line in text:
        line = line.decode('utf-8')
        degree_sign = u"\N{DEGREE SIGN}"
        if 'Latitude' in line:
            line=line.replace('&#', degree_sign)
            line=line.replace('176;', '')
            s3=line[14:23]
            s2=line[43:52]
            latitude = 'Latitude : '+s3
            longitude = 'Longitude : '+s2
            latnlong= latitude + ', ' + longitude
    my_stringIObytes = io.BytesIO()
    wb = Fit_Weibull_2P(failures=q.values)
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode('ascii')
    html_img1 =  'data:image/png;base64,{0}'.format(my_base64_jpgData)

    my_stringIObytes1 = io.BytesIO()
    ep = Fit_Exponential_2P(failures=q.values)
    plt.savefig(my_stringIObytes1, format='jpg')
    my_stringIObytes1.seek(0)
    my_base64_jpgData1 = base64.b64encode(my_stringIObytes1.read()).decode('ascii')
    html_img2 =  'data:image/png;base64,{0}'.format(my_base64_jpgData1)

    my_stringIObytes2 = io.BytesIO()
    g = Fit_Gamma_2P(failures=q.values)
    plt.savefig(my_stringIObytes2, format='jpg')
    my_stringIObytes2.seek(0)
    my_base64_jpgData2 = base64.b64encode(my_stringIObytes2.read()).decode('ascii')
    html_img3 =  'data:image/png;base64,{0}'.format(my_base64_jpgData2)

    my_stringIObytes3 = io.BytesIO()
    ln = Fit_Lognormal_2P(failures=q.values)
    plt.savefig(my_stringIObytes3, format='jpg')
    my_stringIObytes3.seek(0)
    my_base64_jpgData3 = base64.b64encode(my_stringIObytes3.read()).decode('ascii')
    html_img4 =  'data:image/png;base64,{0}'.format(my_base64_jpgData3)

    my_stringIObytes4 = io.BytesIO()
    logist = Fit_Loglogistic_2P(failures=q.values)
    plt.savefig(my_stringIObytes4, format='jpg')
    my_stringIObytes4.seek(0)
    my_base64_jpgData4 = base64.b64encode(my_stringIObytes4.read()).decode('ascii')
    html_img5 =  'data:image/png;base64,{0}'.format(my_base64_jpgData4)

    my_stringIObytes5 = io.BytesIO()
    n = Fit_Normal_2P(failures=q.values)
    plt.savefig(my_stringIObytes5, format='jpg')
    my_stringIObytes5.seek(0)
    my_base64_jpgData5 = base64.b64encode(my_stringIObytes5.read()).decode('ascii')
    html_img6 =  'data:image/png;base64,{0}'.format(my_base64_jpgData5)

    my_stringIObytes6 = io.BytesIO()
    gu = Fit_Gumbel_2P(failures=q.values)
    plt.savefig(my_stringIObytes6, format='jpg')
    my_stringIObytes6.seek(0)
    my_base64_jpgData6 = base64.b64encode(my_stringIObytes6.read()).decode('ascii')
    html_img7 =  'data:image/png;base64,{0}'.format(my_base64_jpgData6)

    ax = Fit_Everything(failures=q.values, show_histogram_plot=False, show_probability_plot=False, show_PP_plot=False)
    result = ax.best_distribution_name
    params = ax.best_distribution.parameters
    context = {
        'html_img': html_img,
        'html_img1': html_img1,
        'html_img2': html_img2,
        'html_img3': html_img3,
        'html_img4': html_img4,
        'html_img5': html_img5,
        'html_img6': html_img6,
        'html_img7': html_img7,
        'table': data,
        'stat' : s,
        'StationNumber' : station,
        'StationName' : s1,
        'latnlong' : latnlong,
        'result' : result,
        'params' : params
    } 
    return render(request, 'faisApp/peak_flow.html', context)


def uploadAnalizer(request):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']='C:/Users/jpall/OneDrive/Documents/serviceaccountkey.json'
    vision_client = vision.ImageAnnotatorClient()
    form = PhotoUploadForm(request.POST, request.FILES or None)
    if form.is_valid():
        pic = request.FILES["file"]
        fs= FileSystemStorage()
        name= fs.save(pic.name, pic)
        file_name= fs.url(name)
        uploaded_file = "C:/Users/jpall/OneDrive/Desktop/Flood-Data-Analytics-App" + file_name
        image_file = io.open(uploaded_file, 'rb')
        content = image_file.read()
        CATEGORIES= ["depth", "flooding", "irrelevant", "pollution"]
        IMG_SIZE = 512
        img_array = cv2.imread(uploaded_file, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (512, 512))
        new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        model = tf.keras.models.load_model("C:/Users/jpall/CNN.model")
        image = vision.types.Image(content=content)
        response = vision_client.annotate_image({
        'image': image,
        'features': [{'type': vision.enums.Feature.Type.LABEL_DETECTION}],
        })
        file_dest="http://localhost:8000"+file_name
        labels = response.label_annotations
        print(labels)
        to_return = []
        for label in labels:
            label_return = floodlabel(label.description, str(round(label.score, 2)))
            to_return.append(label_return)
        context = {
            'pic': file_dest,
            'labels': to_return
        }
        return render(request, 'faisApp/image_analizer.html', context)
    return HttpResponseBadRequest("Image upload form not valid.")

def uploadAnalizer1(request):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']='C:/Users/jpall/OneDrive/Documents/serviceaccountkey.json'
    client = vision.ImageAnnotatorClient()
    form = PhotoUploadForm(request.POST, request.FILES or None)
    if form.is_valid():
        pic = request.FILES["file"]
        fs= FileSystemStorage()
        name= fs.save(pic.name, pic)
        file_name= fs.url(name)
        uploaded_file = "C:/Users/jpall/OneDrive/Desktop/Flood-Data-Analytics-App" + file_name
        image_file = io.open(uploaded_file, 'rb')
        content = image_file.read()
        CATEGORIES= ["depth", "flooding", "irrelevant", "pollution"]
        IMG_SIZE = 512
        img_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (512, 512))
        new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        model = tf.keras.models.load_model("C:/Users/jpall/CNN.model")
        print(new_array.shape)
        prediction = model.predict([new_array])
        print(CATEGORIES[prediction.index(max(prediction))])
        image = vision.types.Image(content=content)
        response = client.object_localization(image=image)   
        file_dest="http://localhost:8000"+file_name
        labels = response.localized_object_annotations
        #print(labels)
        print("hello")
        to_return = []
        for label in labels:
            label_return = floodlabel(label.description, str(round(label.score, 2)))
            to_return.append(label_return)
        context = {
            'pic': file_dest,
            'labels': to_return
        }
        return render(request, 'faisApp/image_analizer1.html', context)
    return HttpResponseBadRequest("Image upload form not valid.")



def realtime(request):
    qs = realtime_usgs_data.objects.all()
    state_list = us_state.objects.all()
    df =  read_frame(qs)
    avg_lat = df['latitude'].mean()
    avg_log = df['longitude'].mean()

    qs = ""
    context = {
        'realtime': qs,
        'states_list': state_list,
        'avg_lat': avg_lat,
        'avg_log': avg_log,
    }
    return render(request, 'faisApp/realtime-flood.html', context)


def floodcam(request):
    return render(request, 'faisApp/realtime-cam.html')

class floodlabel:
    def __init__(self, description, score):
        self.description = description
        self.score = score

def image_to_byte_array(image:PIL.Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
   
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def get_image_label_local(imgByteArray):
    vision_client = vision.ImageAnnotatorClient()
    temp_image = (imgByteArray)
    image = vision.types.Image(content=temp_image) 
    response = vision_client.annotate_image({
    'image': image,
    'features': [{'type': vision.enums.Feature.Type.LABEL_DETECTION}],
    })
    labels = response.label_annotations
    to_return = []
    for label in labels:
        label_return = floodlabel(label.description, str(round(label.score, 2)))
        to_return.append(label_return)
    return to_return

def get_image_label(vision_client, img):
    response = vision_client.annotate_image({
    'image': {'source': {'image_uri': img}},
    'features': [{'type': vision.enums.Feature.Type.LABEL_DETECTION}],
    })

    labels = response.label_annotations
    to_return = []
    for label in labels:
        label_return = floodlabel(label.description, str(round(label.score, 2)))
        to_return.append(label_return)
    return to_return



def floodAnalyzer(request):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']='C:/Users/jpall/OneDrive/Documents/serviceaccountkey.json'
    vision_client = vision.ImageAnnotatorClient()
    img_analized = [
        {
            'url' : 'http://b7b.hdrelay.com/cameras/fa96bb1e-426d-4b40-a820-251713325420/GetOneShot?size=800x450',
            'name' : 'Rocky Creek',
            'labels': get_image_label(vision_client,'http://b7b.hdrelay.com/cameras/fa96bb1e-426d-4b40-a820-251713325420/GetOneShot?size=800x450')
        },
        {
            'url' : 'http://b7b.hdrelay.com/cameras/2fb4ae88-446d-4632-a849-d426240ccca5/GetOneShot?size=800x450',
            'name' : 'Rocky Branch',
            'labels': get_image_label(vision_client,'http://b7b.hdrelay.com/cameras/2fb4ae88-446d-4632-a849-d426240ccca5/GetOneShot?size=800x450')
        },
        {
            'url' : 'http://b6b.hdrelay.com/cameras/42336a6b-5be8-443c-97c9-3d70da459e88/GetOneShot?size=800x450',
            'name' : 'Pee Dee River',
            'labels': get_image_label(vision_client,'http://b7b.hdrelay.com/cameras/2fb4ae88-446d-4632-a849-d426240ccca5/GetOneShot?size=800x450')
        },
        {
            'url' : 'http://b6b.hdrelay.com/cameras/403818bc-9e8d-4857-9eb0-a675c9668782/GetOneShot?size=800x450',
            'name' : 'Tearcoat Branch Upstream',
            'labels': get_image_label(vision_client,'http://b6b.hdrelay.com/cameras/403818bc-9e8d-4857-9eb0-a675c9668782/GetOneShot?size=800x450')
        },
        {
            'url' : 'http://b6b.hdrelay.com/cameras/009e9855-a077-4c2a-8289-10eea8ba9f0f/GetOneShot?size=800x450',
            'name' : 'Tearcoat Branch Downstream',
            'labels': get_image_label(vision_client,'http://b6b.hdrelay.com/cameras/009e9855-a077-4c2a-8289-10eea8ba9f0f/GetOneShot?size=800x450')
        },
        {
            'url' : 'http://b6b.hdrelay.com/cameras/39f23793-0e60-43f6-a0b4-ccb073004640/GetOneShot?size=800x450',
            'name' : 'Pocotaligo River Downstream',
            'labels': get_image_label(vision_client,'http://b6b.hdrelay.com/cameras/39f23793-0e60-43f6-a0b4-ccb073004640/GetOneShot?size=800x450')
        },
    ]

    context = {
        'img': img_analized,
    }
    return render(request, 'faisApp/flood_analyzer.html', context)

def floodAnalyzer1(request):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']='C:/Users/jpall/OneDrive/Documents/serviceaccountkey.json'
    client = vision.ImageAnnotatorClient()
    img_analized = [
        {
            'url' : 'http://b7b.hdrelay.com/cameras/fa96bb1e-426d-4b40-a820-251713325420/GetOneShot?size=800x450',
            'name' : 'Rocky Creek',
            'labels': get_image_label(client,'http://b7b.hdrelay.com/cameras/fa96bb1e-426d-4b40-a820-251713325420/GetOneShot?size=800x450')
        },
        {
            'url' : 'http://b7b.hdrelay.com/cameras/2fb4ae88-446d-4632-a849-d426240ccca5/GetOneShot?size=800x450',
            'name' : 'Rocky Branch',
            'labels': get_image_label(client,'http://b7b.hdrelay.com/cameras/2fb4ae88-446d-4632-a849-d426240ccca5/GetOneShot?size=800x450')
        },
        {
            'url' : 'http://b6b.hdrelay.com/cameras/42336a6b-5be8-443c-97c9-3d70da459e88/GetOneShot?size=800x450',
            'name' : 'Pee Dee River',
            'labels': get_image_label(client,'http://b7b.hdrelay.com/cameras/2fb4ae88-446d-4632-a849-d426240ccca5/GetOneShot?size=800x450')
        },
        {
            'url' : 'http://b6b.hdrelay.com/cameras/403818bc-9e8d-4857-9eb0-a675c9668782/GetOneShot?size=800x450',
            'name' : 'Tearcoat Branch Upstream',
            'labels': get_image_label(client,'http://b6b.hdrelay.com/cameras/403818bc-9e8d-4857-9eb0-a675c9668782/GetOneShot?size=800x450')
        },
        {
            'url' : 'http://b6b.hdrelay.com/cameras/009e9855-a077-4c2a-8289-10eea8ba9f0f/GetOneShot?size=800x450',
            'name' : 'Tearcoat Branch Downstream',
            'labels': get_image_label(client,'http://b6b.hdrelay.com/cameras/009e9855-a077-4c2a-8289-10eea8ba9f0f/GetOneShot?size=800x450')
        },
        {
            'url' : 'http://b6b.hdrelay.com/cameras/39f23793-0e60-43f6-a0b4-ccb073004640/GetOneShot?size=800x450',
            'name' : 'Pocotaligo River Downstream',
            'labels': get_image_label(client,'http://b6b.hdrelay.com/cameras/39f23793-0e60-43f6-a0b4-ccb073004640/GetOneShot?size=800x450')
        },
    ]

    context = {
        'img': img_analized,
    }
    return render(request, 'faisApp/flood_analyzer1.html', context)

def floodFreq(request):
    return render(request, 'faisApp/flood_frequency_analysis.html')


def cams511(request):
    return render(request, 'faisApp/cams511.html')


def twitterStreamer(request):
    qs = tweet_streamer.objects.all()    
    state_list = us_state.objects.all()
    context = {
        'twitter': "",
        'states_list': state_list,
    }
    return render(request, 'faisApp/twitter-streamer.html',context)


def updatetwitterstreamer(request, command):
    states = request.GET['value']
    client = pymongo.MongoClient("mongodb+srv://Admin:Testing321@cluster0-kwe35.gcp.mongodb.net/test?retryWrites=true&w=majority")
    db = client.twitter
    collection = db.tweets
    if states == 'All':
        regx = re.compile("", re.IGNORECASE)
    else: 
        regx = re.compile(states, re.IGNORECASE)
    myquery = {'geo': {'$regex': states}}
    array = []
    newarray_length = collection.count()
    db_length = tweet_streamer.objects.all().count()
    for row in collection.find({'geo': regx}):
        geo_location = row['geo']
        temp = tweet_streamer(text=row['text'], geo=geo_location, user=row['user'], date=row['date'], latitude=float(row['latitude']), longitude=float(row['longitude']))
        array.append(temp)
    tweet_streamer.objects.all()._raw_delete('default')
    if len(array) > 0:        
        tweet_streamer.objects.bulk_create(array)

    tweet_list = tweet_streamer.objects.all()
    

    context = {
        'twitter': tweet_list,
    }
    return  render(request, "faisApp/tweet-streamer-table.html",context )


def twitter(request):
    qs = twitter_data.objects.all()
    qs = ""
    context = {
        'twitter': qs,
    }
    return render(request, 'faisApp/twitter-data.html',context)


def user(request):
    context = {
        'posts': post_local.objects.all()
    }
    return render(request, 'faisApp/user.html', context)

class userListView(LoginRequiredMixin,ListView):
    model = post_local
    template_name = 'faisApp/user.html'
    context_object_name ='posts'
    ordering = ['-date_posted']


class userCreateView(LoginRequiredMixin, CreateView):
    model = post_local
    fields = ['latitude', 'longitude', 'name', 'depth', 'state', ]

    def form_valid(self, form):
        form.instance.author =  self.request.user
        return super().form_valid(form)

class userUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = post_local
    fields = ['latitude', 'longitude', 'name', 'depth', 'state', ]

    def form_valid(self, form):
        form.instance.author =  self.request.user
        return super().form_valid(form)

    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False

class userDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = post_local
    success_url = "/fais/user"
    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False

def earth(request):
    qs = realtime_usgs_data.objects.all()
    try:
        data_url = "https://waterwatch.usgs.gov/webservices/realtime?region=us&format=csv"
        usgs_csv = pd.read_csv(data_url)
        model = pd.DataFrame(usgs_csv)
        model = model[model.flow != 0]
        df = read_frame(qs)
        df.drop(df.index, inplace=True)
        df, model = [d.reset_index(drop=True) for d in (df, model)]
        df['station'] = model['station_nm']
        df['id_num'] = model['site_no']
        df['latitude'] = model['dec_lat_va']
        df['longitude'] = model['dec_long_va']
        df['stage'] = model['stage']
        df['flow'] = model['flow']
        df['avg_ht'] = model ['class']
        df['url'] = model['url']
        realtime_usgs_data.objects.all()._raw_delete('default')
        instances = []
        for index, row in df.iterrows():
            if row is not None:
                try:
                    flow_value = float((row['flow']))
                except ValueError:
                    flow_value = 0.0
                try:
                    stage_value = float(row['stage'])
                except ValueError:
                    stage_value = 0.0
                temp = realtime_usgs_data(station=row['station'],id_num=row['id_num'],latitude=float(row['latitude']),longitude=float(row['longitude']),stage=stage_value,flow=flow_value,avg_ht=row['avg_ht'],url=(row['url']))
                instances.append(temp)
        realtime_usgs_data.objects.bulk_create(instances)
    except:
        return HttpResponse("<script>alert('There is a problem with a USGS Server, please contact the administation for more information')</script>")

    realtime_usgs_data.objects.filter(flow=0.0).delete()
    realtime_usgs_data.objects.get(id_num=2487500).delete()
    realtime_usgs_data.objects.filter(avg_ht=0).delete()
    realtime_usgs_data.objects.filter(avg_ht=1).delete()
    realtime_usgs_data.objects.filter(avg_ht=2).delete()
    realtime_usgs_data.objects.filter(avg_ht=3).delete()
    realtime_usgs_data.objects.filter(avg_ht=4).delete()
    realtime_usgs_data.objects.filter(avg_ht=5).delete()
    realtime_usgs_data.objects.filter(avg_ht=6).delete()
    realtime_usgs_data.objects.filter(avg_ht=7).delete()
    all_stations = realtime_usgs_data.objects.all()
    state_list = us_state.objects.all()
    context = {
        'Stations': all_stations,
        'states_list': state_list,
    }
    return render(request, "faisApp/earth.html", context)

def data(request):
    return render(request, 'faisApp/data.html')

def userdata(request):
    print("form is submitted")
    first_name= request.POST['first_name']
    last_name= request.POST['last_name']
    email= request.POST['email']
    location= request.POST['location']
    flood_depth= request.POST['flood_depth']
    latitude= request.POST['latitude']
    longitude = request.POST['longitude']
    time_stamp = request.POST['time_stamp']
    date = request.POST['date']
    additional_info= request.POST['additional_info']
    user_data = Data(first_name=first_name, last_name=last_name, email=email, additional_info=additional_info, location=location, flood_depth=flood_depth, date=date)
    user_data.save()
    template = get_template('faisApp/pdf_template.html')
    context = {
        'FirstName' : first_name,
        'LastName' : last_name,
        'Email' : email,
        'Location' : location,
        'FloodDepth' : flood_depth,
        'Latitude' : latitude,
        'Longitude' : longitude,
        'TimeStamp' : time_stamp,
        'Date' : date,
        'AdditionalInfo' : additional_info,
    }
    html = template.render(context)
    pdf = render_to_pdf('faisApp/pdf_template.html', context)
    if pdf:
        response = HttpResponse(pdf, content_type='application/pdf')
        filename = "User_%s.pdf" %("Details")
        content = "inline; filename='%s'" %(filename)
        download = request.GET.get("download")
        if download:
            content = "attachment; filename='%s'" %(filename)
        response['Content-Disposition'] = content
        return response
    return render(request, 'faisApp/data.html')

def upload(request):
    context={}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs= FileSystemStorage()
        name= fs.save(uploaded_file.name, uploaded_file)
        context['url']= fs.url(name)
    return render(request, "faisApp/data.html", context)

def is_valid(url):
    """
    Checks whether `url` is a valid URL.
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)
def get_all_images(url):
    """
    Returns all image URLs on a single `url`
    """
    soup = bs(requests.get(url).content, "html.parser")
    urls = []
    for img in tqdm(soup.find_all("img"), "Extracting images"):
        img_url = img.attrs.get("src")
        if not img_url:
            # if img does not contain src attribute, just skip
            continue
        if is_valid(img_url):
            urls.append(img_url)
    return urls
def download(url, pathname):
    """
    Downloads a file given an URL and puts it in the folder `pathname`
    """
    # if path doesn't exist, make that path dir
    if not os.path.isdir(pathname):
        os.makedirs(pathname)
    # download the body of response by chunk, not immediately
    response = requests.get(url, stream=True)
    # get the total file size
    file_size = int(response.headers.get("Content-Length", 0))
    # get the file name
    filename = os.path.join(pathname, url.split("=")[-1])
    # progress bar, changing the unit to bytes instead of iteration (default by tqdm)
    progress = tqdm(response.iter_content(1024), f"Downloading {filename}", total=file_size, unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, "wb") as f:
        for data in progress:
            # write data read to the file
            f.write(data)
            # update the progress bar manually
            progress.update(len(data))
    return filename
def main(url, path):
    # get all images
    imgs = get_all_images(url)
    zipObj = ZipFile('sample.zip', 'w')
    for img in imgs:
        # for each image, download it
        zipObj.write(download(img, path))
    zipObj.close()

def images511(request):
    main("https://floodanalytics.clemson.edu/fais/511-cam", "images")
    file_path = os.path.join(settings.BASE_DIR, 'sample.zip')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            return response
    return render(request, 'faisApp/cams511.html')

def imagesusgs(request):
    main("https://floodanalytics.clemson.edu/fais/realtime-flood-cam", "images")
    file_path = os.path.join(settings.BASE_DIR, 'sample.zip')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            return response
    return render(request, 'faisApp/realtime-cam.html')
