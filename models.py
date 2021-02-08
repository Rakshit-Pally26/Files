from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

from django.urls import reverse
from django_pandas.managers import DataFrameManager

    
class realtime_usgs_data(models.Model):
    station = models.TextField()
    id_num = models.CharField(max_length=100)
    latitude = models.FloatField()
    longitude = models.FloatField()
    flow = models.FloatField()
    stage = models.FloatField()
    avg_ht = models.IntegerField()
    url = models.URLField()
    objects = DataFrameManager()

    def __str__(self):
        return self.id_num


class post_local(models.Model):
    latitude = models.FloatField()
    longitude = models.FloatField()
    name = models.CharField(max_length=100)
    depth = models.FloatField()
    state = models.CharField(max_length=2)
    date_posted = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.name

    
    def get_absolute_url(self):
        return reverse('fais-user')

class us_state(models.Model):
    state_code = models.CharField(max_length=2)
    state_name = models.CharField(max_length=100)

    def __str__(self):
        return self.state_name

class select_state(models.Model):
    state_code = models.CharField(max_length=2)
    state_name = models.CharField(max_length=100)

    def __str__(self):
        return self.state_name
class twitter_data(models.Model):
    tweets = models.TextField()
    date_posted = models.CharField(max_length=100)
    source_url = models.URLField()
    image_url = models.URLField()
    Sentiment = models.IntegerField()

class historical_usgs_data(models.Model):
    date = models.CharField(max_length=100)
    discharge = models.FloatField(blank=True)
    gage_height = models.FloatField(blank=True)

class flood_station_list(models.Model):
    station_number = models.CharField(max_length=100)
    station_name = models.CharField(max_length=100)

class tweet_streamer(models.Model):
    text = models.CharField(max_length=100)
    geo = models.CharField(max_length=100)
    user = models.CharField(max_length=100)
    date = models.CharField(max_length=100)
    latitude = models.FloatField()
    longitude = models.FloatField()

class peak_rate(models.Model):
    site_no = models.CharField(max_length=100)
    peak_dt = models.CharField(max_length=100)
    peak_va = models.CharField(max_length=100)
    gage_ht = models.CharField(max_length=100)

class Data(models.Model):
    first_name= models.CharField(max_length=100, null=True)
    last_name= models.CharField(max_length=100, null=True)
    email= models.EmailField(null=True)
    location = models.CharField(max_length=100, null=True)
    flood_depth = models.IntegerField(null=True)
    latitude= models.CharField(max_length=12, null=True)
    longitude= models.CharField(max_length=13, null=True)
    time_stamp= models.TimeField(null=True)
    date= models.DateField(null=True)
    additional_info= models.TextField(null=True)

