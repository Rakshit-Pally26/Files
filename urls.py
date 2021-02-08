from django.urls import path, re_path
from . import views
from .views import userListView, userCreateView, userUpdateView, userDeleteView
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('fais', views.home, name='fais-home'),
    path('fais/earth-engine', views.earth, name='fais-earth'), 
    path('fais/historical-usgs', views.historical, name='fais-hist'),
    path('fais/realtime-usgs', views.realtime, name='fais-realtime'),
    path('fais/realtime-flood-cam', views.floodcam, name='fais-cam'),
    path('fais/flood-analyzer', views.floodAnalyzer, name='fais-analyzer'),
    path('fais/flood-analyzer1', views.floodAnalyzer1, name='fais-analyzer1'),
    path('fais/flood-freq', views.floodFreq, name='fais-freq'),
    path('fais/twitter', views.twitter, name='fais-twitter'),
    path('fais/twitter-streamer', views.twitterStreamer, name='fais-twitterStreamer'),
    path('fais/511-cam', views.cams511, name='fais-cams511'),
    path('fais/data', views.data, name='data'),
    path('fais/upload', views.upload, name='upload'),
    path('fais/userdata', views.userdata, name='userdata'),
    path('fais/images511', views.images511, name='images511'),
    path('fais/imagesusgs', views.imagesusgs, name='imagesusgs'),
    path('fais/hs-usgs', views.HydroShare, name='fais-hs'),
    #path('fais/user', userListView.as_view() , name='fais-user'),
    #path('fais/user/new', userCreateView.as_view() , name='fais-create'),    
    #path('fais/user/<int:pk>/update', userUpdateView.as_view() , name='fais-update'),
    #path('fais/user/<int:pk>/delete', userDeleteView.as_view() , name='fais-delete'),
    re_path(r'^execute/(?P<command>[a-z]+)',views.execute, name='execute'),
    re_path(r'^executetweet/(?P<command>[a-z]+)',views.executetweet, name='executetweet'),
    re_path(r'^executehydroshare/(?P<command>[a-z]+)',views.executehydroshare, name='executehydroshare'),
    re_path(r'^executeh/(?P<command>[a-z]+)',views.executehistorical, name='executehistorical'),
    re_path(r'^stateselecthist/(?P<command>[a-z]+)',views.stateselecthist, name='stateselecthist'),
    re_path(r'^updatestreamer/(?P<command>[a-z]+)',views.updatetwitterstreamer, name='updatestreamer'),
    re_path(r'^uploadAnalizer',views.uploadAnalizer, name='uploadAnalizer'),
    re_path(r'^uploadAnalizer1',views.uploadAnalizer1, name='uploadAnalizer1'),
    re_path(r'^floodFrequencyAnalysis/(?P<command>[a-z]+)',views.floodFrequencyAnalysis, name='floodFrequencyAnalysis'),
    re_path(r'^saveCSVRealTime',views.saveCSVRealTime, name='saveCSVRealTime'),
    re_path(r'^saveCSVHist',views.saveCSVHist, name='saveCSVHist'),    
    re_path(r'^saveCSVTwitter',views.saveCSVTwitter, name='saveCSVTwitter'),
    re_path(r'^saveCSVTwitterStream',views.saveCSVTwitterStream, name='saveCSVTwitterStream'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)