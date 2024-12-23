from django.urls import path
from . import views

urlpatterns = [
    path('register/',views.register, name='register'),
    path('login/',views.login_view, name='login'),
    path('dashboard/',views.dashboard, name='dashboard'),
    path('meeting/',views.videocall, name='meeting'),
    path('logout/',views.logout_view, name='logout'),
    path('join/',views.join_room, name='join_room'),
    path('',views.index, name='index'),
    path('start_drawing/', views.start_drawing, name='start_drawing'),
    #path('linked_in/', views.linked_in, name='linked_in'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('get_response/', views.get_response, name='get_response'),
    #path('stop_drawing/', views.stop_drawing, name='stop_drawing'),

]