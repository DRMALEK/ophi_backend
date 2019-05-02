from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse


import os

from google.oauth2 import service_account
import googleapiclient.discovery
import subprocess
import json

@csrf_exempt
def create_key(request, user_id):
 """Creates a key for a service account."""

 user_service_account_email =["gina-user@gina-71253.iam.gserviceaccount.com", "gina-user2@gina-71253.iam.gserviceaccount.com"]
 
 user_key_files_path =   "/home/elifildes/token_getter/myapp/keys/"
 user_token_files_path = "/home/elifildes/token_getter/myapp/tokens/"

 #user_id = request.POST.get("user_id")
 print(user_id)
 user_key_file_path  =  user_key_files_path  +  user_id + ".json"
 user_token_file_path = user_token_files_path + user_id + ".json"


 #user_id = rquest.POST.get("user_id", "")
 subprocess.run("gcloud config set account elifildes@gmail.com", shell=True) #Switch to Owner account, inorder to generate the key
 
 try:
    #subprocess.check_output("gcloud iam service-accounts keys create {0} --iam-account {1}".format(user_key_file_path, user_service_account_email[0]),shell=True)
    subprocess.check_output(["gcloud", "iam", "service-accounts", "keys", "create", user_key_file_path, "--iam-account",user_service_account_email[0]])
 except subprocess.CalledProcessError:
    subprocess.check_output(["gcloud", "iam", "service-accounts", "keys", "create", user_key_file_path, "--iam-account", 
user_service_account_email[1]])

# subprocess.run("chmod +x {0}".format(user_key_file_path), shell=True)
 
 subprocess.run("gcloud auth activate-service-account --key-file {0}".format(user_key_file_path), shell=True)

 return HttpResponse("success")

@csrf_exempt
def get_token(request, user_id):

 #user_id = request.POST.get("user_id")
 print(user_id)
 if user_id == "":
     return HttpResponse("Please provide a valid user_id parameter")
 #try:
 # 	user_id = request.POST.get("user_id", "")
 #except:
 #       return "Error please provide user_id parameter"

 #if user_id == "":
 #        return "Please send a valid user_id"

 user_key_files_path = "/home/elifildes/token_getter/myapp/keys/"
 user_key_file_path  = user_key_files_path + user_id + ".json"

 subprocess.run("gcloud auth activate-service-account --key-file {0}".format(user_key_file_path), shell=True)
 token = subprocess.check_output(["gcloud", "auth", "print-access-token"])

 return HttpResponse(token)
