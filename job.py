#!/usr/bin/env python3
import requests
import datetime

def ping_backend():
    url = "https://veille-demo.onrender.com/watch" #api
    
    try:
        print(f"{datetime.datetime.now()}: Pinging {url}")
        response = requests.post(url, timeout=30)
        print(f"Response: {response.status_code}")
        
        if response.status_code == 200:
            print("Job launched")
        else:
            print(f"⚠️ Unexpected status code: {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    ping_backend()  # Just ping once, then exit
