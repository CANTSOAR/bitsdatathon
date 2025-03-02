import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
import datetime as dt
import re
import time

uri = "mongodb+srv://am3567:CwfUpOrjtGK1dtnt@main.guajv.mongodb.net/?retryWrites=true&w=majority&appName=Main"
client = MongoClient(uri, tlsAllowInvalidCertificates=True)
db = client["stocks_db"]
collection = db["mi_data"]

starting_date = dt.datetime(year = 2024, month = 6, day = 19)
date = starting_date

headers = {"User-Agent": "Mozilla/5.0"}

for x in range(4 * 365 + 200):
    date_string = date.strftime("%Y/%m/%d")
    print(date_string)
    mi_url = f"https://markets.businessinsider.com/news/archive/{date_string}"

    retries = 5
    while retries:
        try:
            response = requests.get(mi_url, headers = headers)
            soup = BeautifulSoup(response.text, "html.parser")
            container = soup.find("div", class_="box")
            articles = container.find_all("a")
            break
        except:
            print(f"retrying... {retries - 1} retries left")
            time.sleep(1)
            retries -= 1

    db_articles = []
    for article in articles:
        print(".", end="")
        try:
            link = article["href"]

            if "seekingalpha" in link:
                continue
            
            if link[0] == "/":
                link_response = requests.get("https://markets.businessinsider.com" + link, headers = headers)

                link_soup = BeautifulSoup(link_response.text, "html.parser")

                headline = link_soup.find("h1").text
                time = link_soup.find("span", class_="news-post-quotetime").text
                body = link_soup.find("div", class_="news-content")
                body = [para.text for para in body.find_all("p")]
                body = "".join(body)

                db_article = {
                    "title": headline,
                    "time": time,
                    "body": body
                }
            else:
                link_response = requests.get(link, headers = headers)

                link_soup = BeautifulSoup(link_response.text, "html.parser")

                headline = link_soup.find("h1").text
                time = link_soup.find("time").text.strip()
                body = link_soup.find("div", class_="content-lock-content")
                body = [para.text for para in body.find_all("p")]
                body = "".join(body)

                db_article = {
                    "title": headline,
                    "time": time,
                    "body": body
                }

            db_articles.append(db_article)
        except:
            continue

    collection.insert_many(db_articles)
    print(f"{len(db_articles)} loaded")

    date = date - dt.timedelta(days=1)