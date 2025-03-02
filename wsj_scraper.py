import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
import datetime as dt
import re
import undetected_chromedriver as uc
from selenium.webdriver.chrome.options import Options
from time import sleep

options = Options()
options.add_argument("--disable-gpu")
options.add_argument("start-maximized")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--remote-debugging-port=9222")

options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
options.add_argument("--disable-blink-features=AutomationControlled")

uri = "mongodb+srv://am3567:CwfUpOrjtGK1dtnt@main.guajv.mongodb.net/?retryWrites=true&w=majority&appName=Main"
client = MongoClient(uri, )
db = client["stocks_db"]
collection = db["wsj_data"]

starting_date = dt.datetime.today()
date = starting_date

headers = {"User-Agent": "Mozilla/5.0"}

for x in range(1):
    date_string = date.strftime("%Y/%m/%d")
    print(date_string)
    wsj_url = f"https://www.wsj.com/news/archive/{date_string}"

    response = requests.get(wsj_url, headers = headers)
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("article", class_=re.compile("story"))

    for article in articles:
        if article.find("div", class_=re.compile("articleType")).text in "Stock Market, Economy, Business, U.S. Economy, Tech, Commodities, Finance, Markets, Central Banks, Economic Data, Banks, Banking, Commercial Real Estate, Autos Industry, Stocks, Business World, Risk & Compliance Journal, Europe Economy":
            link = article.find("div", class_=re.compile("headline")).a["href"]

            driver = uc.Chrome(options=options)
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            driver.get(link)
            sleep(30)

            link_response = requests.get(link, headers = headers)
            link_soup = BeautifulSoup(link_response.text, "html.parser")

            print(link_soup)

            headline = link_soup.find("h1", re.compile("StyledHeadline")).text
            time = link_soup.find("time")["datetime"]
            body = link_soup.find_all("p", class_=re.compile("Paragraph"))
            body = [para.text for para in body]
            body = "".join(body)

            db_article = {
                "title": headline,
                "time": time,
                "body": body
            }

            print(db_article)

    date = date - dt.timedelta(days=1)