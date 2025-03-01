import requests
from bs4 import BeautifulSoup
import os
from pymongo import MongoClient

snp_500_stocks = ["AAPL","NVDA","MSFT","AMZN","GOOGL","GOOG","META","BRK-B","TSLA","AVGO","LLY","WMT","JPM","V","MA","XOM","COST","ORCL","UNH","NFLX","PG","JNJ","HD","ABBV","BAC","TMUS","KO","CRM","CVX","WFC","CSCO","PM","ABT","MRK","IBM","GE","LIN","MCD","ACN","MS","AXP","PEP","DIS","ISRG","GS","TMO","PLTR","T","BX","NOW","ADBE","VZ","TXN","RTX","QCOM","INTU","AMGN","PGR","BKNG","SPGI","CAT","AMD","UBER","BSX","BLK","C","PFE","UNP","DHR","SYK","SCHW","NEE","GILD","TJX","LOW","HON","CMCSA","FI","SBUX","BA","DE","AMAT","ADP","PANW","COP","VRTX","BMY","KKR","MDT","NKE","ANET","PLD","MMC","ETN","ADI","CB","LMT","MU","INTC","UPS","ICE","WELL","SO","LRCX","CRWD","AMT","KLAC","MO","WM","GEV","CME","DUK","SHW","ELV","MCO","AON","EQIX","ABNB","AJG","PH","CI","APO","MMM","CTAS","FTNT","MDLZ","CVS","MCK","APH","TT","ORLY","MAR","ITW","CEG","TDG","COF","ECL","PNC","HCA","ZTS","REGN","RSG","CL","MSI","USB","CMG","DELL","WMB","EOG","SPG","APD","WDAY","PYPL","SNPS","EMR","CDNS","GD","NOC","RCL","BDX","BK","FDX","HLT","ROP","CSX","TFC","KMI","OKE","AFL","ADSK","MET","TRV","AZO","CHTR","TGT","SLB","PCAR","JCI","AEP","CARR","NSC","HWM","NXPI","PAYX","DLR","PSA","MNST","FCX","ALL","PSX","CPRT","AMP","O","CMI","AIG","DFS","GWW","COR","GM","D","NEM","NDAQ","MPC","KMB","KR","SRE","ROST","FANG","MSCI","TEL","FICO","HES","OXY","VST","KDP","KVUE","EXC","LULU","BKR","GRMN","TRGP","AME","FAST","YUM","CTVA","GLW","EW","CBRE","URI","VRSK","XEL","CTSH","VLO","CCI","PRU","PEG","AXON","DHI","GEHC","OTIS","LHX","DAL","PWR","IT","F","ETR","ODFL","FIS","TTWO","SYY","KHC","A","PCG","IDXX","ED","HSY","ACGL","DXCM","VICI","IR","DD","RMD","BRO","WTW","HIG","WEC","EA","GIS","EXR","IQV","LYV","TPL","HUM","VMC","ROK","AVB","LVS","NUE","STZ","LEN","MCHP","XYL","WAB","RJF","MTB","CAH","CCL","CSGP","EBAY","UAL","VTR","EFX","IP","MLM","TSCO","MPWR","ANSS","HPQ","FITB","EQR","EQT","CNC","STT","K","BR","WBD","DTE","KEYS","AEE","CHD","IRM","FTV","DOV","SW","DOW","MTD","AWK","FOXA","TYL","GPN","HPE","PPL","EL","PPG","ROL","CPAY","GDDY","EXPE","LYB","FOX","VLTO","SMCI","ATO","TDY","HBAN","WRB","CDW","TROW","DRI","DVN","SYF","SBAC","CINF","ES","HAL","ADM","VRSN","CNP","ERIE","FE","WAT","MKC","CBOE","TSN","WY","CMS","NVR","NTRS","STX","STE","RF","LII","EIX","LH","IFF","DECK","INVH","PHM","ESS","NRG","ZBH","CTRA","STLD","BIIB","NTAP","MAA","PFG","CFG","HUBB","ON","PTC","CLX","BBY","NI","KEY","PODD","DGX","PKG","L","LUV","COO","TER","SNA","ARE","TRMB","FDS","TPR","BAX","GPC","WDC","ULTA","JBL","LDOS","WST","FFIV","DPZ","GEN","RL","MOH","NWS","LNT","UDR","DG","NWSA","OMC","EXPD","JBHT","ZBRA","MAS","EVRG","BLDR","J","HRL","DLTR","BF-B","PNR","EG","KIM","BALL","APTV","AVY","IEX","FSLR","AMCR","DOC","INCY","HOLX","REG","ALGN","CF","TXT","SOLV","CPT","RVTY","SWK","POOL","KMX","JKHY","BXP","TAP","PAYC","CAG","AKAM","CHRW","JNPR","MRNA","NDSN","CPB","DVA","SJM","EPAM","UHS","HST","EMN","ALLE","VTRS","PNW","LKQ","GL","SWKS","BEN","AIZ","IPG","NCLH","BG","MGM","DAY","TECH","AOS","WYNN","WBA","FRT","HAS","ALB","HSIC","CRL","AES","GNRC","MTCH","IVZ","MOS","APA","ENPH","PARA","MHK","LW","MKTX","CZR","HII","BWA","TFX","CE","FMC"]
uri = "mongodb+srv://am3567:CwfUpOrjtGK1dtnt@main.guajv.mongodb.net/?retryWrites=true&w=majority&appName=Main"
client = MongoClient(uri)
db = client["stocks_db"]
collection = db["stock_data"]

for stock in snp_500_stocks:
    print(f"starting {stock}")
    url = f"https://finance.yahoo.com/quote/{stock}/news/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    for article in soup.find_all("div", class_="content"):
        links.append(article.a["href"])

    for link in links[:5]:
        print("adding to db")
        response = requests.get(link, headers = headers)
        soup = BeautifulSoup(response.text, "html.parser")

        #print(stock)

        header = soup.find("div", class_="cover-title").text
        #print(header.text)

        time = soup.find("time", class_="byline-attr-meta-time")["datetime"]
        #print(time["datetime"])

        body = soup.find("div", class_="body-wrap").text
        #print(body.text)

        article = {
            "stock": stock,
            "title": header,
            "time": time,
            "body": body
        }

        collection.insert_one(article)