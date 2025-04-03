from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import spacy
from concurrent.futures import ProcessPoolExecutor
import time

snp_500 = {
    # A
    "aapl": "AAPL",
    "apple": "AAPL",
    "abbv": "ABBV",
    "abbvie": "ABBV",
    "abt": "ABT",
    "abbott": "ABT",
    "acn": "ACN",
    "accenture": "ACN",
    "adbe": "ADBE",
    "adobe": "ADBE",
    "adi": "ADI",
    "analog devices": "ADI",
    "adp": "ADP",
    "automatic data processing": "ADP",
    "adsk": "ADSK",
    "autodesk": "ADSK",
    "aep": "AEP",
    "american electric power": "AEP",
    "aes": "AES",
    "the aes corporation": "AES",
    "aig": "AIG",
    "american international group": "AIG",
    "akam": "AKAM",
    "akamai": "AKAM",
    "all": "ALL",
    "allstate": "ALL",
    "amgn": "AMGN",
    "amgen": "AMGN",
    "amp": "AMP",
    "ameriprise": "AMP",
    "amzn": "AMZN",
    "amazon": "AMZN",
    "anf": "ANF",
    "abercrombie & fitch": "ANF",
    "anthm": "ANTM",
    "anthem": "ANTM",
    "aon": "AON",
    "aon plc": "AON",
    "apa": "APA",
    "apache": "APA",
    "apd": "APD",
    "air products and chemicals": "APD",
    "aph": "APH",
    "amphenol": "APH",
    "aptv": "APTV",
    "aptiv": "APTV",
    "are": "ARE",
    "alexandria real estate equities": "ARE",
    "ato": "ATO",
    "atmos energy": "ATO",
    "avb": "AVB",
    "avalonbay communities": "AVB",
    "avgo": "AVGO",
    "broadcom": "AVGO",
    "avt": "AVT",
    "avnet": "AVT",
    "axp": "AXP",
    "american express": "AXP",
    "ayx": "AYX",
    "alteryx": "AYX",

    # B
    "ba": "BA",
    "boeing": "BA",
    "bac": "BAC",
    "bank of america": "BAC",
    "bax": "BAX",
    "baxter international": "BAX",
    "bbby": "BBBY",
    "bed bath & beyond": "BBBY",
    "bbt": "BBT",
    "bb&t": "BBT",
    "bby": "BBY",
    "best buy": "BBY",
    "bdx": "BDX",
    "becton dickinson": "BDX",
    "ben": "BEN",
    "franklin resources": "BEN",
    "bf.b": "BF.B",
    "brown-forman": "BF.B",
    "bhf": "BHF",
    "brighthouse financial": "BHF",
    "biib": "BIIB",
    "biogen": "BIIB",
    "bk": "BK",
    "bank of new york mellon": "BK",
    "bkng": "BKNG",
    "booking holdings": "BKNG",
    "bkr": "BKR",
    "baker hughes": "BKR",
    "blk": "BLK",
    "blackrock": "BLK",
    "bmy": "BMY",
    "bristol-myers squibb": "BMY",
    "br": "BR",
    "broadridge financial solutions": "BR",
    "brk.b": "BRK.B",
    "berkshire hathaway": "BRK.B",
    "bsx": "BSX",
    "boston scientific": "BSX",
    "bwa": "BWA",
    "borgwarner": "BWA",
    "bxp": "BXP",
    "boston properties": "BXP",

    # C
    "c": "C",
    "citigroup": "C",
    "cag": "CAG",
    "conagra brands": "CAG",
    "cah": "CAH",
    "cardinal health": "CAH",
    "cap": "CAP",
    "capri holdings": "CAP",
    "cat": "CAT",
    "caterpillar": "CAT",
    "cb": "CB",
    "chubb": "CB",
    "cboe": "CBOE",
    "cboe global markets": "CBOE",
    "cbre": "CBRE",
    "cbre group": "CBRE",
    "cc": "CC",
    "chemours": "CC",
    "cci": "CCI",
    "crown castle": "CCI",
    "cck": "CCK",
    "crown holdings": "CCK",
    "ccl": "CCL",
    "carnival": "CCL",
    "cdns": "CDNS",
    "cadence design systems": "CDNS",
    "cdw": "CDW",
    "ce": "CE",
    "celanese": "CE",
    "cern": "CERN",
    "cerner": "CERN",
    "cf": "CF",
    "cf industries": "CF",
    "cfg": "CFG",
    "citizens financial group": "CFG",
    "chd": "CHD",
    "church & dwight": "CHD",
    "chk": "CHK",
    "chesapeake energy": "CHK",
    "chkp": "CHKP",
    "check point software": "CHKP",
    "chs": "CHS",
    "chico's fas": "CHS",
    "ci": "CI",
    "cigna": "CI",
    "cma": "CMA",
    "comerica": "CMA",
    "cmcsa": "CMCSA",
    "comcast": "CMCSA",
    "cme": "CME",
    "cme group": "CME",
    "cmg": "CMG",
    "chipotle mexican grill": "CMG",
    "cmi": "CMI",
    "cummins": "CMI",
    "cms": "CMS",
    "cms energy": "CMS",
    "cna": "CNA",
    "cna financial": "CNA",
    "cno": "CNO",
    "cno financial group": "CNO",
    "cnp": "CNP",

    # D
    "dal": "DAL",
    "delta air lines": "DAL",
    "dd": "DD",
    "dupont": "DD",
    "dhr": "DHR",
    "danaher": "DHR",
    "dov": "DOV",
    "dover": "DOV",
    "dltr": "DLTR",
    "dollar tree": "DLTR",
    "dg": "DG",
    "dollar general": "DG",
    "d": "D",
    "dominion": "D",
    "duk": "DUK",
    "duke energy": "DUK",
    "dri": "DRI",
    "darden": "DRI",

    # E
    "ecl": "ECL",
    "ecolab": "ECL",
    "ed": "ED",
    "consolidated edison": "ED",
    "eix": "EIX",
    "edison international": "EIX",
    "el": "EL",
    "estee lauder": "EL",
    "emn": "EMN",
    "eastman": "EMN",
    "etn": "ETN",
    "eaton": "ETN",
    "etr": "ETR",
    "entergy": "ETR",
    "exc": "EXC",
    "exelon": "EXC",
    "exr": "EXR",
    "extra space storage": "EXR",
    "eog": "EOG",
    "eog resources": "EOG",
    "eqr": "EQR",
    "equity residential": "EQR",
    "ess": "ESS",
    "essex property": "ESS",

    # F
    "f": "F",
    "ford": "F",
    "fdx": "FDX",
    "fedex": "FDX",
    "ffiv": "FFIV",
    "f5 networks": "FFIV",

    # G
    "gpn": "GPN",
    "global payments": "GPN",
    "gs": "GS",
    "goldman sachs": "GS",
    "grmn": "GRMN",
    "garmin": "GRMN",
    "gww": "GWW",
    "grainger": "GWW",
    "gpc": "GPC",
    "genuine parts": "GPC",

    # H
    "hd": "HD",
    "home depot": "HD",
    "hon": "HON",
    "honeywell": "HON",
    "has": "HAS",
    "hasbro": "HAS",
    "hpe": "HPE",
    "hp enterprise": "HPE",
    "hpq": "HPQ",
    "hp inc": "HPQ",

    # I
    "ibm": "IBM",
    "ice": "ICE",
    "intercontinental exchange": "ICE",
    "ipg": "IPG",
    "interpublic": "IPG",
    "intc": "INTC",
    "intel": "INTC",

    # J
    "jpm": "JPM",
    "jpmorgan": "JPM",
    "jci": "JCI",
    "johnson controls": "JCI",

    # K
    "kmb": "KMB",
    "kimberly-clark": "KMB",
    "kr": "KR",
    "kroger": "KR",
    "key": "KEY",
    "keycorp": "KEY",
    "klac": "KLAC",
    "kla": "KLAC",
    "kmi": "KMI",
    "kinder morgan": "KMI",

    # L
    "lmt": "LMT",
    "lockheed martin": "LMT",
    "lly": "LLY",
    "eli lilly": "LLY",
    "low": "LOW",
    "lowe's": "LOW",

    # M
    "ma": "MA",
    "mastercard": "MA",
    "mdt": "MDT",
    "medtronic": "MDT",
    "mmm": "MMM",
    "three m": "MMM",
    "mchp": "MCHP",
    "microchip": "MCHP",
    "mrk": "MRK",
    "merck": "MRK",
    "met": "MET",
    "metlife": "MET",
    "mco": "MCO",
    "moody's": "MCO",
    "msci": "MSCI",
    "mar": "MAR",
    "marriott": "MAR",
    "mro": "MRO",

    # N
    "nke": "NKE",
    "nike": "NKE",
    "nem": "NEM",
    "newmont": "NEM",
    "ntrs": "NTRS",
    "northern trust": "NTRS",
    "nvda": "NVDA",
    "nvidia": "NVDA",
    "noc": "NOC",
    "northrop": "NOC",
    "nrg": "NRG",
    "nrg energy": "NRG",

    # O
    "odfl": "ODFL",
    "old dominion": "ODFL",
    "omc": "OMC",
    "omnicom": "OMC",
    "orcl": "ORCL",
    "oracle": "ORCL",
    "otis": "OTIS",
    "otis worldwide": "OTIS",

    # P
    "payx": "PAYX",
    "paychex": "PAYX",
    "payc": "PAYC",
    "paycom": "PAYC",
    "pypl": "PYPL",
    "paypal": "PYPL",
    "pfe": "PFE",
    "pfizer": "PFE",
    "psx": "PSX",
    "phillips 66": "PSX",
    "pgr": "PGR",
    "progressive": "PGR",
    "pnc": "PNC",
    "pnc financial": "PNC",
    "pru": "PRU",
    "prudential": "PRU",
    "psa": "PSA",
    "public storage": "PSA",
    "pcar": "PCAR",
    "paccar": "PCAR",
    "pld": "PLD",
    "prologis": "PLD",

    # Q
    "qcom": "QCOM",
    "qualcomm": "QCOM",

    # R
    "reg": "REG",
    "regency centers": "REG",
    "rsg": "RSG",
    "republic services": "RSG",
    "rl": "RL",
    "ralph lauren": "RL",
    "rhi": "RHI",
    "robert half": "RHI",
    "rtx": "RTX",
    "raytheon": "RTX",

    # S
    "sbux": "SBUX",
    "starbucks": "SBUX",
    "slb": "SLB",
    "schlumberger": "SLB",
    "sre": "SRE",
    "sempra": "SRE",
    "swks": "SWKS",
    "skyworks": "SWKS",
    "syk": "SYK",
    "stryker": "SYK",
    "snps": "SNPS",
    "synopsys": "SNPS",
    "spgi": "SPGI",
    "s&p global": "SPGI",
    "spg": "SPG",
    "simon property": "SPG",
    "sna": "SNA",
    "snap-on": "SNA",
    "stt": "STT",
    "state street": "STT",

    # T
    "t": "T",
    "att": "T",
    "tmo": "TMO",
    "thermo fisher": "TMO",
    "tgt": "TGT",
    "target": "TGT",
    "txn": "TXN",
    "texas instruments": "TXN",

    # U
    "unh": "UNH",
    "unitedhealth": "UNH",
    "ups": "UPS",
    "uri": "URI",
    "united rentals": "URI",
    "usb": "USB",
    "us bancorp": "USB",

    # V
    "v": "V",
    "visa": "V",
    "vfc": "VFC",
    "vf": "VFC",
    "vrsn": "VRSN",
    "verisign": "VRSN",
    "vtr": "VTR",
    "ventas": "VTR",
    "vz": "VZ",
    "verizon": "VZ",
    "vsat": "VSAT",
    "viasat": "VSAT",

    # W
    "wba": "WBA",
    "walgreens": "WBA",
    "wfc": "WFC",
    "wells fargo": "WFC",
    "wynn": "WYNN",
    "wynn resorts": "WYNN",

    # X
    "xrx": "XRX",
    "xerox": "XRX",

    # Z
    "zbh": "ZBH",
    "zimmer biomet": "ZBH",
    "zts": "ZTS",
    "zoetis": "ZTS"
}

# MongoDB URI - Replace <password> with your actual password
uri = "mongodb+srv://am3567:CwfUpOrjtGK1dtnt@main.guajv.mongodb.net/?retryWrites=true&w=majority&appName=Main"
client = MongoClient(uri)
db = client["stocks_db"]
collection = db["mi_data"]

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Load the pre-trained NER model
nlp = spacy.load("en_core_web_sm")

# Extract stock tickers (companies) from the article
def extract_tickers(article):
    doc = nlp(article)
    tickers = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return tickers

# Function to process articles in parallel
def process_articles(articles):
    results = []
    for article in articles:
        results.append(extract_tickers(article))
        #print(results[-1])
    return results

# Fetch articles from MongoDB collection (you can adjust the query as needed)
def fetch_articles_from_db():
    articles = []
    # Adjust this query if necessary based on how your data is stored
    cursor = collection.find({}, {"_id": 0, "body": 1})  # Assuming "article_text" is the field name for articles
    for document in cursor[:10]:
        articles.append(document["body"])
    return articles

from pymongo import UpdateOne

# Function to filter tickers based on the S&P 500 dictionary
def filter_tickers(tickers):
    # Map tickers to their corresponding S&P 500 stock symbols
    filtered_tickers = set([snp_500.get(ticker.lower()) for ticker in tickers if snp_500.get(ticker.lower())])
    
    # If the filtered list is empty, add 'SPY'
    if not filtered_tickers:
        filtered_tickers.add('SPY')
    
    return list(filtered_tickers)

# Update MongoDB documents with filtered tickers
def update_db_with_tickers(articles, tickers):
    # Prepare a list of update operations
    update_operations = []
    for article, article_tickers in zip(articles, tickers):
        filtered_tickers = filter_tickers(article_tickers)
        # Construct the update operation to add filtered_tickers and remove the body field
        update_operations.append(
            UpdateOne(
                {"body": article},  # Find the document by body content (or adjust as needed)
                {
                    "$set": {"tickers": filtered_tickers},  # Set the new 'tickers' field
                    "$unset": {"body": ""}  # Remove the 'body' field
                }
            )
        )

    # Perform all update operations in bulk
    if update_operations:
        collection.bulk_write(update_operations)
        print(f"Successfully updated {len(update_operations)} documents.")

# Process articles and update the database
def process_and_update_db():
    print("starting")
    articles = fetch_articles_from_db()  # Fetch articles from MongoDB
    for x in range(len(articles) // 1000 + 1):
        tickers = process_articles(articles[x * 1000: (x + 1) * 1000])  # Extract tickers from articles
        update_db_with_tickers(articles[x * 1000: (x + 1) * 1000], tickers)  # Update the documents with filtered tickers

# Run the process and update
process_and_update_db()