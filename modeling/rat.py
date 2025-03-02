import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import pandas as pd
import yfinance as yf

uri = "mongodb+srv://am3567:CwfUpOrjtGK1dtnt@main.guajv.mongodb.net/?retryWrites=true&w=majority&appName=Main"
client = MongoClient(uri)
db = client["stocks_db"]
collection = db["mi_data"]

#Retrieval Augmented Transformer
class RAT(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, output_dim):
        super(RAT, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_layers
        )

        self.attention = nn.Linear(embed_dim, 1)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        
        # Compute attention weights
        weights = F.softmax(self.attention(x).squeeze(-1), dim=1)
        # Apply weights to get weighted sum
        x = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        x = self.fc(x)
        return x

    def train_model(self, epochs, batch_size, X, y):
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                output = self(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                    
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")

    def format_data_seperate(self, stock_data, financial_data, input_length, output_length):
        stock_data = torch.tensor(stock_data, dtype=torch.float32)
        financial_data = torch.tensor(financial_data, dtype=torch.float32)

        combined_data = torch.cat([stock_data, financial_data], dim=1)
        
        num_samples = len(combined_data) - input_length - output_length + 1
        X = torch.zeros(num_samples, input_length, self.input_dim)
        y = torch.zeros(num_samples, output_length, self.output_dim)
        
        for i in range(num_samples):
            X[i] = combined_data[i:i+input_length]

            target_values = stock_data[i+input_length:i+input_length+output_length, :self.output_dim]
            y[i] = target_values

        return X, y
    
    def format_data_combined(self, combined_data, input_length, output_length, stock_data):
        combined_data = torch.tensor(combined_data, dtype=torch.float32)
        
        num_samples = len(combined_data) - input_length - output_length + 1
        X = torch.zeros(num_samples, input_length, self.input_dim)
        y = torch.zeros(num_samples, output_length, self.output_dim)
        
        for i in range(num_samples):
            X[i] = combined_data[i:i+input_length]

            target_values = stock_data[i+input_length:i+input_length+output_length, :self.output_dim]
            y[i] = target_values

        return X, y

    def query_articles(self, X, I, top_k = 5):
        embedding_model = SentenceTransformer("ProsusAI/finbert")
        query_embedding = embedding_model.encode(f"Stock News About {I["stock"]}").tolist()

        pipeline = [
            {
                "$addFields": {
                    "dot_product": {
                        "$sum": {
                            "$map": {
                                "input": {
                                    "$zip": {
                                        "inputs": ["$embedding", query_embedding]
                                    }
                                },
                                "as": "pair",
                                "in": {
                                    "$multiply": [
                                        {"$arrayElemAt": ["$$pair", 0]},
                                        {"$arrayElemAt": ["$$pair", 1]}
                                    ]
                                }
                            }
                        }
                    },
                    "query_norm": {
                        "$sqrt": {
                            "$max": [
                                0.000001,  # Small epsilon to avoid zero
                                {
                                    "$sum": {
                                        "$map": {
                                            "input": "$embedding",
                                            "as": "val",
                                            "in": { "$multiply": ["$$val", "$$val"] }
                                        }
                                    }
                                }
                            ]
                        }
                    },
                    "embedding_norm": {
                        "$sqrt": {
                            "$max": [
                                0.000001,  # Small epsilon to avoid zero
                                {
                                    "$sum": {
                                        "$map": {
                                            "input": query_embedding,
                                            "as": "val",
                                            "in": { "$multiply": ["$$val", "$$val"] }
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }
            },
            {
                "$addFields": {
                    "cosine_similarity": {
                        "$cond": {
                            "if": {
                                "$or": [
                                    {"$eq": ["$query_norm", 0]},
                                    {"$eq": ["$embedding_norm", 0]}
                                ]
                            },
                            "then": 0,  # Default to 0 similarity if either norm is zero
                            "else": {
                                "$divide": ["$dot_product", {"$multiply": ["$query_norm", "$embedding_norm"]}]
                            }
                        }
                    }
                }
            },
            {
                "$sort": {"cosine_similarity": -1}
            },
            {
                "$limit": top_k  # Top 5 results
            }
        ]

        results = list(collection.aggregate(pipeline))
        return results
        
    def get_stock_micro_data(self, list_stocks) -> pd.DataFrame:
        """Given the stocks to query, return a dataframe of combined historical and financial data
        
        Args:
            list_stocks (list[str]): list of stocks to get data for
            
        Returns:
            pd.Dataframe: their data
            
        """

        dict_income_statement_catcher = {
            "Net Income": "Income",
            "Total Income": "Income",
            "Net Revenue": "Revenue",
            "Total Revenue": "Revenue"
        }

        list_income_statement_metrics = [
            "Income", 
            "EBITDA", 
            "Revenue"
        ]

        dict_balance_sheet_catcher = {
            "Net Assets": "Assets",
            "Total Assets": "Assets",
            "Net Debt": "Debt",
            "Total Debt": "Debt",
            "Stockholders Equity": "SH_Equity",
            "Stockholders' Equity": "SH_Equity"
        }

        list_balance_sheet_metrics = [
            "Assets", 
            "Debt", 
            "SH_Equity"
        ]

        dict_cash_flow_catcher = {
            "Free Cash Flow": "FCF",
            "Operating Cash Flow": "OCF",
            "Capital Expenditure": "CAPEX"
        }

        list_cash_flow_metrics = [
            "FCF",
            "OCF",
            "CAPEX"
        ]

        list_stock_data = []

        for str_stock in list_stocks:
            ticker = yf.Ticker(str_stock)

            series_history = ticker.history(period = "max")["Close", "Volume"]
            

            df_income_statement = ticker.financials.rename(index = dict_income_statement_catcher).loc[list_income_statement_metrics]
            df_balance_sheet = ticker.balance_sheet.rename(index = dict_balance_sheet_catcher).loc[list_balance_sheet_metrics]
            df_cash_flow = ticker.cashflow.rename(index = dict_cash_flow_catcher).loc[list_cash_flow_metrics]

            df_one_stock_data = pd.concat([df_income_statement, df_balance_sheet, df_cash_flow])
            df_one_stock_data = df_one_stock_data.T.dropna().resample("D").ffill()

            df_one_stock_data.index = pd.to_datetime(df_one_stock_data.index).tz_localize(None)
            series_history.index = pd.to_datetime(series_history.index).tz_localize(None)

            list_stock_data.append(pd.concat([df_one_stock_data, series_history], axis = 1))

        return pd.concat(list_stock_data, axis = 1, keys = list_stocks).dropna()