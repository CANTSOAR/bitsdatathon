import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

uri = "mongodb+srv://am3567:CwfUpOrjtGK1dtnt@main.guajv.mongodb.net/?retryWrites=true&w=majority&appName=Main"
client = MongoClient(uri)
db = client["stocks_db"]
collection = db["mi_data"]

embedding_model = SentenceTransformer("ProsusAI/finbert")

#Retrieval Augmented Transformer
class RAT(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        super(RAT, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.article_projection = nn.Linear(768, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_layers
        )


        self.decoder = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, output_dim)

        """cursor = collection.find({}, {"embedding": 1})  # Project only the embedding field

        embeddings = []
        counter = 0
        count = collection.count_documents({})
        for doc in cursor:
            if not counter % 1000: print(f"loaded {counter}/{count}")
            embeddings.append(doc['embedding'])
            counter += 1

        # Convert to numpy array and then to a PyTorch tensor
        self.embed_db = np.array(embeddings)"""

    def forward(self, x):
        # Encoder
        x1 = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        #article_embeds = self.query_fast("", "", self.stock)
        #x2 = self.article_projection(article_embeds)
        #x2 = x2.unsqueeze(0).expand(x1.shape[0], -1, -1)
        x2 = self.saved_x2[:x1.shape[0]]

        x = torch.cat([x1, x2], dim=1)
        encoded = self.transformer(x)  # [batch_size, seq_len, embed_dim]
        
        # Get context vector (last hidden state)
        context = encoded[:, -1].unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # Initialize decoder input
        decoder_input = context.repeat(1, self.output_length, 1)  # [batch_size, output_seq_len, embed_dim]
        
        # Decoder
        outputs, _ = self.decoder(decoder_input)  # [batch_size, output_seq_len, embed_dim]
        predictions = self.fc(outputs)  # [batch_size, output_seq_len, output_dim]
        
        return predictions

    def train_model(self, epochs, batch_size, X, y, patience=15, validation_split=0.1):
        """
        Train the RAT model with early stopping and faster convergence techniques
        
        Args:
            epochs: Maximum number of training epochs
            batch_size: Training batch size
            X: Input tensor of shape [samples, seq_len, features]
            y: Target tensor of shape [samples, output_seq_len, output_features]
            patience: Number of epochs to wait for improvement before early stopping
            validation_split: Fraction of data to use for validation
        """
        # Split data into train and validation sets
        val_size = int(len(X) * validation_split)
        train_X, val_X = X[val_size:], X[:val_size]
        train_y, val_y = y[val_size:], y[:val_size]
        
        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        self.query_articles("", "", self.stock_data)
        self.saved_x2 = self.saved_x2.unsqueeze(0).expand(batch_size, -1, -1)

        criterion = nn.MSELoss()
        optimizer = AdamW(self.parameters(), lr=0.005, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Early stopping variables
        best_val_loss = float('inf')
        wait = 0
        best_model = None
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0
            for X_batch, y_batch in train_dataloader:
                optimizer.zero_grad()
                output = self(X_batch)
                
                # Reshape if needed based on your model output
                if output.shape != y_batch.shape:
                    output = output.view(y_batch.shape)
                    
                loss = criterion(output, y_batch)
                loss.backward(retain_graph=True)
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                
            avg_train_loss = train_loss / len(train_dataloader)
            
            # Validation phase
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_dataloader:
                    output = self(X_batch)
                    
                    # Reshape if needed
                    if output.shape != y_batch.shape:
                        output = output.view(y_batch.shape)
                        
                    loss = criterion(output, y_batch)
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(val_dataloader)
            
            # Update learning rate based on validation loss
            scheduler.step(avg_val_loss)
            
            # Print progress
            if (epoch+1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                wait = 0
                # Save best model
                best_model = {k: v.cpu().detach().clone() for k, v in self.state_dict().items()}
            elif avg_val_loss > best_val_loss * 1.2:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    # Restore best model
                    self.load_state_dict(best_model)
                    return best_val_loss
        
        # Restore best model at the end of training
        if best_model is not None:
            self.load_state_dict(best_model)
            
        return best_val_loss

    def predict(self, new_data):
        # Scale new data using stored scalers
        """new_data_np = new_data.numpy() if isinstance(new_data, torch.Tensor) else np.array(new_data)
        scaled_new_data = np.zeros_like(new_data_np)
        
        # Assuming first columns are stock data and rest are financial
        stock_cols = min(len(self.stock_scalers), new_data_np.shape[1])
        for i in range(stock_cols):
            col_data = new_data_np[:, i].reshape(-1, 1)
            scaled_new_data[:, i] = self.stock_scalers[i].transform(col_data).flatten()
        
        for i in range(stock_cols, new_data_np.shape[1]):
            col_data = new_data_np[:, i].reshape(-1, 1)
            fin_idx = i - stock_cols
            if fin_idx < len(self.financial_scalers):
                scaled_new_data[:, i] = self.financial_scalers[fin_idx].transform(col_data).flatten()
        
        # Convert to tensor and make prediction
        tensor_data = torch.tensor(scaled_new_data, dtype=torch.float32)"""
        
        # Format for model input (assuming we need a sequence of input_length)
        if len(new_data) >= self.input_length:
            #model_input = new_data[-self.input_length:].unsqueeze(0)  # Add batch dimension
            model_input = new_data
            self.query_articles("", "", self.stock_data)
            self.saved_x2 = self.saved_x2.unsqueeze(0).expand(len(model_input), -1, -1)
            
            with torch.no_grad():
                output = self(model_input)  # [1, output_length, output_dim]
            
            # Inverse transform the predictions
            output_np = output.numpy()
            unscaled_output = np.zeros_like(output_np)
            
            for i in range(output_np.shape[-1]):
                if i < len(self.stock_scalers):
                    # Reshape for inverse_transform
                    col_data = output_np[0, :, i].reshape(-1, 1)
                    unscaled_output[0, :, i] = self.stock_scalers[i].inverse_transform(col_data).flatten()
            
            return torch.tensor(unscaled_output)
        else:
            raise ValueError(f"Input data must have at least {self.input_length} time steps")

    def format_data_separate(self, stock_data, financial_data, input_length, output_length):
        self.stock_data = stock_data

        self.stock_scalers = []
        self.financial_scalers = []
        
        # Scale each column of stock data independently
        scaled_stock = np.zeros_like(stock_data)
        for i in range(stock_data.shape[1]):
            scaler = StandardScaler()  # You can also use MinMaxScaler()
            # Reshape to 2D array for fit_transform
            col_data = stock_data[:, i].reshape(-1, 1)
            scaled_stock[:, i] = scaler.fit_transform(col_data).flatten()
            self.stock_scalers.append(scaler)
        
        # Scale each column of financial data independently
        scaled_financial = np.zeros_like(financial_data)
        for i in range(financial_data.shape[1]):
            scaler = StandardScaler()
            col_data = financial_data[:, i].reshape(-1, 1)
            scaled_financial[:, i] = scaler.fit_transform(col_data).flatten()
            self.financial_scalers.append(scaler)
        
        # Convert back to tensors
        stock_data = torch.tensor(scaled_stock, dtype=torch.float32)
        financial_data = torch.tensor(scaled_financial, dtype=torch.float32)

        combined_data = torch.cat([stock_data, financial_data], dim=1)
        
        num_samples = len(combined_data) - input_length - output_length + 1
        X = torch.zeros(num_samples, input_length, self.input_dim)
        y = torch.zeros(num_samples, output_length, self.output_dim)
        
        for i in range(num_samples):
            X[i] = combined_data[i:i+input_length]

            target_values = stock_data[i+input_length:i+input_length+output_length, :self.output_dim]
            y[i] = target_values

        self.input_length = input_length
        self.output_length = output_length

        return X, y
    
    def format_data_combined(self, combined_data, input_length, output_length):
        return self.format_data_separate(combined_data[:, :-2], combined_data[:, -2:], input_length, output_length)

    def query_fast(self, X, I, stock_name, top_k = 5):
        query_embedding = embedding_model.encode(f"Stock News About {stock_name}")

        cos_sim = cosine_similarity(query_embedding.reshape(1, -1), self.embed_db)

        # Get the indices of the top k most similar articles
        top_k_indices = np.argsort(cos_sim[0])[::-1][:top_k]

        self.saved_embeds = torch.tensor(self.embed_db[top_k_indices], dtype=torch.float32)
        self.saved_x2 = self.article_projection(self.saved_embeds)

        return torch.tensor(self.embed_db[top_k_indices], dtype=torch.float32)

    def query_articles(self, X, I, stock_name, top_k = 5, show = False):
        query_embedding = embedding_model.encode(f"Stock News About {stock_name}").tolist()

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
        embeddings = [doc['embedding'] for doc in results]
        self.saved_embeds = torch.tensor(embeddings)
        self.saved_x2 = self.article_projection(self.saved_embeds)

        if show:
            string = ""
            for article in results:
                string += f"Title:\n      {article["title"]}\n\n"
                string += f"Body:\n      {article["body"][:200]}\n\n"
                string += "------------------------------------------------------------------\n"

            return string

        return torch.tensor(embeddings)
    
    def get_data(self, stock, start_date, end_date):
        micro_data = self.get_stock_micro_data([stock], start_date, end_date)
        return self.add_stock_macro_data(micro_data, stock)
        
    def get_stock_micro_data(self, list_stocks, start_date, end_date) -> pd.DataFrame:
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

            series_history = ticker.history(start = start_date, end = end_date)[["Close", "Volume"]]
            series_history.columns = ["Close", "Volume"]

            df_income_statement = ticker.financials.rename(index = dict_income_statement_catcher).loc[list_income_statement_metrics]
            df_balance_sheet = ticker.balance_sheet.rename(index = dict_balance_sheet_catcher).loc[list_balance_sheet_metrics]
            df_cash_flow = ticker.cashflow.rename(index = dict_cash_flow_catcher).loc[list_cash_flow_metrics]

            df_one_stock_data = pd.concat([df_income_statement, df_balance_sheet, df_cash_flow])
            df_one_stock_data = df_one_stock_data.T.dropna().resample("D").ffill()

            df_one_stock_data.index = pd.to_datetime(df_one_stock_data.index).tz_localize(None)
            series_history.index = pd.to_datetime(series_history.index).tz_localize(None)

            list_stock_data.append(pd.concat([df_one_stock_data, series_history], axis = 1))

        return pd.concat(list_stock_data, axis = 1, keys = list_stocks).dropna()
    
    def add_stock_macro_data(self, micro_data, stock):
        self.stock = stock
        micro_data.columns = micro_data[stock].columns

        macro_data = pd.read_csv("/Users/mohitunecha/bitsdatathon/data/macro_data.csv", index_col="Date", parse_dates=True)
        micro_data.index = pd.to_datetime(micro_data.index)
        macro_data.index = pd.to_datetime(macro_data.index)

        return pd.merge(macro_data, micro_data, left_index=True, right_index=True, how="inner")