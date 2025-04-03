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
from sklearn.ensemble import GradientBoostingRegressor


uri = "mongodb+srv://am3567:CwfUpOrjtGK1dtnt@main.guajv.mongodb.net/?retryWrites=true&w=majority&appName=Main"
client = MongoClient(uri)
db = client["stocks_db"]
collection = db["mi_data"]

embedding_model = SentenceTransformer("ProsusAI/finbert")

class RAT(nn.Module):
    def __init__(self, stock, input_dim, embed_dim = 64, num_heads = 2, num_layers = 2, output_dim = 2):
        self.stock = stock

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

    def forward(self, x):
        # Encoder
        x1 = self.embedding(x)  # [batch_size, seq_len, embed_dim]

        article_embeds = self.query_articles(self.stock)
        x2 = self.article_projection(article_embeds)
        x2 = x2.unsqueeze(0).expand(x1.shape[0], -1, -1)

        x = torch.cat([x1, x2], dim=1)
        encoded = self.transformer(x1)  # [batch_size, seq_len, embed_dim]
        
        # Get context vector (last hidden state)
        context = encoded[:, -1].unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # Initialize decoder input
        decoder_input = context.repeat(1, self.output_length, 1)  # [batch_size, output_seq_len, embed_dim]
        
        # Decoder
        outputs, _ = self.decoder(decoder_input)  # [batch_size, output_seq_len, embed_dim]
        predictions = self.fc(outputs)  # [batch_size, output_seq_len, output_dim]
        
        return predictions

    def train_model(self, X, y, epochs = 100, batch_size = 32, patience=15, validation_split=0.1):
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
        
        # Format for model input (assuming we need a sequence of input_length)
        if len(new_data) >= self.input_length:
            #model_input = new_data[-self.input_length:].unsqueeze(0)  # Add batch dimension
            model_input = new_data
            
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
            
            return torch.tensor(output_np), torch.tensor(unscaled_output)
        else:
            raise ValueError(f"Input data must have at least {self.input_length} time steps")
        
    def query_articles(self, stock_name, top_k = 5, show = False):
        query_embedding = embedding_model.encode(f"Stock News About {stock_name}").tolist()

        search_query = {
            "$vectorSearch": {
                "index": "vector_index",  # Use your actual index name
                "path": "embedding",  # Ensure this matches your field name
                "queryVector": query_embedding,
                "numCandidates": 100,  # Adjust this based on performance needs
                "limit": top_k
            }
        }

        pipeline = [
            search_query,  # $search stage must be inside an aggregation pipeline
            {"$limit": top_k}  # Optional: Ensures we don't get excessive results
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
        

class Data_Processing():

    def format_data_separate(self, stock_data, financial_data, model, input_length, output_length):
        self.stock_data = stock_data

        model.stock_scalers = []
        model.financial_scalers = []
        
        # Scale each column of stock data independently
        scaled_stock = np.zeros_like(stock_data)
        for i in range(stock_data.shape[1]):
            scaler = StandardScaler()  # You can also use MinMaxScaler()
            # Reshape to 2D array for fit_transform
            col_data = stock_data.values[:, i].reshape(-1, 1)
            scaled_stock[:, i] = scaler.fit_transform(col_data).flatten()
            model.stock_scalers.append(scaler)
        
        # Scale each column of financial data independently
        scaled_financial = np.zeros_like(financial_data)
        for i in range(financial_data.shape[1]):
            scaler = StandardScaler()
            col_data = financial_data.values[:, i].reshape(-1, 1)
            scaled_financial[:, i] = scaler.fit_transform(col_data).flatten()
            model.financial_scalers.append(scaler)
        
        # Convert back to tensors
        stock_data = torch.tensor(scaled_stock, dtype=torch.float32)
        financial_data = torch.tensor(scaled_financial, dtype=torch.float32)

        combined_data = torch.cat([stock_data, financial_data], dim=1)
        
        num_samples = len(combined_data) - input_length - output_length + 1
        X = torch.zeros(num_samples, input_length, model.input_dim)
        y = torch.zeros(num_samples, output_length, model.output_dim)
        
        for i in range(num_samples):
            X[i] = combined_data[i:i+input_length]

            target_values = stock_data[i+input_length:i+input_length+output_length, :model.output_dim]
            y[i] = target_values

        model.input_length = input_length
        model.output_length = output_length

        return X, y

    def format_data_combined(self, combined_data, model, input_length, output_length):
        return self.format_data_separate(combined_data[["Close", "Volume"]], combined_data.drop(["Close", "Volume"], axis = 1), model, input_length, output_length)
    
    def get_data(self, stock, start_date, end_date):
        micro_data = self.get_stock_micro_data([stock], start_date, end_date)
        macro_micro_data = self.add_stock_macro_data(micro_data, stock)
        macro_micro_etf_data = pd.merge(macro_micro_data, self.fetch_etf_data(), left_index=True, right_index=True, how="inner")

        return macro_micro_etf_data
        
    def get_stock_micro_data(self, list_stocks, start_date, end_date) -> pd.DataFrame:
        """Given the stocks to query, return a dataframe of combined historical and financial data
        
        Args:
            list_stocks (list[str]): list of stocks to get data for
            
        Returns:
            pd.Dataframe: their data
            
        """

        for str_stock in list_stocks:
            ticker = yf.Ticker(str_stock)

            series_history = ticker.history(start = start_date, end = end_date)[["Close", "Volume"]]
            series_history.columns = ["Close", "Volume"]

            series_history.index = pd.to_datetime(series_history.index).tz_localize(None)

        return series_history
    
    def add_stock_macro_data(self, micro_data, stock):
        self.stock = stock
        micro_data.columns = micro_data.columns

        macro_data = self.read_macro_data()
        micro_data.index = pd.to_datetime(micro_data.index)
        macro_data.index = pd.to_datetime(macro_data.index)

        return pd.merge(macro_data, micro_data, left_index=True, right_index=True, how="inner")
    
    def read_macro_data(self):
        macro_data = pd.read_csv("../data/new_macro_data.csv", parse_dates=["Date"], index_col="Date")

        # Select only relevant columns
        macro_columns = [
            "Fed_Funds_Rate", "CPI", "GDP", "Unemployment_Rate",
            "Money_Supply", "Retail_Sales", "Oil_Price"
        ]
        macro_data = macro_data[macro_columns].ffill().dropna()
        macro_data.index = pd.to_datetime(macro_data.index)

        return macro_data
  
    def fetch_etf_data(self):
        etf_tickers = [
            "IEF", "LQD", "QQQ", "SPY", "SPYG", "SPYV", "TLT", "^VIX",
            "XLE", "XLF", "XLK", "XLV", "XLY"
        ]
        etf_data = yf.download(etf_tickers)["Close"]

        # Fill forward and drop any missing values
        etf_data = etf_data.ffill().dropna()
        etf_data.index = pd.to_datetime(etf_data.index)

        # Rename columns to be consistent
        etf_data.rename(columns={
            "IEF": "IEF_Close", 
            "LQD": "LQD_Close", 
            "QQQ": "QQQ_Close",
            "SPY": "SPY_Close", 
            "SPYG": "SPYG_Close", 
            "SPYV": "SPYV_Close",
            "TLT": "TLT_Close", 
            "^VIX": "VIX_Close",
            "XLE": "XLE_Close", 
            "XLF": "XLF_Close", 
            "XLK": "XLK_Close",
            "XLV": "XLV_Close", 
            "XLY": "XLY_Close"
        }, inplace=True)

        return etf_data

    def calculate_technical_indicators(self, data):
        # Simple Moving Average (SMA)
        data["SMA_20"] = data["Close"].rolling(window=20).mean()

        # Exponential Moving Average (EMA)
        data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()

        # Relative Strength Index (RSI)
        def calculate_rsi(data, window=14):
            delta = data["Close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=window, min_periods=1).mean()
            avg_loss = loss.rolling(window=window, min_periods=1).mean()

            rs = avg_gain / avg_loss
            data["RSI"] = 100 - (100 / (1 + rs))
            return data

        data = calculate_rsi(data)

        # MACD and Signal Line
        data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
        data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = data["EMA_12"] - data["EMA_26"]
        data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        data["BB_Middle"] = data["Close"].rolling(window=20).mean()
        data["BB_Upper"] = data["BB_Middle"] + 2 * data["Close"].rolling(window=20).std()
        data["BB_Lower"] = data["BB_Middle"] - 2 * data["Close"].rolling(window=20).std()

        data = data.dropna()
        return data

    def merge_all_data(self, stock_data, macro_data, etf_data):
        # Merge stock and macro data
        enriched_data = stock_data.merge(macro_data, left_index=True, right_index=True, how="inner")

        # Merge with ETF/sector data
        enriched_data = enriched_data.merge(etf_data, left_index=True, right_index=True, how="inner")

        # Calculate technical indicators and merge
        enriched_data = self.calculate_technical_indicators(enriched_data)

        return enriched_data
    
    def preprocess_features(self, combined_data):
        df_X = combined_data.drop("Close", axis = 1)  # Replace 'target_column' with your target column
        df_y = combined_data["Close"]

        # Scale features
        scaler = StandardScaler()
        df_X = scaler.fit_transform(df_X)

        # Train the model
        model = GradientBoostingRegressor(n_estimators = 1000,
                                            max_depth=3,
                                            learning_rate=0.1,
                                            n_iter_no_change=10)
        model.fit(df_X, df_y)

        list_importances = model.feature_importances_
        list_feature_names = np.array(combined_data.columns)

        # Sort features by importance
        list_sorted_idx = np.argsort(list_importances)[::-1]  # Sort in descending order
        list_sorted_importances = list_importances[list_sorted_idx]
        list_sorted_features = list_feature_names[list_sorted_idx]

        # Compute cumulative sum of importance
        cumulative_importance = np.cumsum(list_sorted_importances)

        # Select the top features that contribute to 90% of the total importance
        num_features = np.argmax(cumulative_importance >= .9) + 1
        to_keep = set(list_sorted_features[:num_features])

        to_keep.add("Close")
        to_keep.add("Volume")

        return combined_data[list(to_keep)]