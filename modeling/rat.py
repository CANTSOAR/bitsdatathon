import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer

#Retrieval Augmented Transformer
class RAT(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, output_dim):
        super(RAT, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        #article logic
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])  # Predicting based on the last output
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

    def format_data(self, stock_data, financial_data, input_length, output_length):
        stock_data = torch.tensor(stock_data, dtype=torch.float32)
        financial_data = torch.tensor(financial_data, dtype=torch.float32)

        combined_data = torch.cat([stock_data, financial_data], dim=1)
        
        # Create sliding windows
        num_samples = len(combined_data) - input_length - output_length + 1
        X = torch.zeros(num_samples, input_length, self.input_dim)
        y = torch.zeros(num_samples, output_length, self.output_dim)
        
        for i in range(num_samples):
            X[i] = combined_data[i:i+input_length]

            target_values = stock_data[i+input_length:i+input_length+output_length, :self.output_dim]
            y[i] = target_values

        return X, y

    def query_articles(self, X, I):
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

        results = list(news_collection.aggregate(pipeline))
        return results

"""N days x k features + I(nformation)

N = 30: jan 1 - jan 30

k = 6: Dates, Close price, Volume, Current Interest Rate, VIX, PE_ratio

I = basic information: name, summary, sector, industry"""

        

# Example instantiation
model = RAT(input_dim=20, embed_dim=64, num_heads=4, num_layers=3, output_dim=1)
