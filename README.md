# bitsdatathon


We built a stock prediction system that combines financial data and AI-powered news analysis to predict market trends. First, we pull historical stock data, including prices, volume, and financial ratios, using Yahoo Finance. We then store and retrieve relevant news articles from MongoDB, converting them into numerical vectors using a pre-trained language model (FinBERT). This allows us to analyze the sentiment and relevance of news articles to specific stocks. To predict stock movements, we trained a Support Vector Machine (SVM) model using technical indicators like moving averages and RSI, along with financial metrics and news sentiment. Additionally, we implemented a Retrieval-Augmented Transformer (RAT) model, which retrieves similar past articles and stock trends to make deep learning-driven predictions. As an experiment, we also tested Ollama’s Llama3.2 model to generate embeddings and compare results with FinBERT. The final system can predict whether a stock is a buy or sell opportunity by combining stock history, financial data, and news analysis, improving accuracy through AI-driven insights. 🚀


Summary Slides:
https://docs.google.com/presentation/d/1_Irunnqc5JmI4UnlQV4uCyn5y5oxXly69zMXx3dptgg/edit?usp=sharing
