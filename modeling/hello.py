import streamlit as st
import pandas as pd
import plotly.express as px
from rat import RAT

def main():
    st.title("Stock Prediction System")
    
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
    start_date = st.date_input("Select Start Date:")
    end_date = st.date_input("Select End Date:")
    
    if st.button("Predict"): 
        st.write("Fetching stock data...")
        
        # Initialize RAT model
        model = RAT(input_dim=22, embed_dim=64, num_heads=4, num_layers=2, output_dim=1)
        
        # Get stock and macro data from RAT
        # Get data (assuming this returns DataFrame with all features)
        data = model.get_data(ticker, start_date="2020-01-01", end_date=end_date)

        # Separate stock price/volume from other features
        stock_features = data[['Close', 'Volume']].values  # Price and volume
        other_features = data.drop(['Close', 'Volume'], axis=1).values.astype(dtype="float32")  # Other features

        # Format data for training
        input_length = 30  # 30 days of history
        output_length = 15  # predict 15 days ahead
        X, y = model.format_data_separate(stock_features, other_features, input_length, output_length)

        # Train the model
        batch_size = 32
        epochs = 100
        model.train_model(epochs, batch_size, X, y)
        
        if True:
            st.header("Stock Information:", ticker)
            st.subheader("Stock Price & Prediction")

            data = model.get_data(ticker, start_date=start_date, end_date=end_date)

            # Separate stock price/volume from other features
            stock_features = data[['Close', 'Volume']].values  # Price and volume
            other_features = data.drop(['Close', 'Volume'], axis=1).values.astype(dtype="float32")  # Other features

            # Format data for training
            input_length = 30  # 30 days of history
            output_length = 15  # predict 15 days ahead
            X, y = model.format_data_separate(stock_features, other_features, input_length, output_length)
            
            # Make prediction using RAT
            prediction = model.predict(X)[0, :, 0]
            st.subheader(f"Prediction: {prediction[-1]}")

            end_date_extended = pd.to_datetime(end_date) + pd.Timedelta(days=15)

            # Display Stock Data Graph
            fig = px.line(x=range(len(prediction)), y=prediction, title=f"{ticker} Stock Price")
            st.plotly_chart(fig)


            st.subheader(f"Relevant Articles(?):")
            st.write(f"{model.query_articles("", "", ticker, show = True)}")
        else:
            st.error("Error retrieving stock data. Please check the ticker and date range.")

if __name__ == "__main__":
    main()
