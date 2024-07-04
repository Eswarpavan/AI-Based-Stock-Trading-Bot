# AI-Based-Stock-Trading-Bot
This project showcases an AI-driven stock trading bot that leverages machine learning and sentiment analysis to make informed trading decisions. The bot is designed to trade on the Alpaca paper trading platform using the lumibot library for strategy execution and backtesting.

Features
AI-Driven Trading Strategy: Utilizes sentiment analysis of news headlines to inform buy and sell decisions.
Position Sizing: Calculates the optimal number of shares to trade based on available cash and risk parameters.
Backtesting: Supports historical data backtesting using Yahoo Finance data.
Retry Logic: Implements retry logic to handle rate limits and transient errors gracefully.

Prerequisites
Ensure you have the following installed:

Python 3.7+
numpy == 1.26.4
pip (Python package installer)
An Alpaca account with API keys for paper trading

Project Structure
stock_prediction.py: The main script containing the trading strategy, backtesting, and live trading setup.
required.py: finBert transformer model.
