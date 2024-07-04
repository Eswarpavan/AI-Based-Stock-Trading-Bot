from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime
from alpaca_trade_api import REST
from timedelta import Timedelta
from required import estimate_sentiment

# Alpaca API credentials
API_KEY = "PK6F3FI9JVHP2LUKW636"
API_SECRET = "31dBvdjySIwOAntATDERjHfTEveR5y8grS05zF8c"
BASE_URL = "https://paper-api.alpaca.markets"
# Alpaca API credentials dictionary
ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}

# Define the AI Trading Bot Strategy
class AI_Trading_bot(Strategy):
    def initialize(self, symbol: str = "SPY", cash_at_risk: float = .5):
        self.symbol = symbol
        self.sleeptime = "24H"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    # Calculate the position sizing
    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity

    # Get the current and past dates for sentiment analysis
    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    # Fetch news and estimate sentiment
    def get_sentiment(self):
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol,
                                 start=three_days_prior,
                                 end=today)
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment

    # Define trading logic for each iteration
    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()

        if cash > last_price:
            if sentiment == "positive" and probability > .999:
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price * 1.20,
                    stop_loss_price=last_price * .95
                )
                self.submit_order(order)
                self.last_trade = "buy"
            elif sentiment == "negative" and probability > .999:
                if self.last_trade == "buy":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=last_price * .8,
                    stop_loss_price=last_price * 1.05
                )
                self.submit_order(order)
                self.last_trade = "sell"

# Define the backtesting parameters
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 6, 30)
broker = Alpaca(ALPACA_CREDS)
strategy = AI_Trading_bot(name='trade_bot', broker=broker,
                          parameters={"symbol": "SPY",
                                      "cash_at_risk": .5})

# Run the backtest
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol": "SPY", "cash_at_risk": .5}
)

# Uncomment the lines below to run the strategy live
#trader = Trader()
#trader.add_strategy(strategy)
#trader.run_all()
