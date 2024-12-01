

import polars as pl
import numpy as np  
import pandas as pd

def wkly_seasn_simple(
    df: pl.DataFrame,
    X_years: int = 5,
    ewma_span: int = 10,
    ret_type: str = 'pct',
    rm_outliers: float = 5,
    min_lb: int = 0,
    hr_threshold_long: float = 0.5,
    hr_threshold_short: float = -0.5,
    ret_threshold: float = 0.5
) -> pl.DataFrame:
    """
    Implements a trading rule based on weekly seasonality.

    Parameters:
    - df: Polars DataFrame with 'Date', 'Open', 'High', 'Low', 'Close' columns.
    - X_years: Lookback period for seasonality.
    - ewma_span: EWMA smoothing period.
    - ret_type: 'abs' for absolute returns, 'pct' for percentage returns.
    - rm_outliers: Threshold for outlier removal.
    - min_lb: Minimum lookback period for rolling calculations.
    - hr_threshold_long: Threshold for hit rate z-score to go long.
    - hr_threshold_short: Threshold for hit rate z-score to go short.
    - ret_threshold: Threshold for return z-score to validate the signal.

    Returns:
    - Polars DataFrame with columns: 'Signal', 'TradeEntry', 'TradeExit', 'InTrade'.
    """

    # Convert Polars DataFrame to Pandas DataFrame for processing

    # Generate seasonal features
    feats, wklyret, df = generate_seasonal_features(
        df, 
        X_years=X_years, 
        ewma_span=ewma_span, 
        ret_type=ret_type, 
        rm_outliers=rm_outliers, 
        min_lb=min_lb
    )

    df.drop(['Daily_Return','Weekly_Return'], axis=1, errors='ignore', inplace=True)
    
    # Initialize columns for trading rule
    df['Signal'] = 0
    df['TradeEntry'] = None
    df['TradeExit'] = None
    df['InTrade'] = 0

    in_trade = False
    trade_start_date = None
    trade_end_date = None

    # Get a list of all dates for indexing
    all_dates = df.index

    for date in all_dates:
        row = df.loc[date]
        current_week = row['ISO_Week']
        current_year = row['ISO_Year']
        week_key = f'w{current_week}'

        # Skip if the current week has no seasonal data
        if week_key not in feats:
            continue

        # Get the seasonal data for the current week up to the previous year
        seasonal_data = feats[week_key]
        seasonal_data = seasonal_data[seasonal_data['ISO_Year'] < current_year]

        # Skip if insufficient historical data
        if seasonal_data.empty:
            continue

        # Calculate current z-scores
        hr_zs = seasonal_data['hr_zs'].iloc[-1]
        ret_zs = seasonal_data['ret_zs'].iloc[-1]

        # Determine trade direction (long or short)
        signal = 0
        if hr_zs > hr_threshold_long and ret_zs > ret_threshold:
            signal = 1  # Long
        elif hr_zs < hr_threshold_short and ret_zs < -ret_threshold:
            signal = -1  # Short

        # Only update signal if not already in a trade
        if not in_trade and signal != 0:
            df.loc[date, 'Signal'] = signal

            # Enter trade at the close of the first trading day of the current week
            current_week_data = df[(df['ISO_Week'] == current_week) & (df['ISO_Year'] == current_year)]
            if not current_week_data.empty:
                trade_entry_date = current_week_data.index[0]  # First trading day of current week
                previous_date = df.index[df.index.get_loc(trade_entry_date) - 1]
                df.loc[trade_entry_date, 'TradeEntry'] = df.loc[previous_date, 'Close']
                in_trade = True
                trade_start_date = trade_entry_date

                # Determine trade exit date (last trading day of the current week)
                trade_exit_date = current_week_data.index[-1]
                trade_end_date = trade_exit_date
                df.loc[trade_exit_date, 'TradeExit'] = df.loc[trade_exit_date, 'Close']
    
                # Set InTrade for the duration of the trade
                df.loc[trade_start_date:trade_end_date, 'InTrade'] = signal
                df.loc[trade_end_date,'Signal'] = - signal


        # Reset in_trade status after trade ends
        if in_trade and date == trade_end_date:
            in_trade = False
            trade_start_date = None
            trade_end_date = None
            df.loc[trade_end_date,'Signal'] = - signal


    # Convert back to Polars DataFrame
    return pl.from_pandas(df.reset_index())