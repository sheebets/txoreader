import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

st.title('Betting Data Analyzer')
st.text('Filter and visualize betting data with RECDATE vs PRICES scatter plot')

# File uploader
upload_file = st.file_uploader('Upload your CSV file', type=['csv'])

if upload_file:
    # Load data
    df = pd.read_csv(upload_file)
    
    st.header('Data Overview')
    st.write(f"Total records: {len(df)}")
    st.write("Column names:", list(df.columns))
    
    # Show sample data
    with st.expander("View Sample Data"):
        st.write(df.head())
    
    # Create filters in sidebar (always shown)
    st.sidebar.header('Filters')
    
    # Date filter for RECDATE
    st.sidebar.subheader('Date Range')
    
    # Convert RECDATE to datetime for filtering
    try:
        df['RECDATE_temp'] = pd.to_datetime(df['RECDATE'])
        min_date = df['RECDATE_temp'].min().date()
        max_date = df['RECDATE_temp'].max().date()
        
        # Date range picker
        date_range = st.sidebar.date_input(
            'Select Date Range:',
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Handle single date selection
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range if hasattr(date_range, 'year') else min_date
            
        st.sidebar.write(f"Selected: {start_date} to {end_date}")
        
    except Exception as e:
        st.sidebar.error(f"Could not parse dates: {str(e)}")
        date_range = None
        df['RECDATE_temp'] = df['RECDATE']
    
    st.sidebar.subheader('Other Filters')
    
    # SOT filter
    sot_options = ['All'] + sorted(df['SOT'].dropna().unique().tolist())
    selected_sot = st.sidebar.selectbox('Select SOT:', sot_options)
    
    # Apply SOT filter first to narrow down Market Parameters options
    temp_filtered_df = df.copy()
    if selected_sot != 'All':
        temp_filtered_df = temp_filtered_df[temp_filtered_df['SOT'] == selected_sot]
    
    # MARKETPARAMETERS filter with dynamic options based on SOT selection
    available_markets = sorted(temp_filtered_df['MARKETPARAMETERS'].dropna().unique().tolist())
    
    # Search box for Market Parameters
    search_term = st.sidebar.text_input('Search Market Parameters:', placeholder='Type to search...')
    
    # Filter market options based on search term
    if search_term:
        filtered_markets = [market for market in available_markets 
                          if search_term.lower() in str(market).lower()]
    else:
        filtered_markets = available_markets
    
    # Show count of available options
    st.sidebar.write(f"Available Market Parameters: {len(filtered_markets)}")
    
    # Market Parameters selectbox with filtered options
    market_options = ['All'] + filtered_markets
    selected_market = st.sidebar.selectbox('Select Market Parameters:', market_options)
    
    # Apply SOT and Market Parameters filters to get relevant bookmakers
    temp_filtered_df2 = temp_filtered_df.copy()
    if selected_market != 'All':
        temp_filtered_df2 = temp_filtered_df2[temp_filtered_df2['MARKETPARAMETERS'] == selected_market]
    
    # INRUNNING filter with relevant options
    available_inrunning = sorted(temp_filtered_df2['INRUNNING'].dropna().unique().tolist())
    inrunning_options = ['All'] + available_inrunning
    selected_inrunning = st.sidebar.selectbox('Select In-Running:', inrunning_options)
    
    # Apply INRUNNING filter
    temp_filtered_df3 = temp_filtered_df2.copy()
    if selected_inrunning != 'All':
        temp_filtered_df3 = temp_filtered_df3[temp_filtered_df3['INRUNNING'] == selected_inrunning]
    
    # BOOKMAKER filter with multiselect and relevant options only
    available_bookmakers = sorted(temp_filtered_df3['BOOKMAKER'].dropna().unique().tolist())
    st.sidebar.write(f"Available Bookmakers: {len(available_bookmakers)}")
    
    selected_bookmakers = st.sidebar.multiselect('Select Bookmakers:', 
                                                 available_bookmakers, 
                                                 default=available_bookmakers[:5] if len(available_bookmakers) > 5 else available_bookmakers)
    
    # Apply filters
    filtered_df = df.copy()
    
    # Apply date filter
    if date_range and hasattr(start_date, 'year'):
        try:
            # Convert start and end dates to datetime for comparison
            start_datetime = pd.to_datetime(start_date)
            end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # Include full end day
            
            filtered_df = filtered_df[
                (filtered_df['RECDATE_temp'] >= start_datetime) & 
                (filtered_df['RECDATE_temp'] <= end_datetime)
            ]
        except Exception as e:
            st.warning(f"Date filtering error: {str(e)}")
    
    if selected_sot != 'All':
        filtered_df = filtered_df[filtered_df['SOT'] == selected_sot]
    
    if selected_market != 'All':
        filtered_df = filtered_df[filtered_df['MARKETPARAMETERS'] == selected_market]
    
    if selected_inrunning != 'All':
        filtered_df = filtered_df[filtered_df['INRUNNING'] == selected_inrunning]
    
    if selected_bookmakers:
        filtered_df = filtered_df[filtered_df['BOOKMAKER'].isin(selected_bookmakers)]
    
    st.header('Filtered Data')
    st.write(f"Filtered records: {len(filtered_df)}")
    
    if len(filtered_df) > 0:
        # Data preprocessing for plotting
        plot_df = filtered_df.copy()
        
        # Convert RECDATE to datetime (handle various formats)
        try:
            plot_df['RECDATE_parsed'] = pd.to_datetime(plot_df['RECDATE'])
        except:
            st.warning("Could not parse all RECDATE values. Using original values.")
            plot_df['RECDATE_parsed'] = plot_df['RECDATE']
        
        # Handle PRICES - convert to numeric if possible
        def parse_prices(price_str):
            try:
                # If it's already a number
                if pd.isna(price_str):
                    return np.nan
                
                # Try to convert directly to float
                return float(price_str)
            except:
                try:
                    # Try to extract first number from string if it contains multiple values
                    import re
                    numbers = re.findall(r'\d+\.?\d*', str(price_str))
                    if numbers:
                        return float(numbers[0])
                    return np.nan
                except:
                    return np.nan
        
        plot_df['PRICES_numeric'] = plot_df['PRICES'].apply(parse_prices)
        
        # Remove rows where we couldn't parse prices
        plot_df_clean = plot_df.dropna(subset=['PRICES_numeric'])
        
        if len(plot_df_clean) > 0:
            st.header('Scatter Plot: RECDATE vs PRICES (Color-coded by Bookmaker)')
            
            # Create the plot
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Get unique bookmakers and create color map
            unique_bookmakers = plot_df_clean['BOOKMAKER'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_bookmakers)))
            color_map = dict(zip(unique_bookmakers, colors))
            
            # Create scatter plot for each bookmaker (X = RECDATE, Y = PRICES)
            for bookmaker in unique_bookmakers:
                bookmaker_data = plot_df_clean[plot_df_clean['BOOKMAKER'] == bookmaker]
                ax.scatter(x=bookmaker_data['RECDATE_parsed'], 
                          y=bookmaker_data['PRICES_numeric'],
                          alpha=0.7, 
                          c=[color_map[bookmaker]], 
                          label=bookmaker,
                          s=50)
            
            ax.set_xlabel('RECDATE')
            ax.set_ylabel('PRICES')
            ax.set_title(f'RECDATE vs PRICES by Bookmaker (n={len(plot_df_clean)})')
            
            # Add legend
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Rotate x-axis labels if they're dates
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Show summary statistics
            st.header('Summary Statistics')
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader('PRICES Statistics')
                st.write(plot_df_clean['PRICES_numeric'].describe())
            
            with col2:
                st.subheader('Filtered Data Sample')
                st.write(plot_df_clean[['RECDATE', 'PRICES', 'SOT', 'BOOKMAKER']].head(10))
                
        else:
            st.error("No valid numeric price data found after filtering. Please check your PRICES column format.")
            st.write("Sample PRICES values:", df['PRICES'].head(10).tolist())
    
    else:
        st.warning("No data matches the selected filters. Please adjust your filter criteria.")
        
else:
    st.info("Please upload a CSV file to get started.")
    st.write("Expected columns: SPORT, SOT, MARKETPARAMETERS, INRUNNING, BOOKMAKER, RECDATE, PRICES, etc.")
