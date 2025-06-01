import streamlit as st

# IMPORTANT: set_page_config() must be the first Streamlit command
st.set_page_config(
    page_title="POS Transaction Analytics Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other libraries
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import io
import base64
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import psutil for memory monitoring, fallback if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Cache function for file processing
@st.cache_data
def process_uploaded_file(file_content, file_name, file_type):
    """Cache file processing to avoid reprocessing on reruns"""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
        else:
            df = pd.read_excel(io.BytesIO(file_content))
        return df, None
    except Exception as e:
        return None, str(e)

# Data validation function
def validate_data_format(df, file_type):
    """Validate uploaded data format"""
    required_columns = {
        'arca': ['TRANSACTION_DATE', 'TERMINAL_ID', 'MERCHANT_ID'],
        'interswitch': ['DATETIME', 'TERMINAL_ID', 'MERCHANT_ID']
    }
    
    missing_cols = [col for col in required_columns[file_type.lower()] 
                   if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns in {file_type} file: {missing_cols}")
        st.info(f"Expected columns: {required_columns[file_type.lower()]}")
        return False
    return True

# Memory usage display function
def display_memory_usage():
    """Display current memory usage"""
    if PSUTIL_AVAILABLE:
        try:
            memory = psutil.virtual_memory()
            st.sidebar.text(f"💾 Memory: {memory.percent:.1f}% used")
        except Exception:
            st.sidebar.text("💾 Memory monitoring unavailable")
    else:
        st.sidebar.text("💾 Memory monitoring unavailable")

class POSTransactionAnalyzer:
    """
    A comprehensive analyzer for POS terminal transactions from Interswitch and Arca settlement reports.
    """
    
    def __init__(self):
        self.arca_columns = {
            'TRANSACTION_DATE': 'date',
            'TERMINAL_ID': 'terminal_id',
            'MERCHANT_ID': 'merchant_id', 
            'MERCHANT_NAME': 'merchant_name',
            'AMOUNT_IMPACT': 'amount'
        }
        
        self.interswitch_columns = {
            'DATETIME': 'date',
            'TERMINAL_ID': 'terminal_id',
            'MERCHANT_ID': 'merchant_id',
            'MERCHANT_ACCOUNT_NAME': 'merchant_name',
            'AMOUNT_IMPACT': 'amount'
        }
        
        self.processed_data = {}
        self.analysis_results = None
    
    def clean_terminal_merchant_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean terminal and merchant IDs by removing trailing spaces."""
        if 'terminal_id' in df.columns:
            df['terminal_id'] = df['terminal_id'].astype(str).str.strip()
        if 'merchant_id' in df.columns:
            df['merchant_id'] = df['merchant_id'].astype(str).str.strip()
        return df
    
    def standardize_date_format(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Standardize date formats across different reports."""
        try:
            # Try multiple date formats
            date_formats = [
                '%Y-%m-%d',
                '%d/%m/%Y', 
                '%m/%d/%Y',
                '%Y-%m-%d %H:%M:%S',
                '%d/%m/%Y %H:%M:%S',
                '%m/%d/%Y %H:%M:%S'
            ]
            
            for fmt in date_formats:
                try:
                    df[date_col] = pd.to_datetime(df[date_col], format=fmt, errors='raise')
                    break
                except:
                    continue
            else:
                # If none of the formats work, use pandas' automatic parsing
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            return df
        except Exception as e:
            st.warning(f"Date parsing issue - {str(e)}")
            return df
    
    def load_and_process_dataframe(self, df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        """Process a pandas DataFrame directly (for Streamlit file uploads)."""
        try:
            # Add file size check
            if len(df) > 100000:  # Adjust limit as needed
                st.warning(f"Large file detected ({len(df):,} rows). Processing may take time.")
            
            # Check if DataFrame is empty
            if df.empty:
                raise ValueError(f"The {file_type} file appears to be empty")
            
            # Validate data format
            if not validate_data_format(df, file_type):
                return pd.DataFrame()
            
            # Select relevant columns based on file type
            if file_type.lower() == 'arca':
                column_mapping = self.arca_columns
            elif file_type.lower() == 'interswitch':
                column_mapping = self.interswitch_columns
            else:
                raise ValueError("file_type must be 'arca' or 'interswitch'")
            
            # Check if required columns exist
            missing_cols = [col for col in column_mapping.keys() if col not in df.columns]
            if missing_cols:
                st.warning(f"Missing columns in {file_type} file: {missing_cols}")
                # Use available columns only
                available_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
                column_mapping = available_mapping
            
            if not column_mapping:
                raise ValueError(f"No valid columns found for {file_type} file")
            
            # Select and rename columns
            df_processed = df[list(column_mapping.keys())].copy()
            df_processed = df_processed.rename(columns=column_mapping)
            
            # Clean IDs
            df_processed = self.clean_terminal_merchant_ids(df_processed)
            
            # Standardize date format
            if 'date' in df_processed.columns:
                df_processed = self.standardize_date_format(df_processed, 'date')
                df_processed['date'] = df_processed['date'].dt.date
            
            # Convert amount to numeric
            if 'amount' in df_processed.columns:
                df_processed['amount'] = pd.to_numeric(df_processed['amount'], errors='coerce').fillna(0)
            
            # Add source column
            df_processed['source'] = file_type.lower()
            
            return df_processed
            
        except pd.errors.EmptyDataError:
            st.error(f"The {file_type} file is empty or corrupted")
            return pd.DataFrame()
        except pd.errors.ParserError as e:
            st.error(f"Error parsing {file_type} file: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error processing {file_type} data: {str(e)}")
            return pd.DataFrame()
    
    def analyze_transactions_with_progress(self, arca_df: pd.DataFrame, interswitch_df: pd.DataFrame, 
                           date_range: Optional[List[str]] = None) -> pd.DataFrame:
        """Analyze transactions with progress tracking."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text('🔄 Combining dataframes...')
            progress_bar.progress(20)
            
            # Combine dataframes
            combined_df = pd.concat([arca_df, interswitch_df], ignore_index=True)
            
            status_text.text('📅 Filtering by date range...')
            progress_bar.progress(40)
            
            # Filter by date range if provided
            if date_range:
                start_date = pd.to_datetime(date_range[0]).date()
                end_date = pd.to_datetime(date_range[1]).date()
                combined_df = combined_df[
                    (combined_df['date'] >= start_date) & 
                    (combined_df['date'] <= end_date)
                ]
            
            status_text.text('🏪 Processing terminals...')
            progress_bar.progress(60)
            
            # Get unique dates for dynamic column creation
            unique_dates = sorted(combined_df['date'].unique()) if 'date' in combined_df.columns else []
            
            # Get all unique terminal IDs
            all_terminals = pd.concat([
                arca_df['terminal_id'], 
                interswitch_df['terminal_id']
            ]).unique()
            
            results = []
            total_terminals = len(all_terminals)
            
            for idx, terminal_id in enumerate(all_terminals):
                # Update progress for terminal processing
                if idx % 100 == 0:  # Update every 100 terminals
                    progress = 60 + int((idx / total_terminals) * 30)
                    progress_bar.progress(progress)
                    status_text.text(f'🏪 Processing terminal {idx + 1}/{total_terminals}...')
                
                terminal_data = combined_df[combined_df['terminal_id'] == terminal_id]
                
                if terminal_data.empty:
                    continue
                
                # Get merchant info (prefer non-null values)
                merchant_id = terminal_data['merchant_id'].dropna().iloc[0] if not terminal_data['merchant_id'].dropna().empty else 'N/A'
                merchant_name = terminal_data['merchant_name'].dropna().iloc[0] if not terminal_data['merchant_name'].dropna().empty else 'N/A'
                
                row_data = {
                    'TID': terminal_id,
                    'MID': merchant_id,
                    'Merchant_Name': merchant_name
                }
                
                total_count = 0
                total_volume = 0
                
                # Process each date
                for date in unique_dates:
                    date_data = terminal_data[terminal_data['date'] == date]
                    
                    # Arca data for this date
                    arca_date_data = date_data[date_data['source'] == 'arca']
                    arca_count = len(arca_date_data)
                    arca_amount = arca_date_data['amount'].sum() if not arca_date_data.empty else 0
                    
                    # Interswitch data for this date
                    interswitch_date_data = date_data[date_data['source'] == 'interswitch']
                    interswitch_count = len(interswitch_date_data)
                    interswitch_amount = interswitch_date_data['amount'].sum() if not interswitch_date_data.empty else 0
                    
                    # Add to row data
                    date_str = date.strftime('%d %b %Y')
                    row_data[f'{date_str}_Interswitch_Count'] = interswitch_count
                    row_data[f'{date_str}_Interswitch_Amount'] = interswitch_amount
                    row_data[f'{date_str}_Arca_Count'] = arca_count
                    row_data[f'{date_str}_Arca_Amount'] = arca_amount
                    
                    total_count += arca_count + interswitch_count
                    total_volume += arca_amount + interswitch_amount
                
                row_data['Total_Count'] = total_count
                row_data['Total_Volume'] = total_volume
                
                results.append(row_data)
            
            status_text.text('📊 Finalizing results...')
            progress_bar.progress(90)
            
            # Create DataFrame
            results_df = pd.DataFrame(results)
            
            # Sort by TID
            results_df = results_df.sort_values('TID').reset_index(drop=True)
            
            progress_bar.progress(100)
            status_text.text('✅ Analysis complete!')
            
            self.analysis_results = results_df
            return results_df
            
        except Exception as e:
            raise Exception(f"Error during analysis: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()
    
    def analyze_transactions(self, arca_df: pd.DataFrame, interswitch_df: pd.DataFrame, 
                           date_range: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Legacy analyze_transactions method for backward compatibility.
        Now calls the progress version.
        """
        return self.analyze_transactions_with_progress(arca_df, interswitch_df, date_range)
    
    def get_summary_stats(self, results_df: pd.DataFrame) -> Dict:
        """Get summary statistics from the analysis results."""
        try:
            stats = {
                'total_terminals': len(results_df),
                'total_merchants': results_df['MID'].nunique(),
                'total_transactions': results_df['Total_Count'].sum(),
                'total_volume': results_df['Total_Volume'].sum(),
                'avg_transactions_per_terminal': results_df['Total_Count'].mean(),
                'avg_volume_per_terminal': results_df['Total_Volume'].mean(),
                'top_terminals_by_count': results_df.nlargest(5, 'Total_Count')[['TID', 'Merchant_Name', 'Total_Count']].to_dict('records'),
                'top_terminals_by_volume': results_df.nlargest(5, 'Total_Volume')[['TID', 'Merchant_Name', 'Total_Volume']].to_dict('records')
            }
            return stats
        except Exception as e:
            st.error(f"Error calculating summary stats: {str(e)}")
            return {}

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    
    /* Metric card improvements for better visibility */
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Force metric styling to be visible */
    .stMetric {
        background-color: #ffffff !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        border: 1px solid #e0e0e0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Metric label styling */
    .stMetric > div > div > div > div {
        color: #333333 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    /* Metric value styling */
    .stMetric > div > div > div[data-testid="metric-value"] {
        color: #1f77b4 !important;
        font-size: 1.8rem !important;
        font-weight: bold !important;
    }
    
    /* Alternative approach for metric values */
    [data-testid="metric-value"] {
        color: #1f77b4 !important;
        font-size: 1.8rem !important;
        font-weight: bold !important;
    }
    
    /* Metric delta styling */
    [data-testid="metric-delta"] {
        color: #666666 !important;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    
    /* Download section */
    .download-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border: 1px solid #e9ecef;
    }
    
    /* Force dark mode compatibility */
    .stApp > div {
        background-color: #ffffff;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 0.25rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background-color: #1565c0;
        color: white;
    }
    
    /* File uploader styling */
    .stFileUploader > div > div > div {
        background-color: #ffffff;
        border: 2px dashed #1f77b4;
        border-radius: 0.5rem;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: #ffffff;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #ccc;
    }
    
    /* Ensure all text is readable */
    .stMarkdown, .stText, p, div, span {
        color: #333333 !important;
    }
    
    /* Force metric containers to have proper contrast */
    .metric-container {
        background: white !important;
        color: #333 !important;
        border: 1px solid #ddd !important;
        border-radius: 8px !important;
        padding: 16px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = POSTransactionAnalyzer()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def main():
    # Header
    st.markdown('<h1 class="main-header">💳 POS Transaction Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for file uploads and controls
    st.sidebar.title("📁 Data Upload & Controls")
    
    # File upload section
    st.sidebar.markdown("### Upload Settlement Reports")
    
    # Arca files
    st.sidebar.markdown("**Arca Reports:**")
    arca_files = st.sidebar.file_uploader(
        "Upload Arca settlement reports", 
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        key="arca_files"
    )
    
    # Interswitch files  
    st.sidebar.markdown("**Interswitch Reports:**")
    interswitch_files = st.sidebar.file_uploader(
        "Upload Interswitch settlement reports",
        type=['csv', 'xlsx', 'xls'], 
        accept_multiple_files=True,
        key="interswitch_files"
    )
    
    # Date range filter
    st.sidebar.markdown("### 📅 Date Range Filter (Optional)")
    use_date_filter = st.sidebar.checkbox("Apply date range filter")
    
    date_range = None
    if use_date_filter:
        start_date = st.sidebar.date_input("Start Date")
        end_date = st.sidebar.date_input("End Date")
        if start_date and end_date:
            date_range = [start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')]
    
    # Analysis button
    analyze_button = st.sidebar.button("🔍 Run Analysis", type="primary")
    
    # Main content area
    if analyze_button and (arca_files or interswitch_files):
        with st.spinner("Processing files and running analysis..."):
            try:
                # Process uploaded files
                arca_dfs = []
                interswitch_dfs = []
                
                # Process Arca files
                for arca_file in arca_files:
                    # Read file content directly
                    if arca_file.name.endswith('.csv'):
                        df = pd.read_csv(arca_file)
                    else:
                        df = pd.read_excel(arca_file)
                    
                    # Process using analyzer
                    processed_df = st.session_state.analyzer.load_and_process_dataframe(df, "arca")
                    arca_dfs.append(processed_df)
                
                # Process Interswitch files
                for interswitch_file in interswitch_files:
                    # Read file content directly
                    if interswitch_file.name.endswith('.csv'):
                        df = pd.read_csv(interswitch_file)
                    else:
                        df = pd.read_excel(interswitch_file)
                    
                    # Process using analyzer
                    processed_df = st.session_state.analyzer.load_and_process_dataframe(df, "interswitch")
                    interswitch_dfs.append(processed_df)
                
                # Combine dataframes
                arca_combined = pd.concat(arca_dfs, ignore_index=True) if arca_dfs else pd.DataFrame()
                interswitch_combined = pd.concat(interswitch_dfs, ignore_index=True) if interswitch_dfs else pd.DataFrame()
                
                # Run analysis
                results = st.session_state.analyzer.analyze_transactions(
                    arca_combined, interswitch_combined, date_range
                )
                
                st.session_state.analysis_results = results
                
                st.success("✅ Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                return
    
    # Display results if available - THIS IS THE KEY FIX
    if st.session_state.analysis_results is not None:
        results_df = st.session_state.analysis_results
        
        # Summary Statistics with better layout
        st.markdown('<h2 class="section-header">📊 Summary Statistics</h2>', unsafe_allow_html=True)

        # Calculate summary stats
        stats = st.session_state.analyzer.get_summary_stats(results_df)

        # Create a container for better spacing and styling
        with st.container():
            # Add custom CSS for this specific container
            st.markdown("""
            <style>
            div[data-testid="column"] {
                background-color: white !important;
                padding: 1rem !important;
                border-radius: 0.5rem !important;
                border: 1px solid #e0e0e0 !important;
                margin: 0.25rem !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Display metrics in columns with better spacing
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1.2])
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div style="font-size: 0.9rem; color: #666; font-weight: 600; margin-bottom: 8px;">
                        Total Terminals
                    </div>
                    <div style="font-size: 1.8rem; color: #1f77b4; font-weight: bold;">
                        {stats.get('total_terminals', 0):,}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div style="font-size: 0.9rem; color: #666; font-weight: 600; margin-bottom: 8px;">
                        Total Merchants
                    </div>
                    <div style="font-size: 1.8rem; color: #1f77b4; font-weight: bold;">
                        {stats.get('total_merchants', 0):,}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <div style="font-size: 0.9rem; color: #666; font-weight: 600; margin-bottom: 8px;">
                        Total Transactions
                    </div>
                    <div style="font-size: 1.8rem; color: #1f77b4; font-weight: bold;">
                        {stats.get('total_transactions', 0):,}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                total_volume = stats.get('total_volume', 0)
                if total_volume >= 1_000_000_000:  # Billions
                    volume_display = f"₦{total_volume/1_000_000_000:.1f}B"
                elif total_volume >= 1_000_000:  # Millions
                    volume_display = f"₦{total_volume/1_000_000:.1f}M"
                elif total_volume >= 1_000:  # Thousands
                    volume_display = f"₦{total_volume/1_000:.1f}K"
                else:
                    volume_display = f"₦{total_volume:,.0f}"
                
                st.markdown(f"""
                <div class="metric-container">
                    <div style="font-size: 0.9rem; color: #666; font-weight: 600; margin-bottom: 8px;">
                        Total Volume
                    </div>
                    <div style="font-size: 1.8rem; color: #1f77b4; font-weight: bold;">
                        {volume_display}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Add additional metrics row for more details
            st.markdown("---")
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                avg_trans = stats.get('avg_transactions_per_terminal', 0)
                st.markdown(f"""
                <div class="metric-container">
                    <div style="font-size: 0.9rem; color: #666; font-weight: 600; margin-bottom: 8px;">
                        Avg Trans/Terminal
                    </div>
                    <div style="font-size: 1.8rem; color: #1f77b4; font-weight: bold;">
                        {avg_trans:.1f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                avg_volume = stats.get('avg_volume_per_terminal', 0)
                if avg_volume >= 1_000_000:
                    avg_display = f"₦{avg_volume/1_000_000:.1f}M"
                elif avg_volume >= 1_000:
                    avg_display = f"₦{avg_volume/1_000:.1f}K"
                else:
                    avg_display = f"₦{avg_volume:,.0f}"
                
                st.markdown(f"""
                <div class="metric-container">
                    <div style="font-size: 0.9rem; color: #666; font-weight: 600; margin-bottom: 8px;">
                        Avg Volume/Terminal
                    </div>
                    <div style="font-size: 1.8rem; color: #1f77b4; font-weight: bold;">
                        {avg_display}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col7:
                # Active terminals (terminals with transactions > 0)
                active_terminals = len(results_df[results_df['Total_Count'] > 0])
                st.markdown(f"""
                <div class="metric-container">
                    <div style="font-size: 0.9rem; color: #666; font-weight: 600; margin-bottom: 8px;">
                        Active Terminals
                    </div>
                    <div style="font-size: 1.8rem; color: #1f77b4; font-weight: bold;">
                        {active_terminals:,}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col8:
                # Success rate (assuming any transaction count means success)
                success_rate = (active_terminals / stats.get('total_terminals', 1)) * 100
                st.markdown(f"""
                <div class="metric-container">
                    <div style="font-size: 0.9rem; color: #666; font-weight: 600; margin-bottom: 8px;">
                        Activity Rate
                    </div>
                    <div style="font-size: 1.8rem; color: #1f77b4; font-weight: bold;">
                        {success_rate:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Charts section
            st.markdown('<h2 class="section-header">📈 Analytics Charts</h2>', unsafe_allow_html=True)
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Top terminals by transaction count
                if 'top_terminals_by_count' in stats and stats['top_terminals_by_count']:
                    top_count_df = pd.DataFrame(stats['top_terminals_by_count'])
                    fig_count = px.bar(
                        top_count_df,
                        x='TID',
                        y='Total_Count',
                        title='Top 5 Terminals by Transaction Count',
                        labels={'Total_Count': 'Transaction Count', 'TID': 'Terminal ID'}
                    )
                    fig_count.update_layout(height=400)
                    st.plotly_chart(fig_count, use_container_width=True)
            
            with chart_col2:
                # Top terminals by volume
                if 'top_terminals_by_volume' in stats and stats['top_terminals_by_volume']:
                    top_volume_df = pd.DataFrame(stats['top_terminals_by_volume'])
                    fig_volume = px.bar(
                        top_volume_df,
                        x='TID',
                        y='Total_Volume',
                        title='Top 5 Terminals by Transaction Volume',
                        labels={'Total_Volume': 'Transaction Volume (₦)', 'TID': 'Terminal ID'}
                    )
                    fig_volume.update_layout(height=400)
                    st.plotly_chart(fig_volume, use_container_width=True)
            
            # Transaction distribution
            st.markdown("### Transaction Distribution Analysis")
            
            # Create distribution charts
            dist_col1, dist_col2 = st.columns(2)
            
            with dist_col1:
                fig_hist_count = px.histogram(
                    results_df,
                    x='Total_Count',
                    title='Distribution of Transaction Counts per Terminal',
                    labels={'Total_Count': 'Transaction Count', 'count': 'Number of Terminals'}
                )
                st.plotly_chart(fig_hist_count, use_container_width=True)
            
            with dist_col2:
                fig_hist_volume = px.histogram(
                    results_df,
                    x='Total_Volume',
                    title='Distribution of Transaction Volumes per Terminal',
                    labels={'Total_Volume': 'Transaction Volume (₦)', 'count': 'Number of Terminals'}
                )
                st.plotly_chart(fig_hist_volume, use_container_width=True)
            
            # Data table
            st.markdown('<h2 class="section-header">📋 Detailed Analysis Results</h2>', unsafe_allow_html=True)
            
            # Add search and filter options
            search_term = st.text_input("🔍 Search terminals or merchants:", "")
            
            # Filter dataframe based on search
            if search_term:
                mask = (
                    results_df['TID'].astype(str).str.contains(search_term, case=False, na=False) |
                    results_df['MID'].astype(str).str.contains(search_term, case=False, na=False) |
                    results_df['Merchant_Name'].astype(str).str.contains(search_term, case=False, na=False)
                )
                filtered_df = results_df[mask]
            else:
                filtered_df = results_df
            
            # Display dataframe with better formatting
            # Format the dataframe for better display
            display_df = filtered_df.copy()
            
            # Format monetary columns
            money_columns = [col for col in display_df.columns if 'Amount' in col or col == 'Total_Volume']
            for col in money_columns:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"₦{x:,.0f}" if pd.notnull(x) and x != 0 else "₦0")
            
            # Format count columns
            count_columns = [col for col in display_df.columns if 'Count' in col]
            for col in count_columns:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else "0")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
            
            # Download section
            st.markdown('<h2 class="section-header">💾 Download Reports</h2>', unsafe_allow_html=True)
            st.markdown('<div class="download-section">', unsafe_allow_html=True)
            
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                st.markdown("**Download Current Analysis:**")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # CSV download
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv,
                    file_name=f"pos_analysis_{timestamp}.csv",
                    mime="text/csv"
                )
            
            with download_col2:
                st.markdown("**Download as Excel:**")
                # Excel download
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    filtered_df.to_excel(writer, sheet_name='Analysis', index=False)
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📊 Download as Excel",
                    data=excel_data,
                    file_name=f"pos_analysis_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Show instructions when no analysis has been run yet
        st.markdown("""
        ### How to use this dashboard:
        
        1. **Upload Files**: Use the sidebar to upload your Arca and Interswitch settlement reports
           - Supported formats: CSV, Excel (.xlsx, .xls)
           - You can upload multiple files of each type
        
        2. **Set Date Range** (Optional): Apply date filters to analyze specific time periods
        
        3. **Run Analysis**: Click the "Run Analysis" button to process your data
        
        4. **View Results**: 
           - Summary statistics and KPIs
           - Interactive charts and visualizations
           - Detailed data table with search functionality
           - Download reports in CSV or Excel format
        
        ### Features:
        - ✅ Automatic data cleaning and standardization
        - ✅ Multi-date analysis support
        - ✅ Real-time dashboard updates
        - ✅ Professional report generation
        - ✅ Interactive data exploration
        """)
        
        # Sample data format
        with st.expander("📋 View Expected Data Format"):
            st.markdown("**Arca Report Columns:**")
            st.code("""
            TRANSACTION_DATE, TERMINAL_ID, MERCHANT_ID, MERCHANT_NAME, 
            AMOUNT_IMPACT, [other columns...]
            """)
            
            st.markdown("**Interswitch Report Columns:**")
            st.code("""
            DATETIME, TERMINAL_ID, MERCHANT_ID, MERCHANT_ACCOUNT_NAME,
            AMOUNT_IMPACT, [other columns...]
            """)

if __name__ == "__main__":
    main()
