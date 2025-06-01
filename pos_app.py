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
import re
import hashlib
import logging
from io import StringIO
import time
import gc
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Create in-memory log handler instead of file-based
log_stream = StringIO()
logging.basicConfig(
    stream=log_stream,  # ✅ Use in-memory stream instead of file
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_security_event(event_type, details):
    """Log security events to memory"""
    logging.info(f"SECURITY_EVENT: {event_type} - {details}")

# Security imports
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    # Don't show warning in production - just log it
    logging.warning("python-magic not available. File type validation will be limited.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Don't show warning in production
    logging.warning("psutil not available. System monitoring will be limited.")

# Configure secure logging
logging.basicConfig(
    filename='app_security.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_security_event(event_type, details):
    """Log security events"""
    logging.info(f"SECURITY_EVENT: {event_type} - {details}")

def secure_error_handler(func):
    """Decorator for secure error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log error securely without exposing data
            error_id = hashlib.md5(str(e).encode()).hexdigest()[:8]
            st.error(f"Processing error occurred. Error ID: {error_id}")
            # Log to secure location with full details
            logging.error(f"Error {error_id}: {str(e)}")
            return None
    return wrapper

def secure_file_upload(uploaded_file):
    """Secure file upload validation - cloud compatible"""
    if uploaded_file is None:
        return None
    
    # File size limit (50MB)
    MAX_FILE_SIZE = 50 * 1024 * 1024
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("File too large. Maximum size: 50MB")
        log_security_event("FILE_UPLOAD_REJECTED", f"File too large: {uploaded_file.size} bytes")
        return None
    
    # Enhanced extension validation as primary method
    allowed_extensions = ['.csv', '.xlsx', '.xls']
    if not any(uploaded_file.name.lower().endswith(ext) for ext in allowed_extensions):
        st.error("Invalid file extension. Only CSV and Excel files are allowed.")
        log_security_event("FILE_UPLOAD_REJECTED", f"Invalid extension: {uploaded_file.name}")
        return None
    
    # File type validation using magic if available, otherwise skip
    if MAGIC_AVAILABLE:
        try:
            allowed_types = ['text/csv', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            'application/vnd.ms-excel']
            file_content = uploaded_file.read(1024)
            uploaded_file.seek(0)  # Reset file pointer
            
            file_type = magic.from_buffer(file_content, mime=True)
            if file_type not in allowed_types:
                st.error(f"Invalid file type detected: {file_type}")
                log_security_event("FILE_UPLOAD_REJECTED", f"Invalid file type: {file_type}")
                return None
        except Exception as e:
            logging.warning(f"File type validation failed: {str(e)}")
    
    # File content validation
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file, nrows=1)  # Test read
        else:
            df = pd.read_excel(uploaded_file, nrows=1)  # Test read
        uploaded_file.seek(0)  # Reset for actual processing
        
        log_security_event("FILE_UPLOAD_SUCCESS", f"File: {uploaded_file.name}, Size: {uploaded_file.size}")
        return uploaded_file
        
    except Exception as e:
        st.error(f"File appears corrupted or invalid format: {str(e)}")
        log_security_event("FILE_UPLOAD_REJECTED", f"Corrupted file: {uploaded_file.name}")
        return None

def sanitize_data(df):
    """Sanitize dataframe to prevent script injections"""
    try:
        # Remove any potential script injections from string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            df[col] = df[col].astype(str).str.replace(r'[<>"\']', '', regex=True)
            df[col] = df[col].str.replace(r'script|javascript|vbscript', '', regex=True, flags=re.IGNORECASE)
        
        # Remove empty rows and invalid data
        df = df.dropna(how='all')
        return df
    except Exception as e:
        logging.error(f"Data sanitization error: {str(e)}")
        return df

def check_session_timeout():
    """Check and enforce session timeout"""
    if 'session_start' not in st.session_state:
        st.session_state.session_start = time.time()
    
    # Auto-clear data after 1 hour (3600 seconds)
    SESSION_TIMEOUT = 3600
    if time.time() - st.session_state.session_start > SESSION_TIMEOUT:
        cleanup_memory()
        st.warning("Session expired for security. Please refresh the page.")
        st.stop()

def cleanup_memory():
    """Clean up memory and session state"""
    keys_to_remove = []
    for key in st.session_state.keys():
        if key not in ['session_start', 'authenticated']:  # Keep essential keys
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del st.session_state[key]
    
    gc.collect()
    log_security_event("MEMORY_CLEANUP", "Session data cleared")

def check_authentication():
    """Optional authentication for cloud deployment"""
    app_username = st.secrets.get("APP_USERNAME", None) if hasattr(st, 'secrets') else os.getenv('APP_USERNAME')
    app_password = st.secrets.get("APP_PASSWORD", None) if hasattr(st, 'secrets') else os.getenv('APP_PASSWORD')
    
    if not app_username or not app_password:
        return True  # Skip authentication if not configured
    
    def authenticate(username, password):
        return (username == app_username and password == app_password)
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("🔐 Login Required")
        st.markdown("Please enter your credentials to access the dashboard.")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                if authenticate(username, password):
                    st.session_state.authenticated = True
                    log_security_event("LOGIN_SUCCESS", f"User: {username}")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
                    log_security_event("LOGIN_FAILED", f"Failed attempt for user: {username}")
        return False
    return True

# Cache function for file processing with security
@st.cache_data
@secure_error_handler
def process_uploaded_file(file_content, file_name, file_type):
    """Cache file processing to avoid reprocessing on reruns"""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
        else:
            df = pd.read_excel(io.BytesIO(file_content))
        
        # Sanitize the data
        df = sanitize_data(df)
        return df, None
    except Exception as e:
        return None, str(e)

# Data validation function
@secure_error_handler
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
        log_security_event("DATA_VALIDATION_FAILED", f"Missing columns in {file_type}: {missing_cols}")
        return False
    return True

# Memory usage display function
def display_memory_usage():
    """Display current memory usage"""
    if PSUTIL_AVAILABLE:
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            st.sidebar.text(f"💾 Memory: {memory.percent:.1f}%")
            st.sidebar.text(f"🖥️ CPU: {cpu:.1f}%")
            
            # Alert if memory usage is high
            if memory.percent > 80:
                st.sidebar.warning("⚠️ High memory usage!")
        except Exception as e:
            st.sidebar.text("💾 System monitoring unavailable")
            logging.error(f"Memory monitoring error: {str(e)}")
    else:
         # Show a simple message instead of "unavailable"
        st.sidebar.text("💾 Running on cloud platform")

class POSTransactionAnalyzer:
    """
    A comprehensive analyzer for POS terminal transactions from Interswitch and Arca settlement reports.
    Enhanced with security features.
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
    
    @secure_error_handler
    def clean_terminal_merchant_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean terminal and merchant IDs by removing trailing spaces."""
        if 'terminal_id' in df.columns:
            df['terminal_id'] = df['terminal_id'].astype(str).str.strip()
        if 'merchant_id' in df.columns:
            df['merchant_id'] = df['merchant_id'].astype(str).str.strip()
        return df
    
    @secure_error_handler
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
    
    @secure_error_handler
    def load_and_process_dataframe(self, df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        """Process a pandas DataFrame directly (for Streamlit file uploads)."""
        try:
            # Data size check with warning
            if len(df) > 100000:  # Adjust limit as needed
                st.warning(f"Large file detected ({len(df):,} rows). Processing may take time.")
                log_security_event("LARGE_FILE_PROCESSING", f"File size: {len(df)} rows")
            
            # Check if DataFrame is empty
            if df.empty:
                raise ValueError(f"The {file_type} file appears to be empty")
            
            # Sanitize data first
            df = sanitize_data(df)
            
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
            
            # Convert amount to numeric and validate
            if 'amount' in df_processed.columns:
                df_processed['amount'] = pd.to_numeric(df_processed['amount'], errors='coerce').fillna(0)
                # Basic sanity check for amounts
                if df_processed['amount'].max() > 10_000_000:  # 10M limit
                    st.warning("Unusually large transaction amounts detected. Please verify data.")
            
            # Add source column
            df_processed['source'] = file_type.lower()
            
            log_security_event("DATA_PROCESSING_SUCCESS", f"Processed {file_type} file: {len(df_processed)} rows")
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
    
    @secure_error_handler
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
                log_security_event("DATE_FILTER_APPLIED", f"Range: {start_date} to {end_date}")
            
            status_text.text('🏪 Processing terminals...')
            progress_bar.progress(60)
            
            # Get unique dates for dynamic column creation
            unique_dates = sorted(combined_df['date'].unique()) if 'date' in combined_df.columns else []
            
            # Get all unique terminal IDs
            all_terminals = pd.concat([
                arca_df['terminal_id'], 
                interswitch_df['terminal_id']
            ]).unique()
            
            # Limit processing for security (prevent DoS)
            MAX_TERMINALS = 10000
            if len(all_terminals) > MAX_TERMINALS:
                st.warning(f"Large dataset detected ({len(all_terminals)} terminals). Processing first {MAX_TERMINALS} terminals.")
                all_terminals = all_terminals[:MAX_TERMINALS]
            
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
            log_security_event("ANALYSIS_COMPLETED", f"Processed {len(results_df)} terminals")
            return results_df
            
        except Exception as e:
            log_security_event("ANALYSIS_ERROR", str(e))
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
    
    @secure_error_handler
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

# Custom CSS for better styling (same as before)
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
    
    /* Security indicator */
    .security-indicator {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
        text-align: center;
        font-size: 0.9rem;
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
    
    /* Rest of the CSS remains the same... */
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = POSTransactionAnalyzer()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def main():
    # Check session timeout first
    check_session_timeout()
    
    # Check authentication if configured
    if not check_authentication():
        return
    
    # Security indicator
    st.markdown("""
    <div class="security-indicator">
        🔒 Secure Session - Data processed locally and not stored permanently
    </div>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">💳 POS Transaction Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for file uploads and controls
    st.sidebar.title("📁 Data Upload & Controls")
    
    # Display memory usage
    display_memory_usage()
    
    # Add session management controls
    st.sidebar.markdown("### 🛡️ Session Management")
    if st.sidebar.button("🧹 Clear Session Data"):
        cleanup_memory()
        st.success("Session data cleared for security")
        st.rerun()
    
    # Show session timeout info
    if 'session_start' in st.session_state:
        session_duration = int(time.time() - st.session_state.session_start)
        st.sidebar.text(f"⏱️ Session: {session_duration//60}m {session_duration%60}s")
    
    st.sidebar.markdown("---")
    
    # File upload section
    st.sidebar.markdown("### Upload Settlement Reports")
    st.sidebar.markdown("📋 **Security Features:**")
    st.sidebar.markdown("• File size limit: 50MB")
    st.sidebar.markdown("• File type validation")
    st.sidebar.markdown("• Content sanitization")
    st.sidebar.markdown("• Session timeout: 1 hour")
    
    # Arca files
    st.sidebar.markdown("**Arca Reports:**")
    arca_uploaded_files = st.sidebar.file_uploader(
        "Upload Arca settlement reports", 
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        key="arca_files"
    )
    
    # Secure file processing for Arca
    arca_files = []
    if arca_uploaded_files:
        for file in arca_uploaded_files:
            secure_file = secure_file_upload(file)
            if secure_file:
                arca_files.append(secure_file)
    
    # Interswitch files  
    st.sidebar.markdown("**Interswitch Reports:**")
    interswitch_uploaded_files = st.sidebar.file_uploader(
        "Upload Interswitch settlement reports",
        type=['csv', 'xlsx', 'xls'], 
        accept_multiple_files=True,
        key="interswitch_files"
    )
    
    # Secure file processing for Interswitch
    interswitch_files = []
    if interswitch_uploaded_files:
        for file in interswitch_uploaded_files:
            secure_file = secure_file_upload(file)
            if secure_file:
                interswitch_files.append(secure_file)
    
    # Date range filter
    st.sidebar.markdown("### 📅 Date Range Filter (Optional)")
    use_date_filter = st.sidebar.checkbox("Apply date range filter")
    
    date_range = None
    if use_date_filter:
        start_date = st.sidebar.date_input("Start Date")
        end_date = st.sidebar.date_input("End Date")
        if start_date and end_date:
            if end_date < start_date:
                st.sidebar.error("End date must be after start date")
            else:
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
                    if not processed_df.empty:
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
                    if not processed_df.empty:
                        interswitch_dfs.append(processed_df)
                
                # Combine dataframes
                arca_combined = pd.concat(arca_dfs, ignore_index=True) if arca_dfs else pd.DataFrame()
                interswitch_combined = pd.concat(interswitch_dfs, ignore_index=True) if interswitch_dfs else pd.DataFrame()
                
                if arca_combined.empty and interswitch_combined.empty:
                    st.error("No valid data found in uploaded files")
                    return
                
                # Run analysis
                results = st.session_state.analyzer.analyze_transactions(
                    arca_combined, interswitch_combined, date_range
                )
                
                st.session_state.analysis_results = results
                
                st.success("✅ Analysis completed successfully!")
                
            except Exception as e:
                error_id = hashlib.md5(str(e).encode()).hexdigest()[:8]
                st.error(f"❌ Error during analysis. Error ID: {error_id}")
                log_security_event("ANALYSIS_ERROR", f"Error ID: {error_id} - {str(e)}")
                return
    
    # Display results if available
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
        
        ### Security Features:
        - 🔒 Secure file upload validation
        - 🛡️ Data sanitization against injection attacks
        - ⏱️ Automatic session timeout (1 hour)
        - 🧹 Manual session cleanup option
        - 📝 Comprehensive security logging
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