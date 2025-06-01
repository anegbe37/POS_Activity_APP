import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

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
            print(f"Warning: Date parsing issue - {str(e)}")
            return df
    
    def load_and_process_file(self, file_path: str, file_type: str) -> pd.DataFrame:
        """Load and process either Arca or Interswitch file."""
        try:
            # Read file based on extension
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel files.")
            
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
                print(f"Warning: Missing columns in {file_type} file: {missing_cols}")
                # Use available columns only
                available_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
                column_mapping = available_mapping
            
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
            
        except Exception as e:
            raise Exception(f"Error processing {file_type} file: {str(e)}")
    
    def analyze_transactions(self, arca_df: pd.DataFrame, interswitch_df: pd.DataFrame, 
                           date_range: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze transactions and create the required summary report.
        """
        try:
            # Combine dataframes
            combined_df = pd.concat([arca_df, interswitch_df], ignore_index=True)
            
            # Filter by date range if provided
            if date_range:
                start_date = pd.to_datetime(date_range[0]).date()
                end_date = pd.to_datetime(date_range[1]).date()
                combined_df = combined_df[
                    (combined_df['date'] >= start_date) & 
                    (combined_df['date'] <= end_date)
                ]
            
            # Get unique dates for dynamic column creation
            unique_dates = sorted(combined_df['date'].unique()) if 'date' in combined_df.columns else []
            
            # Get all unique terminal IDs
            all_terminals = pd.concat([
                arca_df['terminal_id'], 
                interswitch_df['terminal_id']
            ]).unique()
            
            results = []
            
            for terminal_id in all_terminals:
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
            
            # Create DataFrame
            results_df = pd.DataFrame(results)
            
            # Sort by TID
            results_df = results_df.sort_values('TID').reset_index(drop=True)
            
            self.analysis_results = results_df
            return results_df
            
        except Exception as e:
            raise Exception(f"Error during analysis: {str(e)}")
    
    def export_results(self, results_df: pd.DataFrame, output_dir: str = "output", 
                      filename_prefix: str = "pos_analysis") -> Dict[str, str]:
        """Export results to both CSV and Excel formats."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export to CSV
            csv_filename = f"{filename_prefix}_{timestamp}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            results_df.to_csv(csv_path, index=False)
            
            # Export to Excel with formatting
            excel_filename = f"{filename_prefix}_{timestamp}.xlsx"
            excel_path = os.path.join(output_dir, excel_filename)
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Transaction_Analysis', index=False)
                
                # Get workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets['Transaction_Analysis']
                
                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
            return {
                'csv_path': csv_path,
                'excel_path': excel_path,
                'csv_filename': csv_filename,
                'excel_filename': excel_filename
            }
            
        except Exception as e:
            raise Exception(f"Error exporting results: {str(e)}")
    
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
            print(f"Error calculating summary stats: {str(e)}")
            return {}
    
    def process_multiple_files(self, file_configs: List[Dict]) -> pd.DataFrame:
        """
        Process multiple files and combine results.
        file_configs: List of dicts with keys: 'path', 'type' ('arca' or 'interswitch')
        """
        arca_dfs = []
        interswitch_dfs = []
        
        for config in file_configs:
            df = self.load_and_process_file(config['path'], config['type'])
            if config['type'].lower() == 'arca':
                arca_dfs.append(df)
            else:
                interswitch_dfs.append(df)
        
        # Combine all dataframes by type
        arca_combined = pd.concat(arca_dfs, ignore_index=True) if arca_dfs else pd.DataFrame()
        interswitch_combined = pd.concat(interswitch_dfs, ignore_index=True) if interswitch_dfs else pd.DataFrame()
        
        return self.analyze_transactions(arca_combined, interswitch_combined)

# Example usage
if __name__ == "__main__":
    analyzer = POSTransactionAnalyzer()
    
    # Example of how to use the analyzer
    try:
        # Load files
        arca_df = analyzer.load_and_process_file("arca_report.csv", "arca")
        interswitch_df = analyzer.load_and_process_file("interswitch_report.csv", "interswitch")
        
        # Analyze transactions
        results = analyzer.analyze_transactions(arca_df, interswitch_df)
        
        # Export results
        export_info = analyzer.export_results(results)
        print(f"Results exported to: {export_info['csv_path']} and {export_info['excel_path']}")
        
        # Get summary statistics
        stats = analyzer.get_summary_stats(results)
        print(f"Analysis complete. Total terminals: {stats.get('total_terminals', 0)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


        import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import io
import base64
from pos_streamlitapp import POSTransactionAnalyzer  # Import our analyzer class

# Page configuration
st.set_page_config(
    page_title="POS Transaction Analytics Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
    }
    .stMetric > div {
        white-space: nowrap;
    }
    .stMetric label {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }
    .stMetric [data-testid="metric-value"] {
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    .download-section {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = POSTransactionAnalyzer()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'export_info' not in st.session_state:
    st.session_state.export_info = None

def create_download_link(df, filename, file_format='csv'):
    """Create a download link for dataframe."""
    if file_format == 'csv':
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV File</a>'
    elif file_format == 'excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Analysis', index=False)
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">Download Excel File</a>'
    return href

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
                
                # Save and process Arca files
                for arca_file in arca_files:
                    # Read file content
                    if arca_file.name.endswith('.csv'):
                        df = pd.read_csv(arca_file)
                    else:
                        df = pd.read_excel(arca_file)
                    
                    # Process using analyzer
                    temp_path = f"temp_arca_{arca_file.name}"
                    df.to_csv(temp_path, index=False)
                    processed_df = st.session_state.analyzer.load_and_process_file(temp_path, "arca")
                    arca_dfs.append(processed_df)
                
                # Save and process Interswitch files
                for interswitch_file in interswitch_files:
                    # Read file content
                    if interswitch_file.name.endswith('.csv'):
                        df = pd.read_csv(interswitch_file)
                    else:
                        df = pd.read_excel(interswitch_file)
                    
                    # Process using analyzer
                    temp_path = f"temp_interswitch_{interswitch_file.name}"
                    df.to_csv(temp_path, index=False)
                    processed_df = st.session_state.analyzer.load_and_process_file(temp_path, "interswitch")
                    interswitch_dfs.append(processed_df)
                
                # Combine dataframes
                arca_combined = pd.concat(arca_dfs, ignore_index=True) if arca_dfs else pd.DataFrame()
                interswitch_combined = pd.concat(interswitch_dfs, ignore_index=True) if interswitch_dfs else pd.DataFrame()
                
                # Run analysis
                results = st.session_state.analyzer.analyze_transactions(
                    arca_combined, interswitch_combined, date_range
                )
                
                st.session_state.analysis_results = results
                
                # Export results
                export_info = st.session_state.analyzer.export_results(results)
                st.session_state.export_info = export_info
                
                st.success("✅ Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                return
    
    # Display results if available
    if st.session_state.analysis_results is not None:
        results_df = st.session_state.analysis_results
        
        # Summary Statistics with better layout
        st.markdown('<h2 class="section-header">📊 Summary Statistics</h2>', unsafe_allow_html=True)
        
        # Calculate summary stats
        stats = st.session_state.analyzer.get_summary_stats(results_df)
        
        # Create a container for better spacing
        with st.container():
            # Display metrics in columns with better spacing
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1.2])  # Give more space to volume column
            
            with col1:
                st.metric(
                    label="Total Terminals",
                    value=f"{stats.get('total_terminals', 0):,}"
                )
            
            with col2:
                st.metric(
                    label="Total Merchants", 
                    value=f"{stats.get('total_merchants', 0):,}"
                )
            
            with col3:
                st.metric(
                    label="Total Transactions",
                    value=f"{stats.get('total_transactions', 0):,}"
                )
            
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
                
                st.metric(
                    label="Total Volume",
                    value=volume_display
                )
            
            # Add additional metrics row for more details
            st.markdown("---")
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                avg_trans = stats.get('avg_transactions_per_terminal', 0)
                st.metric(
                    label="Avg Trans/Terminal",
                    value=f"{avg_trans:.1f}"
                )
            
            with col6:
                avg_volume = stats.get('avg_volume_per_terminal', 0)
                if avg_volume >= 1_000_000:
                    avg_display = f"₦{avg_volume/1_000_000:.1f}M"
                elif avg_volume >= 1_000:
                    avg_display = f"₦{avg_volume/1_000:.1f}K"
                else:
                    avg_display = f"₦{avg_volume:,.0f}"
                st.metric(
                    label="Avg Volume/Terminal",
                    value=avg_display
                )
            
            with col7:
                # Active terminals (terminals with transactions > 0)
                active_terminals = len(results_df[results_df['Total_Count'] > 0])
                st.metric(
                    label="Active Terminals",
                    value=f"{active_terminals:,}"
                )
            
            with col8:
                # Success rate (assuming any transaction count means success)
                success_rate = (active_terminals / stats.get('total_terminals', 1)) * 100
                st.metric(
                    label="Activity Rate",
                    value=f"{success_rate:.1f}%"
                )
        
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
        
        # File location info
        if st.session_state.export_info:
            st.info(f"""
            📁 **Files also saved to:**
            - CSV: `{st.session_state.export_info['csv_filename']}`
            - Excel: `{st.session_state.export_info['excel_filename']}`
            
            Location: `output/` folder in your project directory
            """)
    
    else:
        # Instructions when no data is loaded
        st.markdown('<h2 class="section-header">🚀 Getting Started</h2>', unsafe_allow_html=True)
        
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
