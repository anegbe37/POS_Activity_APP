import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import io
import base64
from pos_analyzer import POSTransactionAnalyzer  # Import our analyzer class

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