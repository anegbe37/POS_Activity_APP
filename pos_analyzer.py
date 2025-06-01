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