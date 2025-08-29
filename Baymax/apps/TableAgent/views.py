# import os
# import json
# import polars as pl
# import duckdb
# import requests
# from django.shortcuts import render, redirect
# from django.http import JsonResponse
# from django.conf import settings
# from django.core.files.storage import default_storage
# from django.core.files.base import ContentFile
# from django.contrib import messages
# from django.views.decorators.csrf import csrf_exempt
# from django.utils.decorators import method_decorator
# from django.views import View

# # Initialize DuckDB connection
# con = duckdb.connect()

# class UploadFileView(View):
#     """Handle file upload and redirect to ask page"""
#     def get(self, request):
#         return render(request, 'TableAgent/upload.html')
    
#     def post(self, request):
#         if request.FILES.get('data_file'):
#             data_file = request.FILES['data_file']
            
#             # Save file to media directory
#             file_path = default_storage.save(
#                 f"uploads/{data_file.name}", 
#                 ContentFile(data_file.read())
#             )
            
#             # Store file path in session
#             request.session['uploaded_file'] = file_path
#             request.session['file_name'] = data_file.name
            
#             return redirect('ask')
        
#         return render(request, 'TableAgent/upload.html')

# class AskQuestionView(View):
#     """Handle questions about the uploaded data"""
    
#     def get(self, request):
#         file_path = request.session.get('uploaded_file')
#         file_name = request.session.get('file_name')
        
#         if not file_path:
#             messages.error(request, 'Please upload a file first.')
#             return redirect('upload')
        
#         full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        
#         # Load data with Polars
#         try:
#             # Check file extension to determine how to read it
#             if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
#                 df = pl.read_excel(full_path)
#             else:  # Default to CSV
#                 df = pl.read_csv(full_path)
#             # Register dataframe with DuckDB
#             con.register('data_table', df)
#         except Exception as e:
#             messages.error(request, f'Error reading file: {str(e)}')
#             return redirect('upload')
        
#         context = {
#             'file_name': file_name,
#             'columns': df.columns,
#             'row_count': len(df),
#             'sample_data': df.head(5).to_dicts()
#         }
        
#         return render(request, 'TableAgent/ask.html', context)
    
#     def post(self, request):
#         file_path = request.session.get('uploaded_file')
#         if not file_path:
#             return JsonResponse({'status': 'error', 'message': 'No file uploaded'})
        
#         full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        
#         question = request.POST.get('question')
#         if not question:
#             return JsonResponse({'status': 'error', 'message': 'No question provided'})
        
#         try:
#             # Load data with Polars
#             # Check file extension to determine how to read it
#             file_name = request.session.get('file_name', '')
#             if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
#                 df = pl.read_excel(full_path)
#             else:  # Default to CSV
#                 df = pl.read_csv(full_path)
#             # Register dataframe with DuckDB
#             con.register('data_table', df)
            
#             # Convert natural language to SQL using AI
#             sql_query = convert_to_sql(question, df.columns)
            
#             # Validate that we got a SQL query
#             if not sql_query:
#                 return JsonResponse({
#                     'status': 'error', 
#                     'message': 'Failed to generate SQL query from your question. Please try rephrasing.'
#                 })
            
#             # Execute query with DuckDB
#             try:
#                 result_df = con.execute(sql_query).fetchdf()
#             except Exception as e:
#                 return JsonResponse({
#                     'status': 'error', 
#                     'message': f'Error executing query: {str(e)}. Please try rephrasing your question.'
#                 })
            
#             # Format result for display
#             result_data = {
#                 'query': sql_query,
#                 'columns': result_df.columns.tolist(),
#                 'data': result_df.values.tolist(),
#                 'row_count': len(result_df)
#             }
            
#             return JsonResponse({'status': 'success', 'result': result_data})
#         except Exception as e:
#             return JsonResponse({'status': 'error', 'message': str(e)})

# def convert_to_sql(question, columns):
#     """Convert natural language question to SQL using NVIDIA NIM API"""
#     # Create a detailed prompt for the AI
#     prompt = f"""
#     You are a SQL expert. Convert the following natural language question into a valid SQL query.
    
#     Table name: data_table
#     Available columns: {', '.join(columns)}
    
#     Rules:
#     1. Only use the provided column names
#     2. Return ONLY the SQL query, nothing else
#     3. Make sure the query is valid SQL for DuckDB
#     4. If the question asks for all data, limit to 100 rows
#     5. For counting questions, use COUNT(*)
#     6. For average questions, use AVG(column_name)
#     7. Don't include any markdown formatting or backticks
    
#     Question: {question}
    
#     SQL Query:
#     """
    
#     # Use NVIDIA NIM API
#     if settings.NVIDIA_NIM_API_KEY:
#         try:
#             headers = {
#                 'Authorization': f'Bearer {settings.NVIDIA_NIM_API_KEY}',
#                 'Content-Type': 'application/json',
#             }
            
#             data = {
#                 'model': settings.NVIDIA_NIM_MODEL,
#                 'messages': [
#                     {
#                         'role': 'system',
#                         'content': 'You are a helpful SQL assistant that converts natural language to SQL queries.'
#                     },
#                     {
#                         'role': 'user',
#                         'content': prompt
#                     }
#                 ],
#                 'temperature': 0.1,  # Low temperature for consistent SQL output
#                 'max_tokens': 200
#             }
            
#             response = requests.post(
#                 settings.NVIDIA_NIM_API_ENDPOINT,
#                 headers=headers,
#                 json=data,
#                 timeout=30
#             )
            
#             if response.status_code == 200:
#                 result = response.json()
#                 sql_query = result['choices'][0]['message']['content'].strip()
                
#                 # Clean up the response - remove any extra text or markdown
#                 lines = sql_query.split('\n')
#                 for line in lines:
#                     line = line.strip()
#                     # Remove markdown code block markers if present
#                     if line.startswith('```') or line.startswith('`'):
#                         continue
#                     # If line starts with SELECT, that's our query
#                     if line.upper().startswith('SELECT'):
#                         return line
                
#                 # If we didn't find a SELECT statement, return the first line that looks like SQL
#                 for line in lines:
#                     line = line.strip()
#                     if line.upper().startswith('SELECT'):
#                         return line
                
#                 # Fallback to first line if no SELECT found
#                 first_line = lines[0].strip() if lines else ""
#                 # Check if first line is a valid SQL query
#                 if first_line.upper().startswith('SELECT'):
#                     return first_line
#                 else:
#                     # If all else fails, return a default query
#                     return "SELECT * FROM data_table LIMIT 5"
#             else:
#                 print(f"NVIDIA NIM API error: {response.status_code} - {response.text}")
#                 return _fallback_sql_conversion(question, columns)
#         except Exception as e:
#             print(f"Error calling NVIDIA NIM API: {e}")
#             # Fallback to simple rules if API call fails
#             return _fallback_sql_conversion(question, columns)
#     else:
#         # If no API key, use fallback method
#         return _fallback_sql_conversion(question, columns)

# def _fallback_sql_conversion(question, columns):
#     """Fallback method for SQL conversion when API is unavailable"""
#     question_lower = question.lower()
    
#     if 'top' in question_lower and 'rows' in question_lower:
#         return "SELECT * FROM data_table LIMIT 5"
#     elif 'average' in question_lower or 'mean' in question_lower:
#         # Try to find a numeric column
#         return "SELECT * FROM data_table LIMIT 1"  # Simplified
#     elif 'count' in question_lower:
#         return "SELECT COUNT(*) as count FROM data_table"
#     else:
#         # Default query
#         return "SELECT * FROM data_table LIMIT 5"

# def get_table_info(request):
#     """Return basic information about the uploaded table"""
#     file_path = request.session.get('uploaded_file')
#     if not file_path:
#         return JsonResponse({'status': 'error', 'message': 'No file uploaded'})
    
#     full_path = os.path.join(settings.MEDIA_ROOT, file_path)
    
#     try:
#         file_name = request.session.get('file_name', '')
#         if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
#             df = pl.read_excel(full_path)
#         else:  # Default to CSV
#             df = pl.read_csv(full_path)
            
#         info = {
#             'columns': df.columns,
#             'data_types': [str(dtype) for dtype in df.dtypes],
#             'row_count': len(df),
#             'sample_data': df.head(3).to_dicts()
#         }
#         return JsonResponse({'status': 'success', 'info': info})
#     except Exception as e:
#         return JsonResponse({'status': 'error', 'message': str(e)})

import os
import json
import polars as pl
import pandas as pd
import duckdb
import requests
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from django.core.cache import cache
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import math

# Setup logging
logger = logging.getLogger(__name__)

# Initialize DuckDB connection
con = duckdb.connect()

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=3)

def safe_float(value):
    """Convert value to float, handling NaN and infinity"""
    if value is None or pd.isna(value) or math.isnan(value) or math.isinf(value):
        return None
    try:
        return float(value)
    except:
        return None

def sanitize_for_json(obj):
    """Recursively sanitize an object for JSON serialization"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif pd.isna(obj):
        return None
    else:
        return obj

@dataclass
class DataContext:
    """Enhanced data context with business understanding"""
    file_path: str
    columns: List[str]
    dtypes: List[str]
    row_count: int
    domain_type: str  # e.g., 'support_tickets', 'sales_data', 'inventory'
    key_entities: Dict[str, List[str]]  # e.g., {'products': ['laptop', 'printer'], 'categories': ['hardware', 'software']}
    text_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    id_columns: List[str]
    date_columns: List[str]
    content_samples: Dict[str, List[Any]]
    business_context: str  # Human-readable description of what this data represents
    column_semantics: Dict[str, str]  # What each column likely represents in business terms
    column_value_patterns: Dict[str, str]  # Pattern of values in each column (e.g., "ABC####")

class EnhancedDataUnderstanding:
    """Intelligent data context builder"""
    
    def analyze_data(self, df: pd.DataFrame, file_name: str) -> DataContext:
        """Build comprehensive understanding of the data"""
        
        # Basic structure
        columns = df.columns.tolist()
        dtypes = [str(df[col].dtype) for col in columns]
        row_count = len(df)
        
        # Categorize columns
        text_columns = self._identify_text_columns(df)
        numeric_columns = self._identify_numeric_columns(df)
        categorical_columns = self._identify_categorical_columns(df)
        id_columns = self._identify_id_columns(df)
        date_columns = self._identify_date_columns(df)
        
        # Extract domain understanding
        domain_type = self._detect_domain_type(df, file_name)
        key_entities = self._extract_key_entities(df, text_columns + categorical_columns)
        content_samples = self._get_content_samples(df)
        column_semantics = self._understand_column_semantics(df)
        business_context = self._generate_business_context(df, domain_type, key_entities)
        column_value_patterns = self._extract_value_patterns(df)
        
        return DataContext(
            file_path=file_name,
            columns=columns,
            dtypes=dtypes,
            row_count=row_count,
            domain_type=domain_type,
            key_entities=key_entities,
            text_columns=text_columns,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            id_columns=id_columns,
            date_columns=date_columns,
            content_samples=content_samples,
            business_context=business_context,
            column_semantics=column_semantics,
            column_value_patterns=column_value_patterns
        )
    
    def _extract_value_patterns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Extract patterns of values in each column"""
        patterns = {}
        
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Get sample values
                    sample_values = df[col].dropna().astype(str).head(100).unique()
                    
                    # Look for common patterns
                    if len(sample_values) > 0:
                        # Check for alphanumeric patterns
                        if all(re.match(r'^[A-Z]+\d+$', val) for val in sample_values[:10]):
                            patterns[col] = "LETTERS+NUMBERS (e.g., ABC123)"
                        elif all(re.match(r'^[A-Z]{3}\d{4}$', val) for val in sample_values[:10]):
                            patterns[col] = "XXX#### pattern"
                        elif all(re.match(r'^\d+$', val) for val in sample_values[:10]):
                            patterns[col] = "Numeric string"
                        elif all(re.match(r'^[A-Z]{2,4}$', val) for val in sample_values[:10]):
                            patterns[col] = "Short codes (2-4 letters)"
                        else:
                            # General text
                            avg_len = np.mean([len(val) for val in sample_values])
                            if avg_len < 10:
                                patterns[col] = "Short text/codes"
                            else:
                                patterns[col] = "Variable text"
                except:
                    patterns[col] = "Unknown pattern"
            else:
                patterns[col] = "Numeric"
        
        return patterns
    
    def _identify_text_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify columns containing substantial text"""
        text_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check average string length
                try:
                    avg_length = df[col].astype(str).str.len().mean()
                    if avg_length > 30:  # Likely contains text, not just short codes
                        text_cols.append(col)
                except:
                    pass
        return text_cols
    
    def _identify_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify numeric columns"""
        return df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    
    def _identify_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify categorical columns (limited unique values)"""
        categorical_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.1 and df[col].nunique() < 100:  # Less than 10% unique values
                        categorical_cols.append(col)
                except:
                    pass
        return categorical_cols
    
    def _identify_id_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify ID/identifier columns"""
        id_cols = []
        id_patterns = ['id', 'code', 'number', 'key', 'identifier', '_id', 'num']
        
        for col in df.columns:
            col_lower = col.lower()
            # Check column name
            if any(pattern in col_lower for pattern in id_patterns):
                # Verify it's likely an ID (high uniqueness or sequential)
                try:
                    if df[col].nunique() > len(df) * 0.8:
                        id_cols.append(col)
                    elif df[col].dtype in ['int64', 'int32']:
                        # Check if sequential
                        if len(df) > 1:
                            sorted_vals = df[col].dropna().sort_values()
                            if len(sorted_vals) > 1:
                                diffs = sorted_vals.diff().dropna()
                                if diffs.std() < diffs.mean() * 0.1:  # Roughly sequential
                                    id_cols.append(col)
                except:
                    pass
        return id_cols
    
    def _identify_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify date/time columns"""
        date_cols = []
        date_patterns = ['date', 'time', 'created', 'updated', 'modified', '_at', 'when']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in date_patterns):
                date_cols.append(col)
            elif df[col].dtype == 'object':
                # Try to parse as date
                try:
                    pd.to_datetime(df[col].dropna().head(10))
                    date_cols.append(col)
                except:
                    pass
        return date_cols
    
    def _detect_domain_type(self, df: pd.DataFrame, file_name: str) -> str:
        """Detect what type of business data this represents"""
        
        # Check file name hints
        file_lower = file_name.lower()
        if any(term in file_lower for term in ['ticket', 'support', 'issue', 'problem']):
            return 'support_tickets'
        elif any(term in file_lower for term in ['sales', 'order', 'transaction', 'purchase']):
            return 'sales_data'
        elif any(term in file_lower for term in ['inventory', 'stock', 'product', 'item']):
            return 'inventory'
        elif any(term in file_lower for term in ['customer', 'client', 'user', 'member']):
            return 'customer_data'
        
        # Check column names
        col_names_lower = [col.lower() for col in df.columns]
        
        # Support ticket indicators
        if any(all(term in col_names_lower for term in pattern) for pattern in [
            ['ticket', 'problem'], ['issue', 'description'], ['support', 'status']
        ]):
            return 'support_tickets'
        
        # Sales indicators
        if any(all(term in ' '.join(col_names_lower) for term in pattern) for pattern in [
            ['price', 'quantity'], ['order', 'amount'], ['sale', 'revenue']
        ]):
            return 'sales_data'
        
        # Check content patterns
        for col in df.columns[:5]:  # Check first 5 columns
            if df[col].dtype == 'object':
                try:
                    sample_values = df[col].dropna().astype(str).head(100)
                    content = ' '.join(sample_values).lower()
                    
                    if any(term in content for term in ['hardware', 'software', 'printer', 'computer', 'laptop']):
                        return 'it_equipment'
                    elif any(term in content for term in ['customer', 'purchase', 'order', 'invoice']):
                        return 'sales_data'
                except:
                    pass
        
        return 'general_data'
    
    def _extract_key_entities(self, df: pd.DataFrame, text_columns: List[str]) -> Dict[str, List[str]]:
        """Extract key business entities from the data"""
        entities = {}
        
        # Common entity patterns
        entity_patterns = {
            'products': ['laptop', 'computer', 'printer', 'scanner', 'monitor', 'keyboard', 'mouse', 
                        'software', 'hardware', 'server', 'router', 'phone', 'tablet'],
            'categories': ['hardware', 'software', 'networking', 'peripheral', 'accessory', 'service'],
            'statuses': ['open', 'closed', 'pending', 'resolved', 'active', 'inactive', 'completed'],
            'priorities': ['high', 'medium', 'low', 'urgent', 'critical', 'normal'],
            'departments': ['it', 'hr', 'sales', 'marketing', 'finance', 'operations', 'support']
        }
        
        # Search for entities in categorical columns
        for col in text_columns[:10]:  # Limit to first 10 columns for performance
            if col in df.columns:
                try:
                    # Get unique values
                    unique_vals = df[col].dropna().astype(str).unique()
                    if len(unique_vals) < 1000:  # Only process if reasonable number
                        values_lower = [val.lower() for val in unique_vals]
                        
                        for entity_type, patterns in entity_patterns.items():
                            found = []
                            for pattern in patterns:
                                for val in values_lower:
                                    if pattern in val and val not in found:
                                        found.append(val)
                            
                            if found:
                                if entity_type not in entities:
                                    entities[entity_type] = []
                                entities[entity_type].extend(found[:20])  # Limit to 20 per type
                except Exception as e:
                    logger.debug(f"Error extracting entities from {col}: {e}")
        
        # Deduplicate
        for entity_type in entities:
            entities[entity_type] = list(set(entities[entity_type]))[:10]
        
        return entities
    
    def _get_content_samples(self, df: pd.DataFrame) -> Dict[str, List[Any]]:
        """Get representative samples from each column"""
        samples = {}
        
        for col in df.columns:
            try:
                if df[col].dtype == 'object':
                    # Get diverse samples for text columns
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) > 10:
                        # Get a mix of common and unique values
                        value_counts = df[col].value_counts()
                        common = value_counts.head(5).index.tolist()
                        random_sample = df[col].dropna().sample(min(5, len(df))).tolist()
                        samples[col] = list(set(common + random_sample))[:10]
                    else:
                        samples[col] = unique_vals.tolist()[:10]
                else:
                    # Get range for numeric columns with safe float conversion
                    samples[col] = {
                        'min': safe_float(df[col].min()),
                        'max': safe_float(df[col].max()),
                        'mean': safe_float(df[col].mean()),
                        'sample_values': [safe_float(v) for v in df[col].dropna().sample(min(5, len(df))).tolist()]
                    }
            except Exception as e:
                samples[col] = []
                logger.debug(f"Error sampling {col}: {e}")
        
        return samples
    
    def _understand_column_semantics(self, df: pd.DataFrame) -> Dict[str, str]:
        """Understand what each column represents in business terms"""
        semantics = {}
        
        for col in df.columns:
            col_lower = col.lower()
            
            # ID columns
            if any(term in col_lower for term in ['id', 'code', 'number', 'key']):
                semantics[col] = "Unique identifier"
            
            # Description/text columns
            elif any(term in col_lower for term in ['description', 'desc', 'text', 'comment', 'note']):
                semantics[col] = "Detailed text description"
            
            # Status columns
            elif any(term in col_lower for term in ['status', 'state', 'condition']):
                semantics[col] = "Current status or state"
            
            # Category columns
            elif any(term in col_lower for term in ['category', 'type', 'class', 'group']):
                semantics[col] = "Classification or grouping"
            
            # Value columns
            elif any(term in col_lower for term in ['price', 'cost', 'amount', 'value', 'total']):
                semantics[col] = "Monetary value or amount"
            
            # Quantity columns
            elif any(term in col_lower for term in ['quantity', 'count', 'number', 'qty']):
                semantics[col] = "Quantity or count"
            
            # Date columns
            elif any(term in col_lower for term in ['date', 'time', 'created', 'updated']):
                semantics[col] = "Date/time information"
            
            # Name columns
            elif any(term in col_lower for term in ['name', 'title', 'label']):
                semantics[col] = "Name or title"
            
            # Location columns
            elif any(term in col_lower for term in ['location', 'address', 'city', 'country', 'region']):
                semantics[col] = "Geographic location"
            
            else:
                # Try to infer from content
                if df[col].dtype == 'object':
                    try:
                        avg_length = df[col].astype(str).str.len().mean()
                        if avg_length > 50:
                            semantics[col] = "Long text field"
                        elif df[col].nunique() < 10:
                            semantics[col] = "Category with few options"
                        else:
                            semantics[col] = "Text field"
                    except:
                        semantics[col] = "Text field"
                else:
                    semantics[col] = "Numeric value"
        
        return semantics
    
    def _generate_business_context(self, df: pd.DataFrame, domain_type: str, 
                                  key_entities: Dict[str, List[str]]) -> str:
        """Generate human-readable description of what this data represents"""
        
        context_parts = []
        
        # Domain description
        domain_descriptions = {
            'support_tickets': "IT support ticket system tracking technical issues and resolutions",
            'sales_data': "Sales transaction records showing customer purchases and revenue",
            'inventory': "Inventory management data tracking products and stock levels",
            'customer_data': "Customer information and relationship management data",
            'it_equipment': "IT equipment and asset management records",
            'general_data': "Business operational data"
        }
        
        context_parts.append(f"This appears to be {domain_descriptions.get(domain_type, 'business data')}")
        
        # Add entity context
        if 'products' in key_entities and key_entities['products']:
            products = ', '.join(key_entities['products'][:5])
            context_parts.append(f"dealing with products like {products}")
        
        if 'categories' in key_entities and key_entities['categories']:
            categories = ', '.join(key_entities['categories'][:3])
            context_parts.append(f"organized into categories such as {categories}")
        
        # Add scale context
        context_parts.append(f"containing {len(df):,} records")
        
        return ". ".join(context_parts) + "."

class IntelligentSQLGenerator:
    """Context-aware SQL generation with smart fallbacks"""
    
    def __init__(self):
        self.understanding = EnhancedDataUnderstanding()
    
    def _identify_search_terms(self, question: str, data_context: DataContext) -> Dict[str, str]:
        """Identify terms in the question that are values (not columns)"""
        search_terms = {}
        question_words = question.split()
        
        # Extract potential search values
        potential_values = []
        
        # Look for quoted strings
        quoted = re.findall(r'"([^"]*)"', question)
        potential_values.extend(quoted)
        
        # Look for alphanumeric codes (like SIH1524)
        alphanumeric_pattern = re.findall(r'\b[A-Z]+\d+\b', question, re.IGNORECASE)
        potential_values.extend(alphanumeric_pattern)
        
        # Look for codes with specific patterns
        code_patterns = [
            r'\b[A-Z]{2,4}\d{3,6}\b',  # AB1234, ABC12345
            r'\b\d{4,}\b',              # Pure numbers 4+ digits
            r'\b[A-Z]{3}-\d{3}\b',      # ABC-123
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            potential_values.extend(matches)
        
        # Check each potential value against column names
        column_names_lower = [col.lower() for col in data_context.columns]
        
        for value in potential_values:
            value_lower = value.lower()
            # If it's NOT a column name, it's a search value
            if value_lower not in column_names_lower:
                # Try to determine which column it might belong to
                best_col = self._find_best_column_for_value(value, data_context)
                if best_col:
                    search_terms[value] = best_col
        
        return search_terms
    
    def _find_best_column_for_value(self, value: str, data_context: DataContext) -> Optional[str]:
        """Find the most likely column that contains this value"""
        
        # Check ID columns first for alphanumeric codes
        if re.match(r'^[A-Z]+\d+$', value, re.IGNORECASE):
            for col in data_context.id_columns:
                if col in data_context.content_samples:
                    samples = data_context.content_samples[col]
                    if isinstance(samples, list):
                        # Check if any sample matches the pattern
                        for sample in samples:
                            if re.match(r'^[A-Z]+\d+$', str(sample), re.IGNORECASE):
                                return col
        
        # Check categorical columns
        for col in data_context.categorical_columns:
            if col in data_context.content_samples:
                samples = data_context.content_samples[col]
                if isinstance(samples, list):
                    # Case-insensitive check
                    samples_lower = [str(s).lower() for s in samples]
                    if value.lower() in samples_lower:
                        return col
        
        # Check text columns for partial matches
        for col in data_context.text_columns[:3]:  # Limit to first 3
            if col in data_context.content_samples:
                samples = data_context.content_samples[col]
                if isinstance(samples, list):
                    for sample in samples:
                        if value.lower() in str(sample).lower():
                            return col
        
        # Default to first text or categorical column
        if data_context.text_columns:
            return data_context.text_columns[0]
        elif data_context.categorical_columns:
            return data_context.categorical_columns[0]
        
        return None
    
    async def generate_sql(self, question: str, data_context: DataContext, 
                          df_sample: pd.DataFrame) -> Tuple[str, str]:
        """
        Generate SQL with context awareness
        Returns: (sql_query, generation_method)
        """
        
        # First, identify search terms that are values, not columns
        search_terms = self._identify_search_terms(question, data_context)
        
        # Try AI generation first with rich context
        try:
            sql = await self._ai_generate_sql(question, data_context, df_sample, search_terms)
            if sql and self._validate_sql(sql, data_context):
                return sql, "ai_generated"
        except Exception as e:
            logger.warning(f"AI SQL generation failed: {e}")
        
        # Fallback to pattern-based generation
        sql = self._pattern_based_sql(question, data_context, search_terms)
        if sql:
            return sql, "pattern_based"
        
        # Final fallback - show relevant data
        sql = self._context_aware_fallback(question, data_context)
        return sql, "fallback"
    
    async def _ai_generate_sql(self, question: str, data_context: DataContext, 
                              df_sample: pd.DataFrame, search_terms: Dict[str, str]) -> Optional[str]:
        """Generate SQL using AI with rich context"""
        
        api_key = getattr(settings, 'NVIDIA_NIM_API_KEY', None)
        if not api_key:
            return None
        
        # Build comprehensive prompt
        prompt = self._build_ai_prompt(question, data_context, df_sample, search_terms)
        
        api_endpoint = getattr(settings, 'NVIDIA_NIM_API_ENDPOINT', 
                              'https://integrate.api.nvidia.com/v1/chat/completions')
        
        payload = {
            "model": "meta/llama-3.1-70b-instruct",
            "messages": [
                {
                    "role": "system", 
                    "content": """You are a SQL expert who generates DuckDB queries. 
                    Return ONLY valid SQL - no explanations, no markdown, just the query.
                    Always use 'data_table' as the table name.
                    IMPORTANT: Only use column names that are explicitly listed. Any other terms are search values."""
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 300,
            "stream": False
        }
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(api_endpoint, json=payload, headers=headers, timeout=10) as response:
                if response.status == 200:
                    result = await response.json()
                    sql = result['choices'][0]['message']['content'].strip()
                    # Clean the SQL
                    sql = self._clean_ai_sql(sql)
                    return sql
        
        return None
    
    def _build_ai_prompt(self, question: str, data_context: DataContext, 
                        df_sample: pd.DataFrame, search_terms: Dict[str, str]) -> str:
        """Build comprehensive prompt for AI SQL generation"""
        
        prompt_parts = [
            f"Generate SQL for this question: {question}",
            f"\nBusiness Context: {data_context.business_context}",
            f"\nTable: data_table",
            f"Total Rows: {data_context.row_count:,}",
            "\n⚠️ CRITICAL: These are the ONLY valid column names:"
        ]
        
        # List columns with clear formatting
        for col in data_context.columns:
            dtype = data_context.dtypes[data_context.columns.index(col)]
            semantic = data_context.column_semantics.get(col, "")
            pattern = data_context.column_value_patterns.get(col, "")
            
            col_info = f"- {col} ({dtype}): {semantic}"
            if pattern:
                col_info += f" [Value pattern: {pattern}]"
            
            # Add sample values for better context
            if col in data_context.content_samples:
                samples = data_context.content_samples[col]
                if isinstance(samples, dict):  # Numeric column
                    if samples.get('min') is not None and samples.get('max') is not None:
                        col_info += f" [range: {samples['min']:.2f} to {samples['max']:.2f}]"
                else:  # Text column
                    if samples:
                        sample_str = ', '.join([f"'{str(s)[:20]}'" for s in samples[:3]])
                        col_info += f" [examples: {sample_str}]"
            
            prompt_parts.append(col_info)
        
        # Explicitly identify search terms
        if search_terms:
            prompt_parts.append("\n⚠️ IMPORTANT - These are VALUES to search for (NOT column names):")
            for value, suggested_col in search_terms.items():
                prompt_parts.append(f"- '{value}' - search for this value, likely in column '{suggested_col}'")
        
        # Add examples of correct SQL
        prompt_parts.append("\nExamples of CORRECT SQL:")
        prompt_parts.append("- If asked 'Show me SIH1524': SELECT * FROM data_table WHERE problem_code = 'SIH1524'")
        prompt_parts.append("- If asked 'Find ABC123': SELECT * FROM data_table WHERE id_column LIKE '%ABC123%'")
        prompt_parts.append("- DO NOT use: SELECT SIH1524 FROM data_table (SIH1524 is a value, not a column)")
        
        # Add entity context if relevant
        if data_context.key_entities:
            prompt_parts.append("\nKey entities found in data:")
            for entity_type, entities in data_context.key_entities.items():
                if entities:
                    prompt_parts.append(f"- {entity_type}: {', '.join(entities[:5])}")
        
        # Add sample data rows
        if not df_sample.empty:
            prompt_parts.append("\nSample data rows:")
            prompt_parts.append(df_sample.head(3).to_string())
        
        # Add SQL hints based on question type
        question_lower = question.lower()
        
        if any(term in question_lower for term in ['count', 'how many', 'number of']):
            prompt_parts.append("\nHint: Use COUNT(*) and GROUP BY for counting")
        elif any(term in question_lower for term in ['average', 'mean', 'avg']):
            prompt_parts.append("\nHint: Use AVG() for averages")
        elif any(term in question_lower for term in ['longest', 'shortest', 'maximum length']):
            prompt_parts.append("\nHint: Use LENGTH() function and ORDER BY")
        elif any(term in question_lower for term in ['top', 'highest', 'most', 'best']):
            prompt_parts.append("\nHint: Use ORDER BY DESC and LIMIT")
        
        # Final reminder
        prompt_parts.append("\nREMEMBER: Only use the column names listed above. Any term not in that list is a VALUE to search for.")
        prompt_parts.append("Generate SQL that best answers the question. Return ONLY the SQL query.")
        
        return '\n'.join(prompt_parts)
    
    def _clean_ai_sql(self, sql: str) -> Optional[str]:
        """Clean and validate AI-generated SQL"""
        if not sql:
            return None
        
        # Remove markdown
        sql = sql.replace('```sql', '').replace('```', '').strip()
        
        # Remove non-SQL lines
        lines = []
        for line in sql.split('\n'):
            line = line.strip()
            if line and not any(line.lower().startswith(word) for word in 
                              ['explanation', 'note', 'this', 'the', 'here', 'query']):
                lines.append(line)
        
        sql = ' '.join(lines)
        
        # Ensure it's a SELECT statement
        if not sql.upper().startswith('SELECT'):
            return None
        
        return sql
    
    def _validate_sql(self, sql: str, data_context: DataContext) -> bool:
        """Validate SQL query against known schema"""
        if not sql:
            return False
        
        sql_upper = sql.upper()
        
        # Basic structure validation
        if not sql_upper.startswith('SELECT'):
            return False
        
        if 'FROM' not in sql_upper:
            return False
        
        # Extract all identifiers from SQL
        # This is a simple check - looks for words that might be column references
        sql_words = re.findall(r'\b\w+\b', sql)
        
        # Check for invalid column references
        valid_columns_upper = [col.upper() for col in data_context.columns]
        
        # Skip SQL keywords
        sql_keywords = {'SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'ORDER', 'LIMIT', 
                       'AS', 'COUNT', 'AVG', 'SUM', 'MAX', 'MIN', 'AND', 'OR', 
                       'LIKE', 'IN', 'NOT', 'NULL', 'IS', 'DESC', 'ASC', 'DISTINCT',
                       'LENGTH', 'LOWER', 'UPPER', 'data_table', 'DATA_TABLE'}
        
        # Look for potential column references
        for word in sql_words:
            word_upper = word.upper()
            # If it looks like it might be a column reference (not a keyword, not a string literal)
            if (word_upper not in sql_keywords and 
                not word.startswith("'") and 
                not word.isdigit() and
                len(word) > 1):
                # Check if it's a valid column
                if word_upper not in valid_columns_upper and word != '*':
                    # Might be an invalid column reference
                    logger.warning(f"Potential invalid column reference: {word}")
        
        return True
    
    def _pattern_based_sql(self, question: str, data_context: DataContext, 
                          search_terms: Dict[str, str]) -> Optional[str]:
        """Generate SQL using pattern matching with context awareness"""
        
        question_lower = question.lower()
        
        # If we have identified search terms, prioritize search queries
        if search_terms:
            return self._generate_search_query(search_terms, question_lower, data_context)
        
        # Entity-aware patterns
        for entity_type, entities in data_context.key_entities.items():
            for entity in entities:
                if entity.lower() in question_lower:
                    return self._generate_entity_query(question_lower, entity, entity_type, data_context)
        
        # Check for potential value searches (alphanumeric codes)
        alphanumeric = re.findall(r'\b[A-Z]+\d+\b', question, re.IGNORECASE)
        if alphanumeric:
            # Find best column for these values
            for value in alphanumeric:
                best_col = self._find_best_column_for_value(value, data_context)
                if best_col:
                    return f"SELECT * FROM data_table WHERE {best_col} = '{value}'"
        
        # Question type patterns
        patterns = {
            'count': self._generate_count_query,
            'average': self._generate_average_query,
            'sum': self._generate_sum_query,
            'longest': self._generate_longest_query,
            'top': self._generate_top_query,
            'show': self._generate_show_query,
            'list': self._generate_list_query,
            'find': self._generate_find_query
        }
        
        for pattern, generator in patterns.items():
            if pattern in question_lower:
                sql = generator(question_lower, data_context)
                if sql:
                    return sql
        
        return None
    
    def _generate_search_query(self, search_terms: Dict[str, str], question: str, 
                              data_context: DataContext) -> str:
        """Generate SQL for searching specific values"""
        
        if len(search_terms) == 1:
            # Single search term
            value, col = list(search_terms.items())[0]
            
            # Check if exact match or LIKE
            if 'exact' in question or '=' in question:
                return f"SELECT * FROM data_table WHERE {col} = '{value}'"
            else:
                return f"SELECT * FROM data_table WHERE {col} LIKE '%{value}%'"
        
        else:
            # Multiple search terms
            where_conditions = []
            for value, col in search_terms.items():
                where_conditions.append(f"{col} LIKE '%{value}%'")
            
            where_clause = ' OR '.join(where_conditions)
            return f"SELECT * FROM data_table WHERE {where_clause}"
    
    def _generate_entity_query(self, question: str, entity: str, entity_type: str, 
                              data_context: DataContext) -> str:
        """Generate query for specific entity"""
        
        # Find columns that might contain this entity
        relevant_cols = []
        
        # Check categorical columns first
        for col in data_context.categorical_columns:
            if col in data_context.content_samples:
                samples = data_context.content_samples[col]
                if any(entity.lower() in str(s).lower() for s in samples):
                    relevant_cols.append(col)
        
        # Check text columns if needed
        if not relevant_cols:
            relevant_cols = data_context.text_columns[:2]  # Limit to first 2 text columns
        
        if not relevant_cols:
            relevant_cols = [data_context.columns[0]]  # Fallback to first column
        
        # Generate appropriate query based on question type
        if any(term in question for term in ['count', 'how many', 'number']):
            col = relevant_cols[0]
            return f"""
            SELECT '{entity}' as search_term, COUNT(*) as count 
            FROM data_table 
            WHERE LOWER({col}) LIKE '%{entity.lower()}%'
            """
        
        else:
            # Show records containing the entity
            col = relevant_cols[0]
            return f"""
            SELECT * 
            FROM data_table 
            WHERE LOWER({col}) LIKE '%{entity.lower()}%'
            LIMIT 100
            """
    
    def _generate_count_query(self, question: str, data_context: DataContext) -> str:
        """Generate COUNT query based on context"""
        
        # Find what to count
        target = self._extract_count_target(question)
        
        if target:
            # Find relevant column
            col = self._find_relevant_column(target, data_context)
            
            if col:
                return f"""
                SELECT {col}, COUNT(*) as count
                FROM data_table
                GROUP BY {col}
                ORDER BY count DESC
                LIMIT 50
                """
        
        # Default count all
        return "SELECT COUNT(*) as total_count FROM data_table"
    
    def _generate_average_query(self, question: str, data_context: DataContext) -> str:
        """Generate AVG query"""
        
        # Find numeric columns mentioned or implied
        for col in data_context.numeric_columns:
            col_lower = col.lower()
            if col_lower in question or any(term in col_lower for term in 
                                           ['price', 'cost', 'amount', 'value', 'score']):
                return f"""
                SELECT 
                    AVG({col}) as average_{col},
                    MIN({col}) as min_{col},
                    MAX({col}) as max_{col},
                    COUNT(*) as count
                FROM data_table
                WHERE {col} IS NOT NULL
                """
        
        # Default to first numeric column
        if data_context.numeric_columns:
            col = data_context.numeric_columns[0]
            return f"SELECT AVG({col}) as average FROM data_table"
        
        return None
    
    def _generate_sum_query(self, question: str, data_context: DataContext) -> str:
        """Generate SUM query"""
        
        for col in data_context.numeric_columns:
            col_lower = col.lower()
            if col_lower in question or any(term in col_lower for term in 
                                           ['total', 'amount', 'value', 'revenue', 'cost']):
                
                # Check if we need to group by something
                group_col = self._find_grouping_column(question, data_context)
                
                if group_col:
                    return f"""
                    SELECT {group_col}, SUM({col}) as total_{col}
                    FROM data_table
                    GROUP BY {group_col}
                    ORDER BY total_{col} DESC
                    LIMIT 20
                    """
                else:
                    return f"SELECT SUM({col}) as total_{col} FROM data_table"
        
        return None
    
    def _generate_longest_query(self, question: str, data_context: DataContext) -> str:
        """Generate query for longest text"""
        
        # Find text column
        text_col = None
        
        for col in data_context.text_columns:
            if any(term in col.lower() for term in ['description', 'desc', 'text', 'content']):
                text_col = col
                break
        
        if not text_col and data_context.text_columns:
            text_col = data_context.text_columns[0]
        
        if text_col:
            # Include ID column if available
            id_col = data_context.id_columns[0] if data_context.id_columns else data_context.columns[0]
            
            return f"""
            SELECT 
                {id_col} as id,
                {text_col} as text_content,
                LENGTH({text_col}) as character_count
            FROM data_table
            WHERE {text_col} IS NOT NULL
            ORDER BY character_count DESC
            LIMIT 10
            """
        
        return None
    
    def _generate_top_query(self, question: str, data_context: DataContext) -> str:
        """Generate TOP/ORDER BY query"""
        
        # Extract number (default 10)
        import re
        numbers = re.findall(r'\d+', question)
        limit = int(numbers[0]) if numbers else 10
        
        # Find what to order by
        for col in data_context.numeric_columns:
            if col.lower() in question:
                return f"""
                SELECT *
                FROM data_table
                ORDER BY {col} DESC
                LIMIT {limit}
                """
        
        # Default to first numeric column
        if data_context.numeric_columns:
            col = data_context.numeric_columns[0]
            return f"SELECT * FROM data_table ORDER BY {col} DESC LIMIT {limit}"
        
        return f"SELECT * FROM data_table LIMIT {limit}"
    
    def _generate_show_query(self, question: str, data_context: DataContext) -> str:
        """Generate basic SELECT query"""
        
        # Check for specific values mentioned
        alphanumeric = re.findall(r'\b[A-Z]+\d+\b', question, re.IGNORECASE)
        if alphanumeric:
            # This is likely a search for specific value
            value = alphanumeric[0]
            best_col = self._find_best_column_for_value(value, data_context)
            if best_col:
                return f"SELECT * FROM data_table WHERE {best_col} = '{value}'"
        
        # Extract limit
        import re
        numbers = re.findall(r'\d+', question)
        limit = int(numbers[0]) if numbers else 100
        limit = min(limit, 1000)  # Cap at 1000
        
        return f"SELECT * FROM data_table LIMIT {limit}"
    
    def _generate_list_query(self, question: str, data_context: DataContext) -> str:
        """Generate LIST query with context awareness"""
        
        # Find what to list
        for col in data_context.categorical_columns[:3]:
            if col.lower() in question:
                return f"""
                SELECT DISTINCT {col}, COUNT(*) as count
                FROM data_table
                GROUP BY {col}
                ORDER BY count DESC
                """
        
        return self._generate_show_query(question, data_context)
    
    def _generate_find_query(self, question: str, data_context: DataContext) -> str:
        """Generate FIND/WHERE query"""
        
        # Extract search terms
        search_terms = []
        question_words = question.lower().split()
        
        # Look for quoted strings
        import re
        quoted = re.findall(r'"([^"]*)"', question)
        search_terms.extend(quoted)
        
        # Look for alphanumeric codes
        alphanumeric = re.findall(r'\b[A-Z]+\d+\b', question, re.IGNORECASE)
        search_terms.extend(alphanumeric)
        
        # Look for entities
        for entity_list in data_context.key_entities.values():
            for entity in entity_list:
                if entity.lower() in question:
                    search_terms.append(entity)
        
        if search_terms:
            # Build WHERE clause
            where_conditions = []
            
            # For each search term, find the best column
            for term in search_terms[:2]:  # Limit complexity
                best_col = self._find_best_column_for_value(term, data_context)
                if best_col:
                    where_conditions.append(f"{best_col} LIKE '%{term}%'")
                else:
                    # Search in text columns
                    for col in data_context.text_columns[:2]:
                        where_conditions.append(f"LOWER({col}) LIKE '%{term.lower()}%'")
            
            if where_conditions:
                where_clause = ' OR '.join(where_conditions)
                return f"""
                SELECT *
                FROM data_table
                WHERE {where_clause}
                LIMIT 100
                """
        
        return None
    
    def _context_aware_fallback(self, question: str, data_context: DataContext) -> str:
        """Generate intelligent fallback query based on data understanding"""
        
        # Show sample of most relevant data based on domain
        if data_context.domain_type == 'support_tickets':
            # Show recent or high priority tickets
            if 'priority' in data_context.categorical_columns:
                return f"""
                SELECT * FROM data_table 
                ORDER BY priority 
                LIMIT 50
                """
            
        elif data_context.domain_type == 'sales_data':
            # Show recent or high value sales
            for col in data_context.numeric_columns:
                if any(term in col.lower() for term in ['amount', 'value', 'total', 'revenue']):
                    return f"""
                    SELECT * FROM data_table 
                    ORDER BY {col} DESC 
                    LIMIT 50
                    """
        
        # Default: show diverse sample
        return "SELECT * FROM data_table LIMIT 100"
    
    def _extract_count_target(self, question: str) -> Optional[str]:
        """Extract what to count from question"""
        # Remove common words
        words = question.lower().split()
        ignore_words = {'how', 'many', 'count', 'number', 'of', 'the', 'a', 'an', 'what', 'is', 'are'}
        
        meaningful_words = [w for w in words if w not in ignore_words and len(w) > 2]
        
        if meaningful_words:
            return meaningful_words[0]
        
        return None
    
    def _find_relevant_column(self, target: str, data_context: DataContext) -> Optional[str]:
        """Find column most relevant to target term"""
        
        target_lower = target.lower()
        
        # Check column names
        for col in data_context.columns:
            if target_lower in col.lower():
                return col
        
        # Check categorical columns content
        for col in data_context.categorical_columns:
            if col in data_context.content_samples:
                samples = data_context.content_samples[col]
                if any(target_lower in str(s).lower() for s in samples):
                    return col
        
        # Check text columns as last resort
        for col in data_context.text_columns[:2]:
            return col
        
        return None
    
    def _find_grouping_column(self, question: str, data_context: DataContext) -> Optional[str]:
        """Find column to group by based on question"""
        
        # Look for "by" keyword
        if ' by ' in question:
            words_after_by = question.split(' by ')[-1].split()
            if words_after_by:
                target = words_after_by[0]
                
                for col in data_context.categorical_columns:
                    if target in col.lower():
                        return col
        
        # Default grouping columns based on domain
        if data_context.domain_type == 'support_tickets':
            for col in ['category', 'type', 'status', 'priority']:
                if col in data_context.categorical_columns:
                    return col
        
        elif data_context.domain_type == 'sales_data':
            for col in ['product', 'category', 'customer', 'region']:
                if col in data_context.categorical_columns:
                    return col
        
        return None

class HumanLikeAnalysisEngine:
    """Generate conversational, insightful analysis"""
    
    def __init__(self):
        self.understanding = EnhancedDataUnderstanding()
    
    async def analyze_results(self, question: str, query_result: pd.DataFrame, 
                            data_context: DataContext, full_df: pd.DataFrame,
                            sql_query: str = None) -> Dict[str, Any]:
        """Generate comprehensive human-like analysis"""
        
        # Generate analysis components
        executive_summary = await self._generate_executive_summary(
            question, query_result, data_context, full_df
        )
        
        detailed_insights = self._extract_detailed_insights(
            query_result, data_context, question
        )
        
        professional_opinion = await self._form_professional_opinion(
            question, query_result, data_context, detailed_insights
        )
        
        recommendations = self._generate_recommendations(
            question, detailed_insights, data_context
        )
        
        follow_up_suggestions = self._suggest_follow_ups(
            question, query_result, data_context
        )
        
        # Combine into cohesive response
        analysis = self._orchestrate_response(
            executive_summary,
            detailed_insights,
            professional_opinion,
            recommendations,
            follow_up_suggestions,
            query_result
        )
        
        return analysis
    
    async def _generate_executive_summary(self, question: str, results: pd.DataFrame,
                                        data_context: DataContext, full_df: pd.DataFrame) -> str:
        """Generate executive summary using AI or fallback"""
        
        try:
            api_key = getattr(settings, 'NVIDIA_NIM_API_KEY', None)
            if not api_key:
                return self._fallback_executive_summary(question, results, data_context)
            
            # Prepare context
            results_summary = self._summarize_for_ai(results)
            
            prompt = f"""You're a friendly data analyst explaining findings to a colleague.

Their question: {question}

Data context: {data_context.business_context}

Query returned {len(results)} results from {len(full_df):,} total records.

Results summary:
{results_summary}

Provide a natural, conversational 2-3 sentence summary that directly answers their question.
Start with something like "Looking at your data..." or "Here's what I found..."
Be specific with numbers and percentages."""

            api_endpoint = getattr(settings, 'NVIDIA_NIM_API_ENDPOINT', 
                                  'https://integrate.api.nvidia.com/v1/chat/completions')
            
            payload = {
                "model": "meta/llama-3.1-70b-instruct",
                "messages": [
                    {"role": "system", "content": "You are a friendly, helpful data analyst."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 200,
                "stream": False
            }
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(api_endpoint, json=payload, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            logger.warning(f"AI summary generation failed: {e}")
        
        return self._fallback_executive_summary(question, results, data_context)
    
    def _fallback_executive_summary(self, question: str, results: pd.DataFrame,
                                   data_context: DataContext) -> str:
        """Generate executive summary without AI"""
        
        if results.empty:
            return f"I searched for data related to '{question}' but didn't find any matching records. This could mean the search criteria were too specific or there genuinely aren't any relevant entries in your {data_context.domain_type}."
        
        question_lower = question.lower()
        
        # Count queries
        if 'count' in results.columns:
            total = results['count'].sum()
            top_item = results.iloc[0]
            top_name = top_item[results.columns[0]]
            top_count = top_item['count']
            percentage = (top_count / total * 100) if total > 0 else 0
            
            return f"Looking at your {data_context.domain_type}, I found {total:,} total items matching your criteria. The most common is '{top_name}' with {top_count:,} occurrences ({percentage:.1f}% of the total). The data shows {len(results)} different categories overall."
        
        # Average queries
        elif any(col.startswith('average_') for col in results.columns):
            avg_col = [col for col in results.columns if col.startswith('average_')][0]
            avg_value = results[avg_col].iloc[0] if not results.empty else 0
            
            return f"Based on your {len(results):,} records, the average value is {avg_value:,.2f}. This gives you a good baseline for understanding typical values in your {data_context.domain_type}."
        
        # General data queries
        else:
            return f"Here's what I found in your {data_context.domain_type}: {len(results):,} records matching your query. The data includes {len(results.columns)} different attributes that might help answer your question about {question}."
    
    def _extract_detailed_insights(self, results: pd.DataFrame, data_context: DataContext,
                                  question: str) -> Dict[str, Any]:
        """Extract detailed insights from results"""
        
        insights = {
            'patterns': [],
            'anomalies': [],
            'trends': [],
            'key_findings': [],
            'statistics': {}
        }
        
        if results.empty:
            return insights
        
        # Analyze based on result type
        if 'count' in results.columns:
            insights.update(self._analyze_count_results(results))
        
        elif 'character_count' in results.columns:
            insights.update(self._analyze_text_length_results(results))
        
        else:
            # General data analysis
            insights.update(self._analyze_general_results(results, data_context))
        
        # Sanitize statistics for JSON
        insights['statistics'] = sanitize_for_json(insights['statistics'])
        
        return insights
    
    def _analyze_count_results(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Analyze count/grouping results"""
        
        insights = {
            'patterns': [],
            'anomalies': [],
            'key_findings': [],
            'statistics': {}
        }
        
        total = results['count'].sum()
        
        # Distribution analysis
        if len(results) > 1:
            top_pct = (results.iloc[0]['count'] / total * 100) if total > 0 else 0
            
            if top_pct > 50:
                insights['patterns'].append(
                    f"Heavy concentration: {results.iloc[0][results.columns[0]]} represents {top_pct:.1f}% of all cases"
                )
            elif top_pct < 10:
                insights['patterns'].append(
                    "Highly distributed: No single category dominates, suggesting diverse data"
                )
            
            # Check for long tail
            bottom_half = results[len(results)//2:]
            bottom_total = bottom_half['count'].sum()
            if bottom_total < total * 0.2:
                insights['patterns'].append(
                    f"Long tail distribution: Bottom {len(bottom_half)} categories represent only {(bottom_total/total*100):.1f}% of total"
                )
        
        # Find anomalies
        if len(results) > 5:
            mean_count = results['count'].mean()
            std_count = results['count'].std()
            
            for idx, row in results.iterrows():
                if row['count'] > mean_count + 2 * std_count:
                    insights['anomalies'].append(
                        f"{row[results.columns[0]]} is unusually high with {row['count']:,} occurrences"
                    )
        
        # Key statistics
        insights['statistics'] = {
            'total_items': int(total),
            'unique_categories': len(results),
            'average_per_category': safe_float(results['count'].mean()),
            'median_per_category': safe_float(results['count'].median())
        }
        
        return insights
    
    def _analyze_text_length_results(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Analyze text length results"""
        
        insights = {
            'patterns': [],
            'key_findings': [],
            'statistics': {}
        }
        
        if not results.empty and 'character_count' in results.columns:
            max_length = results['character_count'].max()
            min_length = results['character_count'].min()
            avg_length = results['character_count'].mean()
            
            insights['key_findings'].append(
                f"Text lengths range from {min_length:,} to {max_length:,} characters"
            )
            
            if max_length > avg_length * 3:
                insights['patterns'].append(
                    f"The longest entry is {(max_length/avg_length):.1f}x longer than average, indicating significant variation"
                )
            
            # Extract and analyze content if available
            if 'text_content' in results.columns and len(results) > 0:
                longest_text = str(results.iloc[0]['text_content'])
                
                # Simple technology extraction
                tech_keywords = ['Python', 'Java', 'JavaScript', 'SQL', 'API', 'Docker', 
                               'AWS', 'React', 'Node.js', 'database']
                
                found_tech = []
                for tech in tech_keywords:
                    if tech.lower() in longest_text.lower():
                        found_tech.append(tech)
                
                if found_tech:
                    insights['key_findings'].append(
                        f"Technologies mentioned: {', '.join(found_tech)}"
                    )
        
        return insights
    
    def _analyze_general_results(self, results: pd.DataFrame, 
                                data_context: DataContext) -> Dict[str, Any]:
        """Analyze general query results"""
        
        insights = {
            'patterns': [],
            'trends': [],
            'key_findings': [],
            'statistics': {}
        }
        
        # Numeric analysis
        numeric_cols = results.select_dtypes(include=['number']).columns
        
        for col in numeric_cols[:3]:  # Analyze top 3 numeric columns
            if len(results) > 0:
                col_stats = {
                    'mean': safe_float(results[col].mean()),
                    'median': safe_float(results[col].median()),
                    'std': safe_float(results[col].std()),
                    'min': safe_float(results[col].min()),
                    'max': safe_float(results[col].max())
                }
                
                insights['statistics'][col] = col_stats
                
                # Check for patterns
                if col_stats['std'] is not None and col_stats['mean'] is not None:
                    if col_stats['std'] > col_stats['mean'] * 0.5:
                        insights['patterns'].append(
                            f"High variability in {col}: standard deviation is {(col_stats['std']/col_stats['mean']*100):.1f}% of mean"
                        )
                
                if col_stats['median'] is not None and col_stats['mean'] is not None:
                    if col_stats['median'] < col_stats['mean'] * 0.8:
                        insights['patterns'].append(
                            f"{col} appears right-skewed: median ({col_stats['median']:.2f}) is significantly lower than mean ({col_stats['mean']:.2f})"
                        )
        
        # Categorical analysis
        cat_cols = results.select_dtypes(include=['object']).columns
        
        for col in cat_cols[:2]:  # Analyze top 2 categorical columns
            if col in data_context.categorical_columns:
                value_counts = results[col].value_counts()
                if len(value_counts) > 1:
                    insights['key_findings'].append(
                        f"{col} has {len(value_counts)} unique values, with '{value_counts.index[0]}' being most common"
                    )
        
        return insights
    
    async def _form_professional_opinion(self, question: str, results: pd.DataFrame,
                                       data_context: DataContext, 
                                       insights: Dict[str, Any]) -> str:
        """Form professional opinion based on data"""
        
        try:
            api_key = getattr(settings, 'NVIDIA_NIM_API_KEY', None)
            if not api_key:
                return self._fallback_professional_opinion(insights, data_context)
            
            # Build context for opinion formation
            context = f"""Based on the analysis of {data_context.business_context}:

Question asked: {question}

Key insights discovered:
- Patterns: {', '.join(insights['patterns'][:3]) if insights['patterns'] else 'No clear patterns'}
- Anomalies: {', '.join(insights['anomalies'][:2]) if insights['anomalies'] else 'No anomalies detected'}
- Key findings: {', '.join(insights['key_findings'][:3]) if insights['key_findings'] else 'Standard distribution'}

Statistics: {json.dumps(sanitize_for_json(insights['statistics']), indent=2) if insights['statistics'] else 'No statistics'}

Form a professional opinion as an experienced data analyst. Include:
1. What this data tells us from a business perspective
2. Any concerns or opportunities you see
3. Your confidence level in these insights
4. What similar patterns you've seen before

Be conversational and insightful. Use phrases like "In my experience...", "What stands out to me...", "I'd be interested to know..."
"""

            api_endpoint = getattr(settings, 'NVIDIA_NIM_API_ENDPOINT', 
                                  'https://integrate.api.nvidia.com/v1/chat/completions')
            
            payload = {
                "model": "meta/llama-3.1-70b-instruct",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are an experienced data analyst with 10+ years of experience. You provide thoughtful, nuanced opinions based on data patterns."
                    },
                    {"role": "user", "content": context}
                ],
                "temperature": 0.8,
                "max_tokens": 400,
                "stream": False
            }
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(api_endpoint, json=payload, headers=headers, timeout=15) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            logger.warning(f"AI opinion generation failed: {e}")
        
        return self._fallback_professional_opinion(insights, data_context)
    
    def _fallback_professional_opinion(self, insights: Dict[str, Any], 
                                     data_context: DataContext) -> str:
        """Generate professional opinion without AI"""
        
        opinion_parts = ["## My Professional Opinion\n"]
        
        if insights['patterns']:
            opinion_parts.append("What stands out to me in this data is the clear patterns I'm seeing:")
            for pattern in insights['patterns'][:2]:
                opinion_parts.append(f"• {pattern}")
            opinion_parts.append("")
        
        if insights['anomalies']:
            opinion_parts.append("I'd pay special attention to these anomalies:")
            for anomaly in insights['anomalies'][:2]:
                opinion_parts.append(f"• {anomaly}")
            opinion_parts.append("\nThese could represent either issues to address or opportunities to explore.")
        
        # Add domain-specific insights
        if data_context.domain_type == 'support_tickets':
            opinion_parts.append("\nIn my experience with support data, these patterns often indicate:")
            opinion_parts.append("• Resource allocation opportunities")
            opinion_parts.append("• Training needs in specific areas")
            opinion_parts.append("• Potential process improvements")
        
        elif data_context.domain_type == 'sales_data':
            opinion_parts.append("\nFrom a sales perspective, this suggests:")
            opinion_parts.append("• Market segment performance variations")
            opinion_parts.append("• Potential for targeted campaigns")
            opinion_parts.append("• Revenue optimization opportunities")
        
        opinion_parts.append("\nI'd recommend digging deeper into the specific patterns identified to understand root causes.")
        
        return '\n'.join(opinion_parts)
    
    def _generate_recommendations(self, question: str, insights: Dict[str, Any],
                                data_context: DataContext) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Pattern-based recommendations
        if insights['patterns']:
            if any('concentration' in p or 'dominates' in p for p in insights['patterns']):
                recommendations.append(
                    "**Diversification Strategy**: Consider strategies to reduce dependency on dominant categories"
                )
            
            if any('distributed' in p or 'diverse' in p for p in insights['patterns']):
                recommendations.append(
                    "**Segmentation Approach**: With diverse data, consider segment-specific strategies"
                )
        
        # Anomaly-based recommendations
        if insights['anomalies']:
            recommendations.append(
                "**Investigate Anomalies**: The unusual patterns detected warrant deeper investigation"
            )
        
        # Domain-specific recommendations
        if data_context.domain_type == 'support_tickets':
            if insights['statistics'] and insights['statistics'].get('total_items', 0) > 100:
                recommendations.append(
                    "**Automation Opportunity**: With this volume, consider automating common issue resolutions"
                )
        
        elif data_context.domain_type == 'sales_data':
            recommendations.append(
                "**Revenue Analysis**: Analyze high-value transactions for replication strategies"
            )
        
        # Question-specific recommendations
        question_lower = question.lower()
        
        if 'startup' in question_lower or 'business' in question_lower:
            recommendations.extend([
                "**Market Validation**: Use this data to validate market demand before investing",
                "**MVP Approach**: Start with the most common use cases identified in the data",
                "**Risk Assessment**: Pay attention to failure patterns in similar ventures"
            ])
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _suggest_follow_ups(self, question: str, results: pd.DataFrame,
                          data_context: DataContext) -> List[str]:
        """Suggest relevant follow-up questions"""
        
        suggestions = []
        
        question_lower = question.lower()
        
        # Based on current question type
        if any(term in question_lower for term in ['count', 'how many']):
            suggestions.extend([
                "What's the trend over time for these counts?",
                "Which categories have grown the most?",
                "What factors correlate with higher counts?"
            ])
        
        elif any(term in question_lower for term in ['average', 'mean']):
            suggestions.extend([
                "What are the outliers affecting this average?",
                "How does this compare to industry benchmarks?",
                "What's the distribution look like?"
            ])
        
        elif any(term in question_lower for term in ['longest', 'shortest']):
            suggestions.extend([
                "What patterns exist in the longest entries?",
                "Is length correlated with any other factors?",
                "What's the typical length distribution?"
            ])
        
        # Based on data context
        if data_context.domain_type == 'support_tickets':
            suggestions.extend([
                "What's the average resolution time by category?",
                "Which issues have the highest recurrence rate?",
                "What's the customer satisfaction by issue type?"
            ])
        
        elif data_context.domain_type == 'sales_data':
            suggestions.extend([
                "What's the customer lifetime value distribution?",
                "Which products have the highest profit margins?",
                "What seasonal patterns exist in the data?"
            ])
        
        # Based on results
        if not results.empty:
            if 'count' in results.columns and len(results) > 5:
                suggestions.append(
                    f"Why is '{results.iloc[0][results.columns[0]]}' so dominant?"
                )
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def _orchestrate_response(self, executive_summary: str, insights: Dict[str, Any],
                            professional_opinion: str, recommendations: List[str],
                            follow_ups: List[str], results: pd.DataFrame) -> Dict[str, Any]:
        """Orchestrate all components into cohesive response"""
        
        # Build main analysis narrative
        analysis_parts = []
        
        # Executive summary (conversational opener)
        analysis_parts.append(executive_summary)
        analysis_parts.append("")
        
        # Key insights (if any)
        if insights['key_findings'] or insights['patterns']:
            analysis_parts.append("## 📊 What the Data Shows\n")
            
            if insights['key_findings']:
                for finding in insights['key_findings'][:3]:
                    analysis_parts.append(f"• {finding}")
                analysis_parts.append("")
            
            if insights['patterns']:
                analysis_parts.append("**Patterns I noticed:**")
                for pattern in insights['patterns'][:3]:
                    analysis_parts.append(f"• {pattern}")
                analysis_parts.append("")
            
            if insights['anomalies']:
                analysis_parts.append("**🚨 Anomalies to note:**")
                for anomaly in insights['anomalies'][:2]:
                    analysis_parts.append(f"• {anomaly}")
                analysis_parts.append("")
        
        # Professional opinion
        if professional_opinion:
            analysis_parts.append(professional_opinion)
            analysis_parts.append("")
        
        # Recommendations
        if recommendations:
            analysis_parts.append("## 💡 My Recommendations\n")
            for i, rec in enumerate(recommendations, 1):
                analysis_parts.append(f"{i}. {rec}")
            analysis_parts.append("")
        
        # Follow-up questions
        if follow_ups:
            analysis_parts.append("## 🔍 Questions to Explore Next\n")
            analysis_parts.append("Based on what we've found, you might want to ask:")
            for question in follow_ups[:3]:
                analysis_parts.append(f"• {question}")
        
        # Prepare response with sanitized data
        response = {
            'analysis': '\n'.join(analysis_parts),
            'insights': sanitize_for_json(insights),
            'has_recommendations': len(recommendations) > 0,
            'follow_up_questions': follow_ups[:3],
            'confidence_level': self._calculate_confidence(results, insights)
        }
        
        return response
    
    def _calculate_confidence(self, results: pd.DataFrame, insights: Dict[str, Any]) -> str:
        """Calculate confidence level in the analysis"""
        
        if results.empty:
            return "low"
        
        if len(results) < 10:
            return "medium"
        
        if len(insights['patterns']) > 2 or len(insights['key_findings']) > 2:
            return "high"
        
        return "medium"
    
    def _summarize_for_ai(self, results: pd.DataFrame) -> str:
        """Summarize results for AI consumption"""
        
        if results.empty:
            return "No results found"
        
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"Result shape: {results.shape}")
        
        # Column types
        summary_parts.append(f"Columns: {', '.join(results.columns)}")
        
        # For count results
        if 'count' in results.columns:
            total = results['count'].sum()
            top_5 = results.head(5)
            
            summary_parts.append(f"Total count: {total}")
            summary_parts.append("Top 5 items:")
            
            for idx, row in top_5.iterrows():
                item = row[results.columns[0]]
                count = row['count']
                pct = (count / total * 100) if total > 0 else 0
                summary_parts.append(f"- {item}: {count} ({pct:.1f}%)")
        
        # For numeric results
        else:
            numeric_cols = results.select_dtypes(include=['number']).columns
            
            for col in numeric_cols[:3]:
                col_min = safe_float(results[col].min())
                col_max = safe_float(results[col].max())
                col_mean = safe_float(results[col].mean())
                
                if col_min is not None and col_max is not None and col_mean is not None:
                    summary_parts.append(f"{col}: min={col_min:.2f}, max={col_max:.2f}, mean={col_mean:.2f}")
        
        # Sample rows
        if len(results) > 5:
            summary_parts.append(f"\nShowing first 5 of {len(results)} rows")
            summary_parts.append(results.head(5).to_string())
        else:
            summary_parts.append("\nAll results:")
            summary_parts.append(results.to_string())
        
        return '\n'.join(summary_parts)

class UploadFileView(View):
    """Handle file upload with enhanced understanding"""
    
    def get(self, request):
        return render(request, 'TableAgent/upload.html')
    
    def post(self, request):
        if request.FILES.get('data_file'):
            data_file = request.FILES['data_file']
            
            # Save file
            file_path = default_storage.save(
                f"uploads/{data_file.name}", 
                ContentFile(data_file.read())
            )
            
            # Store in session
            request.session['uploaded_file'] = file_path
            request.session['file_name'] = data_file.name
            
            # Build enhanced understanding
            self._build_data_understanding(file_path, data_file.name)
            
            return redirect('ask')
        
        return render(request, 'TableAgent/upload.html')
    
    def _build_data_understanding(self, file_path: str, file_name: str):
        """Build and cache comprehensive data understanding"""
        try:
            full_path = os.path.join(settings.MEDIA_ROOT, file_path)
            
            # Load data
            if file_name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(full_path)
            else:
                df = pd.read_csv(full_path)
            
            # Build understanding
            understanding = EnhancedDataUnderstanding()
            data_context = understanding.analyze_data(df, file_name)
            
            # Convert to serializable format
            context_dict = {
                'file_path': data_context.file_path,
                'columns': data_context.columns,
                'dtypes': data_context.dtypes,
                'row_count': data_context.row_count,
                'domain_type': data_context.domain_type,
                'key_entities': data_context.key_entities,
                'text_columns': data_context.text_columns,
                'numeric_columns': data_context.numeric_columns,
                'categorical_columns': data_context.categorical_columns,
                'id_columns': data_context.id_columns,
                'date_columns': data_context.date_columns,
                'content_samples': sanitize_for_json(data_context.content_samples),
                'business_context': data_context.business_context,
                'column_semantics': data_context.column_semantics,
                'column_value_patterns': data_context.column_value_patterns
            }
            
            # Cache for 2 hours
            cache_key = f"data_context_{file_path}"
            cache.set(cache_key, context_dict, 7200)
            
            logger.info(f"Built data understanding for {file_name}: {data_context.domain_type} with {data_context.row_count} rows")
            
        except Exception as e:
            logger.error(f"Error building data understanding: {str(e)}")

class AskQuestionView(View):
    """Simplified question handler with sequential intelligence"""
    
    def __init__(self):
        self.sql_generator = IntelligentSQLGenerator()
        self.analysis_engine = HumanLikeAnalysisEngine()
    
    def get(self, request):
        file_path = request.session.get('uploaded_file')
        file_name = request.session.get('file_name')
        
        if not file_path:
            messages.error(request, 'Please upload a file first.')
            return redirect('upload')
        
        # Get data context
        data_context = self._get_data_context(file_path)
        
        if data_context:
            context = {
                'file_name': file_name,
                'columns': data_context['columns'],
                'row_count': data_context['row_count'],
                'business_context': data_context['business_context'],
                'domain_type': data_context['domain_type'],
                'sample_data': []  # We'll get this separately
            }
        else:
            # Fallback if context not cached
            full_path = os.path.join(settings.MEDIA_ROOT, file_path)
            try:
                if file_name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(full_path)
                else:
                    df = pd.read_csv(full_path)
                
                context = {
                    'file_name': file_name,
                    'columns': df.columns.tolist(),
                    'row_count': len(df),
                    'business_context': f"Data from {file_name}",
                    'domain_type': 'general_data',
                    'sample_data': df.head(5).to_dict('records')
                }
            except Exception as e:
                messages.error(request, f'Error reading file: {str(e)}')
                return redirect('upload')
        
        # Get sample data
        if not context.get('sample_data'):
            try:
                full_path = os.path.join(settings.MEDIA_ROOT, file_path)
                if file_name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(full_path)
                else:
                    df = pd.read_csv(full_path)
                context['sample_data'] = df.head(5).to_dict('records')
            except:
                context['sample_data'] = []
        
        return render(request, 'TableAgent/ask.html', context)
    
    def post(self, request):
        """Single unified processing path for all questions"""
        
        file_path = request.session.get('uploaded_file')
        if not file_path:
            return JsonResponse({'status': 'error', 'message': 'No file uploaded'})
        
        question = request.POST.get('question')
        if not question:
            return JsonResponse({'status': 'error', 'message': 'No question provided'})
        
        try:
            # Sequential processing pipeline
            result = asyncio.run(self._process_question_sequentially(question, file_path))
            return JsonResponse({'status': 'success', 'result': result})
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    async def _process_question_sequentially(self, question: str, file_path: str) -> Dict[str, Any]:
        """Process question through sequential intelligence pipeline"""
        
        logger.info(f"Processing question: {question}")
        
        # Step 1: Load data and context
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        file_name = file_path.split('/')[-1]
        
        if file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(full_path)
        else:
            df = pd.read_csv(full_path)
        
        logger.info(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
        
        # Step 2: Get or build data context
        data_context = self._get_or_build_data_context(file_path, df, file_name)
        
        # Step 3: Generate SQL with context awareness
        sql_query, generation_method = await self.sql_generator.generate_sql(
            question, data_context, df.head(100)
        )
        
        logger.info(f"Generated SQL ({generation_method}): {sql_query}")
        
        # Step 4: Execute query with error recovery
        query_result, error_occurred = self._execute_query_with_recovery(sql_query, df, question, data_context)
        
        if query_result is None:
            # Fallback to showing sample data
            query_result = df.head(50)
            logger.warning("Query failed, showing sample data")
        
        # Step 5: Generate human-like analysis
        analysis_result = await self.analysis_engine.analyze_results(
            question, query_result, data_context, df, sql_query
        )
        
        # Step 6: Format response with sanitization
        formatted_data = self._format_results(query_result)
        
        # Ensure all data is JSON-serializable
        response = {
            'analysis': analysis_result['analysis'],
            'data': formatted_data,
            'has_data': len(formatted_data['data']) > 0,
            'row_count': len(query_result),
            'query_executed': True,
            'columns': formatted_data['columns'],
            'insights': sanitize_for_json(analysis_result.get('insights', {})),
            'confidence': analysis_result.get('confidence_level', 'medium'),
            'follow_up_questions': analysis_result.get('follow_up_questions', []),
            'generation_method': generation_method,
            'error_recovered': error_occurred
        }
        
        return response
    
    def _execute_query_with_recovery(self, sql_query: str, df: pd.DataFrame, 
                                    question: str, data_context: DataContext) -> Tuple[Optional[pd.DataFrame], bool]:
        """Execute SQL query with error recovery for column not found errors"""
        error_occurred = False
        
        try:
            # Register DataFrame with DuckDB
            con.register('data_table', df)
            
            # Execute query
            result = con.execute(sql_query).fetchdf()
            
            return result, error_occurred
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"Query execution error: {error_str}")
            
            # Check if it's a column not found error
            if "not found in FROM clause" in error_str or "Referenced column" in error_str:
                error_occurred = True
                
                # Extract the problematic term
                match = re.search(r'Referenced column "([^"]+)"', error_str)
                if match:
                    problematic_term = match.group(1)
                    logger.info(f"Detected '{problematic_term}' was mistaken for a column")
                    
                    # Find the best column for this value
                    best_col = self.sql_generator._find_best_column_for_value(
                        problematic_term, data_context
                    )
                    
                    if best_col:
                        # Generate a corrected query
                        corrected_sql = f"SELECT * FROM data_table WHERE {best_col} = '{problematic_term}'"
                        logger.info(f"Attempting corrected query: {corrected_sql}")
                        
                        try:
                            result = con.execute(corrected_sql).fetchdf()
                            return result, error_occurred
                        except:
                            pass
            
            # Try simpler query as fallback
            try:
                return con.execute("SELECT * FROM data_table LIMIT 50").fetchdf(), error_occurred
            except:
                return None, error_occurred
    
    def _get_data_context(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get cached data context"""
        cache_key = f"data_context_{file_path}"
        return cache.get(cache_key)
    
    def _get_or_build_data_context(self, file_path: str, df: pd.DataFrame, 
                                  file_name: str) -> DataContext:
        """Get cached context or build new one"""
        
        # Try cache first
        cached_context = self._get_data_context(file_path)
        
        if cached_context:
            # Convert dict back to DataContext
            return DataContext(**cached_context)
        
        # Build fresh context
        understanding = EnhancedDataUnderstanding()
        data_context = understanding.analyze_data(df, file_name)
        
        # Cache it with sanitization
        context_dict = {
            'file_path': data_context.file_path,
            'columns': data_context.columns,
            'dtypes': data_context.dtypes,
            'row_count': data_context.row_count,
            'domain_type': data_context.domain_type,
            'key_entities': data_context.key_entities,
            'text_columns': data_context.text_columns,
            'numeric_columns': data_context.numeric_columns,
            'categorical_columns': data_context.categorical_columns,
            'id_columns': data_context.id_columns,
            'date_columns': data_context.date_columns,
            'content_samples': sanitize_for_json(data_context.content_samples),
            'business_context': data_context.business_context,
            'column_semantics': data_context.column_semantics,
            'column_value_patterns': data_context.column_value_patterns
        }
        
        cache_key = f"data_context_{file_path}"
        cache.set(cache_key, context_dict, 7200)
        
        return data_context
    
    def _format_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Format results for display with proper sanitization"""
        if df is None or df.empty:
            return {'columns': [], 'data': []}
        
        # Convert to JSON-serializable format
        columns = df.columns.tolist()
        
        # Handle different data types with sanitization
        data = []
        for _, row in df.head(500).iterrows():  # Limit to 500 rows for performance
            row_data = []
            for col in columns:
                value = row[col]
                
                # Handle different types with sanitization
                if pd.isna(value):
                    row_data.append(None)
                elif isinstance(value, (np.integer, np.int64)):
                    row_data.append(int(value))
                elif isinstance(value, (np.floating, np.float64)):
                    # Handle NaN and infinity
                    if math.isnan(value) or math.isinf(value):
                        row_data.append(None)
                    else:
                        row_data.append(float(value))
                else:
                    row_data.append(str(value))
            
            data.append(row_data)
        
        return {
            'columns': columns,
            'data': data
        }

def get_table_info(request):
    """Return enhanced table information"""
    file_path = request.session.get('uploaded_file')
    if not file_path:
        return JsonResponse({'status': 'error', 'message': 'No file uploaded'})
    
    # Get cached context
    cache_key = f"data_context_{file_path}"
    data_context = cache.get(cache_key)
    
    if data_context:
        return JsonResponse({
            'status': 'success',
            'info': sanitize_for_json({
                'columns': data_context['columns'],
                'data_types': data_context['dtypes'],
                'row_count': data_context['row_count'],
                'domain_type': data_context['domain_type'],
                'business_context': data_context['business_context'],
                'key_entities': data_context.get('key_entities', {}),
                'column_descriptions': data_context.get('column_semantics', {}),
                'column_patterns': data_context.get('column_value_patterns', {})
            })
        })
    
    # Fallback to basic info
    try:
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        file_name = request.session.get('file_name', '')
        
        if file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(full_path)
        else:
            df = pd.read_csv(full_path)
        
        info = {
            'columns': df.columns.tolist(),
            'data_types': [str(df[col].dtype) for col in df.columns],
            'row_count': len(df),
            'domain_type': 'general_data',
            'business_context': f"Data from {file_name}"
        }
        
        return JsonResponse({'status': 'success', 'info': sanitize_for_json(info)})
        
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})