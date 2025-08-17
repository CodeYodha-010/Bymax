# apps/TestAgent/views.py - ADVANCED VERSION
import os
import hashlib
import logging
import pandas as pd
import duckdb
import requests
import json
import re
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from django.urls import reverse
from . import forms

# --- Logging Setup ---
logger = logging.getLogger(__name__)

# --- Helper function to get DuckDB connection ---
def get_duckdb_connection(request):
    """Establishes a DuckDB connection and generates a session-specific table name."""
    db_path = 'user_data.db'
    con = duckdb.connect(db_path)
    session_key = request.session.session_key or 'default_session'
    table_name_hash = hashlib.md5(session_key.encode()).hexdigest()
    table_name = f"session_data_{table_name_hash}"
    request.session['duckdb_table_name'] = table_name
    return con, table_name

# --- Enhanced Data Profiling Function ---
def perform_enhanced_profiling(con, table_name, df, request):
    """Analyzes the uploaded data and stores metadata."""
    table_metadata = {
        'basic_info': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_names': df.columns.tolist()
        },
        'columns': {}
    }

    logger.info(f"Starting enhanced profiling for table: {table_name}")
    try:
        for col in df.columns:
            col_metadata = {
                'name': col,
                'dtype_pandas': str(df[col].dtype),
                'dtype_duckdb': None,
                'semantic_type': 'unknown',
                'stats': {},
                'distinct_count': None,
                'top_values': [],
                'sample_values': []
            }

            dtype_result = con.execute(f"SELECT data_type FROM information_schema.columns WHERE table_name = '{table_name}' AND column_name = '{col}'").fetchone()
            if dtype_result:
                col_metadata['dtype_duckdb'] = dtype_result[0]

            sample_vals = con.execute(f"SELECT DISTINCT {col} FROM {table_name} WHERE {col} IS NOT NULL LIMIT 5").fetchall()
            col_metadata['sample_values'] = [row[0] for row in sample_vals]

            # Enhanced semantic type inference
            col_name_lower = col.lower()
            if 'id' in col_name_lower or 'identifier' in col_name_lower:
                col_metadata['semantic_type'] = 'identifier'
            elif any(keyword in col_name_lower for keyword in ['date', 'time', 'year', 'month']):
                col_metadata['semantic_type'] = 'temporal'
            elif col_metadata['dtype_duckdb'] == 'VARCHAR' and df[col].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}', na=False).any():
                col_metadata['semantic_type'] = 'temporal'
            elif any(keyword in col_name_lower for keyword in ['count', 'number', 'amount', 'price', 'score', 'rate']):
                col_metadata['semantic_type'] = 'numerical'
            elif any(keyword in col_name_lower for keyword in ['category', 'bucket', 'type', 'status', 'department', 'organisation', 'title', 'technology', 'problem', 'statement', 'description']):
                col_metadata['semantic_type'] = 'categorical'

            if pd.api.types.is_numeric_dtype(df[col]) or col_metadata['semantic_type'] == 'numerical':
                col_metadata['semantic_type'] = 'numerical'
                stats_query = f"""
                    SELECT
                        MIN({col}) as min_val,
                        MAX({col}) as max_val,
                        AVG({col}) as avg_val,
                        STDDEV_SAMP({col}) as stddev_val,
                        COUNT({col}) as non_null_count
                    FROM {table_name}
                """
                stats_result = con.execute(stats_query).fetchone()
                if stats_result:
                    col_metadata['stats'] = {
                        'min': stats_result[0], 'max': stats_result[1],
                        'mean': stats_result[2], 'stddev': stats_result[3],
                        'count': stats_result[4]
                    }
            else:
                if col_metadata['semantic_type'] == 'unknown':
                    col_metadata['semantic_type'] = 'text'
                distinct_count_result = con.execute(f"SELECT COUNT(DISTINCT {col}) FROM {table_name}").fetchone()
                col_metadata['distinct_count'] = distinct_count_result[0] if distinct_count_result else 0

                if col_metadata['distinct_count'] > 0 and col_metadata['distinct_count'] <= 50:
                    top_values_query = f"""
                        SELECT {col}, COUNT(*) as freq
                        FROM {table_name}
                        GROUP BY {col}
                        ORDER BY freq DESC
                        LIMIT 10
                    """
                    top_values_result = con.execute(top_values_query).fetchall()
                    col_metadata['top_values'] = [{'value': row[0], 'count': row[1]} for row in top_values_result]

            table_metadata['columns'][col] = col_metadata

        request.session['table_metadata'] = table_metadata
        logger.info(f"Enhanced profiling complete for table: {table_name}")
        return True
    except Exception as e:
        logger.error(f"Error during enhanced profiling for table {table_name}: {e}")
        return False

# --- ADVANCED: Intent Classification Function ---
def classify_question_intent(question, table_metadata):
    """Advanced intent classification with better context understanding."""
    api_key = getattr(settings, 'NVIDIA_NIM_API_KEY', None)
    api_endpoint = getattr(settings, 'NVIDIA_NIM_API_ENDPOINT', 'https://integrate.api.nvidia.com/v1/chat/completions')
    if not api_key:
        logger.error("NVIDIA_NIM_API_KEY not found for intent classification.")
        return None

    # Prepare enhanced context from metadata
    schema_summary = []
    for col_name, col_meta in table_metadata.get('columns', {}).items():
        summary_line = f"  - {col_name} ({col_meta.get('semantic_type', 'unknown')})"
        if col_meta.get('distinct_count') is not None:
            summary_line += f" [~{col_meta['distinct_count']} unique values]"
        elif col_meta.get('stats'):
            summary_line += f" [Numerical: mean={col_meta['stats'].get('mean', 'N/A'):.2f}]"
        if col_meta.get('top_values'):
            top_vals = [str(tv['value'])[:30] for tv in col_meta['top_values'][:3]]
            summary_line += f" [Common: {', '.join(top_vals)}]"
        schema_summary.append(summary_line)
    schema_info_str = "\n".join(schema_summary)

    intent_prompt = f"""
    You are an advanced AI assistant analyzing user questions about a dataset.

    Dataset Schema Summary:
    {schema_info_str}

    User Question: {question}

    Analyze this question and provide detailed classification:

    1. Intent Category: Choose from [Simple_Lookup, Count_Aggregation, Filtering, Comparison, Trend_Analysis, Statistical_Analysis, Text_Search, Data_Exploration, Complex_Query]
    2. Key Entities: Identify specific columns, values, or concepts mentioned
    3. Query Complexity: Rate as [Low, Medium, High] based on required operations
    4. Expected Result Type: [Single_Value, List, Table, Summary, Analysis]
    5. Suggested SQL Approach: Outline the main SQL operations needed
    6. Potential Challenges: Identify any data type issues or complex requirements
    7. Result Validation: How to verify the answer matches the user's intent

    Format as JSON:
    {{
      "intent_category": "...",
      "key_entities": ["...", "..."],
      "query_complexity": "...",
      "expected_result_type": "...",
      "suggested_approach": ["...", "..."],
      "potential_challenges": ["...", "..."],
      "result_validation": "..."
    }}
    """

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    payload = {
        "model": getattr(settings, 'NVIDIA_NIM_MODEL', 'meta/llama-3.1-405b-instruct'),
        "messages": [
            {"role": "system", "content": "You are an expert data analyst. Respond only with valid JSON for query classification."},
            {"role": "user", "content": intent_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 700,
        "top_p": 1,
        "stream": False
    }

    try:
        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        intent_json_str = response_data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        intent_data = json.loads(intent_json_str)
        logger.debug(f"Advanced intent classified: {intent_data}")
        return intent_data
    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error in advanced intent classification: {e}")
        return None

# --- ADVANCED: Result Formatter ---
def format_advanced_response(question, query_result_data, result_columns, intent_data, safe_sql):
    """Formats query results into structured, user-friendly responses."""
    
    if not query_result_data:
        return {
            'type': 'no_results',
            'message': "Your query executed successfully but returned no results.",
            'suggestions': ["Try broadening your search criteria", "Check for typos in filter values", "Verify the data contains what you're looking for"]
        }
    
    result_count = len(query_result_data)
    expected_type = intent_data.get('expected_result_type', 'Unknown') if intent_data else 'Unknown'
    
    # Single value result
    if result_count == 1 and len(result_columns) == 1:
        value = query_result_data[0][0]
        return {
            'type': 'single_value',
            'value': value,
            'message': f"**Answer**: {value}",
            'context': f"This is the result for: '{question}'",
            'sql_info': f"Query found exactly one result from the data."
        }
    
    # Small list (≤ 10 items)
    elif result_count <= 10 and len(result_columns) == 1:
        values = [str(row[0]) for row in query_result_data]
        return {
            'type': 'small_list',
            'count': result_count,
            'items': values,
            'message': f"**Found {result_count} results**:",
            'formatted_list': "\n".join([f"• {item}" for item in values])
        }
    
    # Medium list (≤ 50 items)
    elif result_count <= 50 and len(result_columns) <= 3:
        formatted_items = []
        for row in query_result_data:
            if len(result_columns) == 1:
                formatted_items.append(f"• {row[0]}")
            elif len(result_columns) == 2:
                formatted_items.append(f"• **{row[0]}**: {row[1]}")
            else:  # 3 columns
                formatted_items.append(f"• **{row[0]}** | {row[1]} | {row[2]}")
        
        return {
            'type': 'medium_list',
            'count': result_count,
            'columns': result_columns,
            'message': f"**Found {result_count} results**:",
            'formatted_list': "\n".join(formatted_items),
            'table_preview': query_result_data[:20]  # First 20 for table view
        }
    
    # Large result set - provide summary
    else:
        # Try to create meaningful summary
        summary_info = {
            'total_records': result_count,
            'columns': result_columns,
            'sample_data': query_result_data[:5]
        }
        
        # If it's a counting/aggregation query, highlight key numbers
        if any(col.lower() in ['count', 'total', 'sum', 'avg', 'max', 'min'] for col in result_columns):
            key_metrics = []
            for i, col in enumerate(result_columns):
                if col.lower() in ['count', 'total', 'sum', 'avg', 'max', 'min']:
                    values = [row[i] for row in query_result_data if row[i] is not None]
                    if values:
                        if col.lower() in ['count', 'total', 'sum']:
                            key_metrics.append(f"**{col}**: {sum(values):,}")
                        elif col.lower() == 'avg':
                            key_metrics.append(f"**{col}**: {sum(values)/len(values):.2f}")
                        else:
                            key_metrics.append(f"**{col}**: {max(values) if col.lower() == 'max' else min(values)}")
            
            if key_metrics:
                return {
                    'type': 'summary_with_metrics',
                    'count': result_count,
                    'metrics': key_metrics,
                    'message': f"**Analysis Results** ({result_count:,} records):",
                    'formatted_metrics': "\n".join(key_metrics),
                    'table_preview': query_result_data[:10]
                }
        
        return {
            'type': 'large_dataset',
            'count': result_count,
            'columns': result_columns,
            'message': f"**Large Result Set** ({result_count:,} records found)",
            'summary': f"Your query returned {result_count:,} records with {len(result_columns)} columns: {', '.join(result_columns)}",
            'table_preview': query_result_data[:20],
            'note': f"Showing first 20 results. Total: {result_count:,} records."
        }

# --- ADVANCED: Response Generator ---
def generate_advanced_natural_response(formatted_result, question, intent_data):
    """Generates natural language response from formatted results."""
    
    result_type = formatted_result['type']
    
    if result_type == 'no_results':
        return {
            'answer': formatted_result['message'],
            'suggestions': formatted_result['suggestions'],
            'format_type': 'no_results'
        }
    
    elif result_type == 'single_value':
        return {
            'answer': formatted_result['message'],
            'context': formatted_result['context'],
            'format_type': 'single_value'
        }
    
    elif result_type in ['small_list', 'medium_list']:
        answer = formatted_result['message'] + "\n\n" + formatted_result['formatted_list']
        
        # Add insights for lists
        if result_type == 'medium_list' and formatted_result['count'] > 5:
            answer += f"\n\n*Total of {formatted_result['count']} items found.*"
        
        return {
            'answer': answer,
            'count': formatted_result['count'],
            'format_type': result_type,
            'table_data': formatted_result.get('table_preview')
        }
    
    elif result_type == 'summary_with_metrics':
        answer = formatted_result['message'] + "\n\n" + formatted_result['formatted_metrics']
        if formatted_result['count'] > 10:
            answer += f"\n\n*Based on analysis of {formatted_result['count']:,} records.*"
        
        return {
            'answer': answer,
            'metrics': formatted_result['metrics'],
            'format_type': 'summary',
            'table_data': formatted_result.get('table_preview')
        }
    
    elif result_type == 'large_dataset':
        answer = formatted_result['message'] + "\n\n" + formatted_result['summary']
        if formatted_result.get('note'):
            answer += f"\n\n*{formatted_result['note']}*"
        
        return {
            'answer': answer,
            'count': formatted_result['count'],
            'format_type': 'large_dataset',
            'table_data': formatted_result.get('table_preview'),
            'full_data_available': True
        }
    
    return {
        'answer': "Results processed successfully.",
        'format_type': 'unknown'
    }

# --- ADVANCED: Query Validator ---
def validate_query_results(question, formatted_result, intent_data, safe_sql):
    """Validates if the query results actually answer the user's question."""
    
    validations = []
    warnings = []
    
    # Check if we got results when we expected them
    if formatted_result['type'] == 'no_results':
        if any(keyword in question.lower() for keyword in ['list', 'show', 'find', 'get']):
            warnings.append("No results found - this might mean the data doesn't contain what you're looking for.")
    
    # Check for potential data issues
    expected_type = intent_data.get('expected_result_type') if intent_data else None
    actual_type = formatted_result['type']
    
    if expected_type == 'Single_Value' and actual_type not in ['single_value']:
        warnings.append(f"Expected a single answer, but got {formatted_result.get('count', 0)} results.")
    
    elif expected_type == 'List' and actual_type in ['large_dataset']:
        validations.append(f"Found {formatted_result.get('count', 0)} results as expected.")
    
    # Check for SQL pattern matches
    if 'AI' in question and 'AI' not in safe_sql.upper():
        warnings.append("Your question mentions 'AI' but the search might have used different terms.")
    
    return {
        'validations': validations,
        'warnings': warnings,
        'confidence': 'high' if not warnings else 'medium' if len(warnings) == 1 else 'low'
    }

# --- MAIN VIEWS (upload_file remains the same) ---

def upload_file(request):
    if not request.session.session_key:
        request.session.create()
        logger.debug("Created new session for upload.")

    if request.method == 'POST':
        form = forms.UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                uploaded_file = request.FILES['file']
                file_extension = uploaded_file.name.split('.')[-1].lower()

                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file, on_bad_lines='skip')
                elif file_extension in ['xlsx', 'xls']:
                    df = pd.read_excel(uploaded_file)
                else:
                    messages.error(request, 'Unsupported file format. Please upload CSV or Excel files.')
                    logger.warning(f"Unsupported file format uploaded: {file_extension}")
                    return render(request, 'upload.html', {'form': form})

                if df.empty:
                    messages.error(request, 'The uploaded file is empty or could not be parsed.')
                    logger.warning("Uploaded file was empty or could not be parsed.")
                    return render(request, 'upload.html', {'form': form})

                con, table_name = get_duckdb_connection(request)
                logger.info(f"Loading data into DuckDB table: {table_name}")

                con.register('temp_df', df)
                con.execute(f"DROP TABLE IF EXISTS {table_name}")
                con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_df")
                con.unregister('temp_df')

                profiling_success = perform_enhanced_profiling(con, table_name, df, request)
                if not profiling_success:
                    messages.warning(request, 'File uploaded, but detailed data analysis failed. Some advanced features might be limited.')

                con.close()

                request.session['columns'] = df.columns.tolist()
                request.session['table_loaded'] = True
                if 'table_data' in request.session:
                    del request.session['table_data']
                if 'chat_history' in request.session:
                    del request.session['chat_history']

                messages.success(request, "File uploaded successfully and analyzed with advanced profiling.")
                logger.info("File uploaded and data analyzed successfully.")
                return redirect('TestAgent:askquestion')

            except Exception as e:
                error_msg = f'Error processing file: {str(e)}'
                messages.error(request, error_msg)
                logger.error(f"Upload error: {str(e)}", exc_info=True)
        else:
            messages.error(request, 'Please correct the errors below.')
            logger.warning(f"Upload form errors: {form.errors}")
    else:
        form = forms.UploadFileForm()
    return render(request, 'upload.html', {'form': form})

# --- ADVANCED ASK QUESTION VIEW ---
def askquestion(request):
    if not request.session.session_key:
        messages.error(request, "Session expired or not found. Please re-upload your data.")
        logger.warning("Session key missing in askquestion.")
        return redirect('TestAgent:upload')

    if not request.session.get('table_loaded', False):
        messages.error(request, "No file uploaded or data not loaded.")
        logger.info("User accessed askquestion without loaded data.")
        return redirect('TestAgent:upload')

    columns = request.session.get('columns', [])
    table_name = request.session.get('duckdb_table_name', None)
    table_metadata = request.session.get('table_metadata', {})

    if not table_name:
        messages.error(request, "Session data missing (table name). Please re-upload the file.")
        logger.error("Session data missing: duckdb_table_name")
        return redirect('TestAgent:upload')

    con = None
    try:
        con, _ = get_duckdb_connection(request)
        logger.debug(f"Using DuckDB table for query: {table_name}")

        table_exists_result = con.execute("""
            SELECT COUNT(*) AS count
            FROM information_schema.tables
            WHERE table_name = ?
        """, [table_name]).fetchone()

        if not table_exists_result or table_exists_result[0] == 0:
            messages.error(request, f"Data table '{table_name}' not found. Please re-upload the file.")
            logger.warning(f"Data table '{table_name}' not found in DuckDB.")
            request.session['table_loaded'] = False
            for key in ['duckdb_table_name', 'table_metadata', 'chat_history']:
                if key in request.session:
                    del request.session[key]
            return redirect('TestAgent:upload')

        row_count_result = con.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
        row_count = row_count_result[0] if row_count_result else 0
        data_points = row_count * len(columns)

        # Get sample data for context
        sample_df = con.execute(f"SELECT * FROM {table_name} LIMIT 2").fetchdf()
        sample_data_str = "No data available."
        if not sample_df.empty:
            max_col_width = 50
            truncated_sample_df = sample_df.copy()
            for col in truncated_sample_df.columns:
                if truncated_sample_df[col].dtype == "object":
                    truncated_sample_df[col] = truncated_sample_df[col].apply(
                        lambda x: (str(x)[:max_col_width] + '...') if isinstance(x, str) and len(str(x)) > max_col_width else str(x)
                    )
            sample_data_str = truncated_sample_df.to_string(index=False)

    except Exception as e:
        error_msg = f"Error retrieving table information: {e}"
        messages.error(request, error_msg)
        logger.error(error_msg, exc_info=True)
        return redirect('TestAgent:upload')
    finally:
        if con:
            try:
                con.close()
            except Exception as close_error:
                logger.warning(f"Error closing DuckDB connection: {close_error}")

    if request.method == 'POST':
        form = forms.AskQuestionForm(request.POST)
        if form.is_valid():
            question = form.cleaned_data['question']
            con = None
            
            try:
                # === ADVANCED PROCESSING PIPELINE ===
                
                # STEP 1: Enhanced Intent Classification
                logger.info(f"Processing advanced query: {question}")
                intent_data = classify_question_intent(question, table_metadata)
                
                # STEP 2: Conversational History
                chat_history = request.session.get('chat_history', [])
                history_str = ""
                if chat_history:
                    history_items = []
                    for turn in chat_history[-3:]:
                        history_items.append(f"Q: {turn['question']}\nSQL: {turn['sql']}")
                    history_str = "\n---\nRecent Context:\n" + "\n".join(history_items)

                # STEP 3: Enhanced SQL Generation
                schema_info_lines = []
                for col_name, col_meta in table_metadata.get('columns', {}).items():
                    line = f"  - {col_name} (Type: {col_meta.get('dtype_duckdb', 'unknown')}, Semantic: {col_meta.get('semantic_type', 'unknown')})"
                    if col_meta.get('sample_values'):
                        sample_str = ", ".join(map(str, col_meta['sample_values'][:3]))
                        line += f" [Examples: {sample_str}]"
                    if col_meta.get('top_values'):
                        top_vals = [str(tv['value']) for tv in col_meta['top_values'][:2]]
                        line += f" [Common: {', '.join(top_vals)}]"
                    schema_info_lines.append(line)
                
                enhanced_schema_info = "\n".join(schema_info_lines) if schema_info_lines else "Basic columns: " + ", ".join(columns)

                intent_context = ""
                if intent_data:
                    intent_context = f"""
Query Analysis:
- Intent: {intent_data.get('intent_category', 'Unknown')}
- Complexity: {intent_data.get('query_complexity', 'Unknown')}
- Expected Result: {intent_data.get('expected_result_type', 'Unknown')}
- Approach: {'; '.join(intent_data.get('suggested_approach', []))}
"""

                sql_generation_prompt = f"""
You are an expert SQL analyst for DuckDB. Generate a precise, optimized query.

Table: '{table_name}'
Schema with Context:
{enhanced_schema_info}

{intent_context}

{history_str}

Current Question: {question}

Requirements:
1. Generate ONLY a valid DuckDB SQL query - no explanations, no markdown
2. Use proper data type casting when needed (e.g., CAST(column AS DATE))
3. For text searches, use ILIKE for case-insensitive matching
4. For complex filters, consider multiple conditions
5. Add appropriate LIMIT if result set might be very large
6. Optimize for the expected result type

Query:"""

                # Generate SQL
                api_key = getattr(settings, 'NVIDIA_NIM_API_KEY', None)
                api_endpoint = getattr(settings, 'NVIDIA_NIM_API_ENDPOINT', 'https://integrate.api.nvidia.com/v1/chat/completions')

                if not api_key:
                    raise ValueError("NVIDIA_NIM_API_KEY not found in Django settings.")

                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                
                sql_payload = {
                    "model": getattr(settings, 'NVIDIA_NIM_MODEL', 'meta/llama-3.1-405b-instruct'),
                    "messages": [
                        {"role": "system", "content": "You are an expert SQL generator. Output only clean SQL queries without markdown or explanations."},
                        {"role": "user", "content": sql_generation_prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 600,
                    "top_p": 1,
                    "stream": False
                }

                logger.debug("Generating advanced SQL query...")
                sql_response = requests.post(api_endpoint, headers=headers, json=sql_payload, timeout=120)
                sql_response.raise_for_status()
                sql_response_data = sql_response.json()
                generated_sql_raw = sql_response_data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

                if not generated_sql_raw:
                    raise ValueError("LLM returned an empty SQL query.")

                # Clean and validate SQL
                safe_sql = generated_sql_raw.strip()
                if safe_sql.startswith("```sql"):
                    safe_sql = safe_sql[6:].lstrip()
                if safe_sql.startswith("```"):
                    safe_sql = safe_sql[3:].lstrip()
                if safe_sql.endswith("```"):
                    safe_sql = safe_sql[:-3].rstrip()
                safe_sql = safe_sql.strip()

                # Security validation
                upper_sql = safe_sql.upper()
                if not (upper_sql.startswith("SELECT") or upper_sql.startswith("WITH")):
                    raise ValueError("Invalid SQL query generated.")

                disallowed_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
                for keyword in disallowed_keywords:
                    if f" {keyword} " in f" {upper_sql} ":
                        raise ValueError(f"SQL contains disallowed keyword: {keyword}")

                logger.info(f"Generated SQL: {safe_sql}")

                # STEP 4: Execute SQL with Smart Retry
                con, _ = get_duckdb_connection(request)
                try:
                    result_cursor = con.execute(safe_sql)
                    query_result_data = result_cursor.fetchall()
                    result_columns = [desc[0] for desc in result_cursor.description]
                except duckdb.Error as e:
                    logger.warning(f"SQL execution failed: {e}")
                    # Smart retry for common issues
                    if "Binder Error" in str(e):
                        retry_prompt = f"""
The query failed with: {e}
Original query: {safe_sql}
User question: {question}

Fix this query by adding proper type casting or correcting column references.
Return ONLY the corrected SQL:"""

                        sql_payload["messages"] = [
                            {"role": "system", "content": "Fix the SQL query. Return only corrected SQL."},
                            {"role": "user", "content": retry_prompt}
                        ]
                        
                        retry_response = requests.post(api_endpoint, headers=headers, json=sql_payload, timeout=60)
                        retry_response.raise_for_status()
                        corrected_sql = retry_response.json().get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                        
                        # Clean corrected SQL
                        if corrected_sql.startswith("```sql"):
                            corrected_sql = corrected_sql[6:].lstrip()
                        if corrected_sql.endswith("```"):
                            corrected_sql = corrected_sql[:-3].rstrip()
                        safe_sql = corrected_sql.strip()
                        
                        logger.info(f"Retrying with corrected SQL: {safe_sql}")
                        result_cursor = con.execute(safe_sql)
                        query_result_data = result_cursor.fetchall()
                        result_columns = [desc[0] for desc in result_cursor.description]
                    else:
                        raise e

                # STEP 5: Save to conversation history
                chat_history.append({'question': question, 'sql': safe_sql})
                request.session['chat_history'] = chat_history[-10:]  # Keep last 10

                # STEP 6: Advanced Result Formatting
                formatted_result = format_advanced_response(question, query_result_data, result_columns, intent_data, safe_sql)
                
                # STEP 7: Generate Natural Language Response
                natural_response = generate_advanced_natural_response(formatted_result, question, intent_data)
                
                # STEP 8: Validate Results
                validation_result = validate_query_results(question, formatted_result, intent_data, safe_sql)

                # STEP 9: Create Enhanced Response Context
                response_context = {
                    'main_answer': natural_response['answer'],
                    'format_type': natural_response['format_type'],
                    'result_count': formatted_result.get('count', 0),
                    'confidence': validation_result['confidence'],
                    'warnings': validation_result['warnings'],
                    'validations': validation_result['validations'],
                    'intent_info': intent_data,
                    'query_info': {
                        'sql': safe_sql,
                        'execution_time': 'Success',
                        'result_type': formatted_result['type']
                    }
                }

                # Generate follow-up suggestions
                followup_suggestions = generate_followup_suggestions(question, formatted_result, intent_data, table_metadata)

                # Prepare table data for display
                con, _ = get_duckdb_connection(request)
                if natural_response.get('table_data'):
                    # Show query results as table
                    display_df = pd.DataFrame(natural_response['table_data'], columns=result_columns)
                else:
                    # Show general data preview
                    display_df = con.execute(f"SELECT * FROM {table_name} LIMIT 50").fetchdf()
                
                table_html = display_df.to_html(classes='cyber-table table table-striped', escape=False, index=False)
                con.close()

                return render(request, 'ask.html', {
                    'form': form,
                    'response': response_context['main_answer'],
                    'response_context': response_context,
                    'followup_suggestions': followup_suggestions,
                    'table_html': table_html,
                    'columns': columns,
                    'table_shape': (row_count, len(columns)),
                    'data_points': data_points,
                    'advanced_mode': True
                })

            except requests.exceptions.RequestException as req_e:
                error_msg = f"Error communicating with AI service: {req_e}"
                logger.error(f"API Error: {req_e}", exc_info=True)
            except ValueError as val_e:
                error_msg = f"Error processing your request: {val_e}"
                logger.error(f"Processing Error: {val_e}", exc_info=True)
            except Exception as e:
                error_msg = f"An unexpected error occurred. Please try again."
                logger.critical(f"Unexpected Error: {e}", exc_info=True)
            finally:
                if con:
                    try:
                        con.close()
                    except Exception:
                        pass

            # Error handling - show preview table
            try:
                con, _ = get_duckdb_connection(request)
                preview_df = con.execute(f"SELECT * FROM {table_name} LIMIT 50").fetchdf()
                table_html = preview_df.to_html(classes='cyber-table table table-striped', escape=False, index=False)
                con.close()
            except Exception:
                table_html = "<p>Error generating data preview.</p>"

            return render(request, 'ask.html', {
                'form': form,
                'error': error_msg,
                'table_html': table_html,
                'columns': columns,
                'table_shape': (row_count, len(columns)),
                'data_points': data_points
            })

    # GET request - show initial form
    form = forms.AskQuestionForm()
    try:
        con, _ = get_duckdb_connection(request)
        preview_df = con.execute(f"SELECT * FROM {table_name} LIMIT 50").fetchdf()
        table_html = preview_df.to_html(classes='cyber-table table table-striped', escape=False, index=False)
        con.close()
    except Exception as e:
        logger.error(f"Error generating preview table: {e}")
        table_html = "<p>Error loading data preview.</p>"

    return render(request, 'ask.html', {
        'form': form,
        'table_html': table_html,
        'columns': columns,
        'table_shape': (row_count, len(columns)),
        'data_points': data_points,
        'table_metadata': table_metadata  # Pass metadata for advanced features
    })

# --- ADVANCED: Follow-up Suggestions Generator ---
def generate_followup_suggestions(question, formatted_result, intent_data, table_metadata):
    """Generates intelligent follow-up question suggestions."""
    
    suggestions = []
    
    # Based on intent category
    intent_category = intent_data.get('intent_category') if intent_data else None
    
    if intent_category == 'Text_Search':
        suggestions.extend([
            "Count how many results contain specific keywords",
            "Group results by category or type",
            "Find the most common themes in the results"
        ])
    
    elif intent_category == 'Count_Aggregation':
        suggestions.extend([
            "Break down the count by different categories",
            "Show the trend over time periods",
            "Compare with other related metrics"
        ])
    
    elif intent_category == 'Filtering':
        suggestions.extend([
            "Expand the filter criteria to see more results",
            "Apply additional filters to narrow down further",
            "Compare filtered results with the full dataset"
        ])
    
    # Based on result type
    result_type = formatted_result['type']
    
    if result_type == 'no_results':
        suggestions.extend([
            "Try searching with broader terms",
            "Check what data is actually available",
            "Look for similar or related concepts"
        ])
    
    elif result_type in ['small_list', 'medium_list']:
        suggestions.extend([
            "Get more details about specific items",
            "Analyze patterns in the results",
            "Compare these results with other criteria"
        ])
    
    elif result_type == 'large_dataset':
        suggestions.extend([
            "Summarize the results by key categories",
            "Find the top or most significant items",
            "Apply filters to focus on specific areas"
        ])
    
    # Based on available columns
    available_columns = table_metadata.get('columns', {})
    categorical_cols = [col for col, meta in available_columns.items() 
                       if meta.get('semantic_type') == 'categorical']
    
    if categorical_cols:
        col_name = categorical_cols[0]
        suggestions.append(f"Analyze results by {col_name}")
    
    # Remove duplicates and limit
    unique_suggestions = list(dict.fromkeys(suggestions))[:5]
    
    return unique_suggestions

def clear_table(request):
    """Clear all session data related to the table."""
    session_keys_to_clear = ['table_loaded', 'columns', 'duckdb_table_name', 'table_data', 'table_metadata', 'chat_history']
    for key in session_keys_to_clear:
        if key in request.session:
            del request.session[key]

    messages.success(request, "Table data cleared successfully.")
    logger.info("User cleared table data.")
    return redirect('TestAgent:upload')