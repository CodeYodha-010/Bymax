# apps/TestAgent/views.py - ADVANCED VERSION
import os
import hashlib
import logging
import pandas as pd
import duckdb
import httpx
import json
import re
import asyncio
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from django.urls import reverse
from . import forms
from . import prompts
from asgiref.sync import sync_to_async

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

import tempfile

# --- Enhanced Data Profiling Function ---
def perform_enhanced_profiling(con, table_name, request, df=None):
    """Analyzes uploaded data and stores metadata. Handles both DataFrame and DB table sources."""
    # 1. Get basic info and column types
    if df is not None:
        columns = df.columns.tolist()
        total_rows = len(df)
        pandas_dtypes = {col: str(df[col].dtype) for col in columns}
    else:
        columns = [desc[0] for desc in con.execute(f"DESCRIBE {table_name}").fetchall()]
        total_rows = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        pandas_dtypes = {}  # Not available without pandas DataFrame

    table_metadata = {
        'basic_info': {'total_rows': total_rows, 'total_columns': len(columns), 'column_names': columns},
        'columns': {}
    }
    logger.info(f"Starting optimized enhanced profiling for table: {table_name}")

    try:
        duckdb_types_results = con.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'").fetchall()
        duckdb_types = {name: dtype for name, dtype in duckdb_types_results}
        numeric_duckdb_types = ['BIGINT', 'INTEGER', 'SMALLINT', 'TINYINT', 'UBIGINT', 'UINTEGER', 'USMALLINT', 'UTINYINT', 'DECIMAL', 'DOUBLE', 'FLOAT', 'REAL']

        # 2. Build and execute a single query for all stats
        stat_queries = []
        for col in columns:
            safe_col = f'"{col}"'
            is_numeric = duckdb_types.get(col) in numeric_duckdb_types

            numeric_part = f"""
                MIN({safe_col}) as min_val, MAX({safe_col}) as max_val, AVG({safe_col}) as avg_val,
                STDDEV_SAMP({safe_col}) as stddev_val, COUNT({safe_col}) as non_null_count, NULL as top_values
            """ if is_numeric else f"""
                NULL as min_val, NULL as max_val, NULL as avg_val,
                NULL as stddev_val, NULL as non_null_count,
                (SELECT list(STRUCT_PACK(value := {safe_col}, count := count)) FROM (SELECT {safe_col}, COUNT(*) as count FROM {table_name} WHERE {safe_col} IS NOT NULL GROUP BY {safe_col} ORDER BY count DESC LIMIT 10)) as top_values
            """

            query = f"""
            SELECT
                '{col.replace("'", "''")}' as column_name,
                (SELECT list(DISTINCT {safe_col}) FROM {table_name} WHERE {safe_col} IS NOT NULL LIMIT 5) as sample_values,
                COUNT(DISTINCT {safe_col}) as distinct_count,
                {numeric_part}
            FROM {table_name}
            """
            stat_queries.append(query)

        if not stat_queries: return False

        full_query = "\nUNION ALL\n".join(stat_queries)
        all_stats = con.execute(full_query).fetchall()
        stats_map = {row[0]: row for row in all_stats}

        # 3. Populate metadata from the single query result
        for col in columns:
            stats_row = stats_map.get(col)
            if not stats_row: continue

            col_metadata = {
                'name': col, 'dtype_pandas': pandas_dtypes.get(col), 'dtype_duckdb': duckdb_types.get(col),
                'semantic_type': 'unknown', 'stats': {}, 'distinct_count': stats_row[2],
                'top_values': [{'value': r['value'], 'count': r['count']} for r in stats_row[8]] if stats_row[8] else [],
                'sample_values': [v for v in stats_row[1]]
            }

            # 4. Perform semantic type inference
            col_name_lower = col.lower()
            if 'id' in col_name_lower or 'identifier' in col_name_lower: col_metadata['semantic_type'] = 'identifier'
            elif any(kw in col_name_lower for kw in ['date', 'time', 'year', 'month']): col_metadata['semantic_type'] = 'temporal'
            elif df is not None and duckdb_types.get(col) == 'VARCHAR' and df[col].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}', na=False).any():
                col_metadata['semantic_type'] = 'temporal'
            elif any(kw in col_name_lower for kw in ['count', 'number', 'amount', 'price', 'score', 'rate']): col_metadata['semantic_type'] = 'numerical'
            elif any(kw in col_name_lower for kw in ['category', 'bucket', 'type', 'status', 'department', 'organisation', 'title', 'technology', 'problem', 'statement', 'description']):
                col_metadata['semantic_type'] = 'categorical'

            is_numeric_final = (df is not None and pd.api.types.is_numeric_dtype(df[col])) or col_metadata['semantic_type'] == 'numerical'
            if is_numeric_final:
                col_metadata['semantic_type'] = 'numerical'
                if stats_row and stats_row[3] is not None:
                    col_metadata['stats'] = {'min': stats_row[3], 'max': stats_row[4], 'mean': stats_row[5], 'stddev': stats_row[6], 'count': stats_row[7]}
            elif col_metadata['semantic_type'] == 'unknown':
                col_metadata['semantic_type'] = 'text'

            table_metadata['columns'][col] = col_metadata

        request.session['table_metadata'] = table_metadata
        logger.info(f"Optimized enhanced profiling complete for table: {table_name}")
        return True
    except Exception as e:
        logger.error(f"Error during optimized enhanced profiling for table {table_name}: {e}", exc_info=True)
        return False

# --- ASYNC NVIDIA API Caller ---
async def call_nvidia_api_async(payload, timeout=60):
    """Asynchronously calls the NVIDIA NIM API."""
    api_key = getattr(settings, 'NVIDIA_NIM_API_KEY', None)
    api_endpoint = getattr(settings, 'NVIDIA_NIM_API_ENDPOINT', 'https://integrate.api.nvidia.com/v1/chat/completions')
    if not api_key:
        raise ValueError("NVIDIA_NIM_API_KEY not found in Django settings.")

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(api_endpoint, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            response_data = response.json()
            return response_data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error in async NVIDIA API call: {e}")
            return None


# --- ADVANCED: Intent Classification Function ---
async def classify_question_intent_async(question, table_metadata):
    """Asynchronously classifies question intent."""
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

    intent_prompt = prompts.INTENT_CLASSIFICATION_PROMPT.format(
        schema_info=schema_info_str,
        question=question
    )

    payload = {
        "model": getattr(settings, 'NVIDIA_NIM_MODEL', 'meta/llama-3.1-405b-instruct'),
        "messages": [
            {"role": "system", "content": "You are an expert data analyst. Respond only with valid JSON for query classification."},
            {"role": "user", "content": intent_prompt}
        ],
        "temperature": 0.1, "max_tokens": 700, "top_p": 1, "stream": False
    }

    try:
        intent_json_str = await call_nvidia_api_async(payload)
        if intent_json_str:
            intent_data = json.loads(intent_json_str)
            logger.debug(f"Advanced intent classified: {intent_data}")
            return intent_data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding intent JSON: {e}")
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

                con, table_name = get_duckdb_connection(request)
                profiling_success = False
                columns = []

                if file_extension == 'csv':
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_f:
                        for chunk in uploaded_file.chunks():
                            temp_f.write(chunk)
                        temp_file_path = temp_f.name

                    logger.info(f"Loading CSV from {temp_file_path} into DuckDB table: {table_name}")
                    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{temp_file_path}')")
                    os.remove(temp_file_path)

                    profiling_success = perform_enhanced_profiling(con, table_name, request)

                elif file_extension in ['xlsx', 'xls']:
                    df = pd.read_excel(uploaded_file)
                    if df.empty:
                        raise ValueError("The uploaded Excel file is empty or could not be parsed.")

                    logger.info(f"Loading Excel data into DuckDB table: {table_name}")
                    con.execute(f"DROP TABLE IF EXISTS {table_name}")
                    con.register('temp_df', df)
                    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_df")
                    con.unregister('temp_df')

                    profiling_success = perform_enhanced_profiling(con, table_name, request, df=df)
                else:
                    messages.error(request, 'Unsupported file format. Please upload CSV or Excel files.')
                    return render(request, 'upload.html', {'form': form})

                if not profiling_success:
                    messages.warning(request, 'File uploaded, but detailed data analysis failed. Some advanced features might be limited.')

                # Get column names from the created table
                columns = [desc[0] for desc in con.execute(f"DESCRIBE {table_name}").fetchall()]
                con.close()

                request.session['columns'] = columns
                request.session['table_loaded'] = True
                if 'table_data' in request.session: del request.session['table_data']
                if 'chat_history' in request.session: del request.session['chat_history']

                messages.success(request, "File uploaded and analyzed successfully.")
                logger.info("File uploaded and data analyzed successfully.")
                return redirect('TestAgent:askquestion')

            except Exception as e:
                error_msg = f'Error processing file: {str(e)}'
                messages.error(request, error_msg)
                logger.error(f"Upload error: {str(e)}", exc_info=True)
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = forms.UploadFileForm()
    return render(request, 'upload.html', {'form': form})

# --- ADVANCED ASK QUESTION VIEW ---
async def askquestion(request):
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

        table_exists_result = await sync_to_async(con.execute)("""
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

        row_count_result = await sync_to_async(con.execute)(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
        row_count = row_count_result[0] if row_count_result else 0
        data_points = row_count * len(columns)

    except Exception as e:
        error_msg = f"Error retrieving table information: {e}"
        messages.error(request, error_msg)
        logger.error(error_msg, exc_info=True)
        return redirect('TestAgent:upload')
    finally:
        if con:
            try:
                await sync_to_async(con.close)()
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
                intent_data = await classify_question_intent_async(question, table_metadata)
                
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

                sql_generation_prompt = prompts.SQL_GENERATION_PROMPT.format(
                    table_name=table_name,
                    schema_info=enhanced_schema_info,
                    intent_context=intent_context,
                    history_str=history_str,
                    question=question
                )

                sql_payload = {
                    "model": getattr(settings, 'NVIDIA_NIM_MODEL', 'meta/llama-3.1-405b-instruct'),
                    "messages": [
                        {"role": "system", "content": "You are an expert SQL generator. Output only clean SQL queries without markdown or explanations."},
                        {"role": "user", "content": sql_generation_prompt}
                    ],
                    "temperature": 0.1, "max_tokens": 600, "top_p": 1, "stream": False
                }

                logger.debug("Generating advanced SQL query...")
                generated_sql_raw = await call_nvidia_api_async(sql_payload, timeout=120)

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
                    result_cursor = await sync_to_async(con.execute)(safe_sql)
                    query_result_data = await sync_to_async(result_cursor.fetchall)()
                    result_columns = [desc[0] for desc in result_cursor.description]
                except duckdb.Error as e:
                    logger.warning(f"SQL execution failed: {e}")
                    # Smart retry for common issues
                    if "Binder Error" in str(e):
                        retry_prompt = prompts.SQL_RETRY_PROMPT.format(
                            error=e,
                            sql=safe_sql,
                            question=question
                        )
                        sql_payload["messages"] = [
                            {"role": "system", "content": "Fix the SQL query. Return only corrected SQL."},
                            {"role": "user", "content": retry_prompt}
                        ]
                        
                        corrected_sql_raw = await call_nvidia_api_async(sql_payload, timeout=60)
                        if not corrected_sql_raw:
                            raise e

                        # Clean corrected SQL
                        if corrected_sql_raw.startswith("```sql"):
                            corrected_sql_raw = corrected_sql_raw[6:].lstrip()
                        if corrected_sql_raw.endswith("```"):
                            corrected_sql_raw = corrected_sql_raw[:-3].rstrip()
                        safe_sql = corrected_sql_raw.strip()
                        
                        logger.info(f"Retrying with corrected SQL: {safe_sql}")
                        result_cursor = await sync_to_async(con.execute)(safe_sql)
                        query_result_data = await sync_to_async(result_cursor.fetchall)()
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
                    display_df = await sync_to_async(con.execute)(f"SELECT * FROM {table_name} LIMIT 50").fetchdf()
                
                table_html = display_df.to_html(classes='cyber-table table table-striped', escape=False, index=False)
                await sync_to_async(con.close)()

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

            except (httpx.RequestError, ValueError) as e:
                error_msg = f"Error processing your request: {e}"
                logger.error(f"Processing Error: {e}", exc_info=True)
            except Exception as e:
                error_msg = f"An unexpected error occurred. Please try again."
                logger.critical(f"Unexpected Error: {e}", exc_info=True)
            finally:
                if con:
                    try:
                        await sync_to_async(con.close)()
                    except Exception:
                        pass

            # Error handling - show preview table
            try:
                con, _ = get_duckdb_connection(request)
                preview_df = await sync_to_async(con.execute)(f"SELECT * FROM {table_name} LIMIT 50").fetchdf()
                table_html = preview_df.to_html(classes='cyber-table table table-striped', escape=False, index=False)
                await sync_to_async(con.close)()
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
        preview_df = await sync_to_async(con.execute)(f"SELECT * FROM {table_name} LIMIT 50").fetchdf()
        table_html = preview_df.to_html(classes='cyber-table table table-striped', escape=False, index=False)
        await sync_to_async(con.close)()
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