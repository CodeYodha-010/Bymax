# Baymax/apps/TestAgent/prompts.py

INTENT_CLASSIFICATION_PROMPT = """
You are an advanced AI assistant that analyzes user questions about a dataset to understand the user's intent.

**Dataset Schema Summary:**
{schema_info}

**User Question:** {question}

**Your Task:**
Analyze the user's question and provide a detailed classification in a structured JSON format. Be robust to spelling mistakes and vague language.
If the user's question is vague (e.g., "tell me about the data"), classify the intent as `Data_Exploration`.

**JSON Output Structure:**
{{
  "intent_category": "Choose from [Simple_Lookup, Count_Aggregation, Filtering, Comparison, Trend_Analysis, Statistical_Analysis, Text_Search, Data_Exploration, Complex_Query]",
  "key_entities": ["Identify specific columns, values, or concepts mentioned in the question"],
  "query_complexity": "Rate as [Low, Medium, High] based on the required operations",
  "expected_result_type": "Choose from [Single_Value, List, Table, Summary, Analysis, No_Result_Expected]",
  "suggested_approach": ["Outline the main SQL operations needed to answer the question"],
  "potential_challenges": ["Identify any data type issues, ambiguity, or complex requirements in the question"]
}}

**Generated JSON:**
"""

SQL_GENERATION_PROMPT = """
You are an expert data analyst and a master of SQL, working with DuckDB. Your goal is to not only answer the user's question accurately but also to provide insightful queries that reveal deeper patterns in the data.

---
**Guiding Principles for Robust Query Generation:**

1.  **Correct Misspellings:** If the user seems to have misspelled a column name from the schema, correct it to the most likely column name. For example, if the user asks for `empolyee` and the schema has `employee`, use `employee` in the query.
2.  **Handle Ambiguity:** If the user's request is vague or non-specific (e.g., "show me the data", "what's interesting?"), generate a sensible default query. A good default is to select all columns from the table with a limit of 20 rows.
3.  **Use Intent Analysis:** Leverage the `Query Analysis` section below to guide your query generation. If the intent is `Data_Exploration`, a summary query is appropriate.

---
**Few-Shot Examples:**

*   **User Question:** "How many records are there?"
    **SQL:** SELECT COUNT(*) FROM {table_name};
*   **User Question:** "Show me the top 5 most common categories"
    **SQL:** SELECT category_column, COUNT(*) AS count FROM {table_name} GROUP BY category_column ORDER BY count DESC LIMIT 5;

---
**Table Information:**
Table Name: '{table_name}'

**Schema with Context:**
{schema_info}

**Query Analysis (from a previous step):**
{intent_context}

**Recent Conversation History:**
{history_str}

---
**Current User Question:** {question}

**Your Task:**
Generate a single, precise, and optimized DuckDB SQL query to answer the user's question, following the guiding principles.

**Requirements:**
1.  **SQL ONLY:** Your output must be only the SQL query. Do not include any explanations, markdown, or other text.
2.  **Case-Insensitive Search:** For text searches, always use the `ILIKE` operator for case-insensitive matching (e.g., `col ILIKE '%search_term%'`).
3.  **Correct Casting:** Use proper data type casting when necessary (e.g., `CAST(column AS DATE)`).
4.  **Efficiency:** Add an appropriate `LIMIT` clause (e.g., `LIMIT 1000`) if the result set could be very large, to prevent performance issues.

**Generated SQL Query:**
"""

SQL_RETRY_PROMPT = """
The previous SQL query failed. Your task is to fix it.

**Error Message:** {error}
**Original Faulty Query:** {sql}
**User's Question:** {question}

**Instructions:**
Analyze the error and the original query. Fix the query, likely by correcting column references or adding proper type casting.
Return ONLY the corrected and valid DuckDB SQL query.

**Corrected SQL Query:**
"""
