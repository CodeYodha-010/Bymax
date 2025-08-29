
# Bymax

> **Note:** This project is currently in the early stages of development. Active coding and feature implementation have not yet begun. The repository serves as a foundation for future work, and contributions or feedback are welcome as the project evolves.

Baymax is an AI agent designed for large data analysis, capable of automating up to 40% of a data analyst's tasks. It provides an intuitive interface and leverages AI to assist with data exploration, visualization, and reporting.

## Enhanced Agent with GPTOSS 120B Model Integration

The agent has been enhanced with GPTOSS 120B model integration for improved NLP and JSON generation:

### Key Enhancements

1. **JSON Prompt Conversion**: User questions are converted into structured JSON format
2. **GPTOSS Intent Classification**: Uses GPTOSS 120B model for better intent classification
3. **Structured Query Generation**: GPTOSS converts natural language to structured queries
4. **Natural Explanations**: GPTOSS generates human-like explanations of results
5. **Enhanced Data Analysis**: Structured data guides the analysis process
6. **Support for Complex Queries**: Handles filters, aggregations, grouping, sorting, and limits
7. **Detailed Explanations**: Provides comprehensive explanations of results
8. **Benefit-Oriented Responses**: Explains the value and meaning of results to users

## Features
- Automated data analysis and reporting
- User-friendly web interface
- AI-powered insights and recommendations
- Customizable for various data sources
- JSON prompt conversion for accurate interpretation
- Support for complex queries with filters, aggregations, and grouping
- Enhanced intent classification using structured data

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/CodeYodha-010/Bymax.git
   cd Bymax
   ```

2. **Set up a Python virtual environment (recommended):**
   ```sh
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # Or
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is missing, install Django:)*
   ```sh
   pip install django
   ```

4. **Apply migrations:**
   ```sh
   python Baymax/manage.py migrate
   ```

5. **Run the development server:**
   ```sh
   python Baymax/manage.py runserver
   ```

6. **Access the app:**
   Open your browser and go to `http://127.0.0.1:8000/`

## Project Structure

```
Baymax/
├── Baymax/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   ├── views.py
│   └── wsgi.py
├── db.sqlite3
├── manage.py
├── static/
│   └── baymax.png
├── templates/
│   └── baymax.html
```

## Deep Overview

Baymax is built using Django, a robust Python web framework. The main interface (`baymax.html`) provides a clean and modern UI for users to interact with the AI agent. The backend leverages Django views and models to process data and deliver insights. Static assets and templates are organized for easy customization.

### How It Works
1. **User Interaction:** Users access Baymax via a web browser and upload or select data for analysis.
2. **GPTOSS Processing:** User questions are processed by the GPTOSS 120B model for intent classification and structured query generation.
3. **AI Processing:** The backend processes the structured data, applies AI models, and generates insights.
4. **Detailed Explanation:** Results are explained in detail with reasoning and benefits using GPTOSS-generated explanations.
5. **Visualization:** Results are displayed in a user-friendly format, with options for further exploration.

### Customization
You can extend Baymax by adding new Django apps, integrating additional AI models, or connecting to external data sources.

## GPTOSS 120B Model Integration

The agent now uses the GPTOSS 120B model for NLP and JSON generation tasks:

1. **Intent Classification**: GPTOSS classifies question intents more accurately than rule-based systems
2. **Structured Query Generation**: Converts natural language to structured queries for precise data analysis
3. **Natural Explanations**: Generates human-like explanations of results with reasoning and benefits

To use the GPTOSS integration, you need a GPTOSS API key in your `.env` file.

## Detailed Explanations

The enhanced agent now provides comprehensive explanations for all responses:

1. **Why this answer?** - Explains the reasoning behind the result
2. **What does this mean for you?** - Explains the benefits and value of the information
3. **How I calculated this** - Describes the methodology used
4. **Tips for better results** - Provides guidance for more effective queries

## Testing Enhancements

To test the JSON prompt conversion enhancement, you can use the Django management command:

```sh
python Baymax/manage.py test_json_converter "Your question here"
```

To test the GPTOSS integration:

```sh
python Baymax/test_gptoss_integration.py
```

## License
See the LICENSE file for details.
