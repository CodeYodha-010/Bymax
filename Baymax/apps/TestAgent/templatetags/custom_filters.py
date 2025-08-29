
from django import template
from django.template.defaultfilters import stringfilter

register = template.Library()

@register.filter
@stringfilter
def split(value, delimiter):
    """Splits the string by the given delimiter."""
    return value.split(delimiter)

@register.filter
@stringfilter
def get_chart_icon(analysis_type):
    """Return the appropriate Font Awesome icon for analysis type."""
    icons = {
        'comprehensive': 'bar-chart',
        'statistical': 'calculator',
        'correlation': 'link',
        'trend': 'line-chart',
        'anomaly': 'exclamation-triangle',
        'distribution': 'chart-pie',
        'comparative': 'balance-scale',
        'predictive': 'crystal-ball'
    }
    return icons.get(analysis_type, 'chart-bar')

@register.filter
@stringfilter
def get_type_description(analysis_type):
    """Return a human-readable description for analysis type."""
    descriptions = {
        'comprehensive': 'Complete analysis',
        'statistical': 'Statistical measures',
        'correlation': 'Relationship analysis',
        'trend': 'Pattern detection',
        'anomaly': 'Outlier detection',
        'distribution': 'Data distribution',
        'comparative': 'Cross-section analysis',
        'predictive': 'Future insights'
    }
    return descriptions.get(analysis_type, 'Analysis')

@register.filter
def format_number(value, decimals=2):
    """Format a number with specified decimal places."""
    try:
        return f"{float(value):,.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)

@register.filter
def format_percentage(value, decimals=1):
    """Format a decimal as percentage."""
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (ValueError, TypeError):
        return str(value)

@register.filter
def get_dict_value(dictionary, key):
    """Get value from dictionary by key."""
    return dictionary.get(key, '')

@register.filter
def truncate_text(text, max_length=50):
    """Truncate text to specified length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + '...'

@register.filter
def format_date(date_obj, format_string='%Y-%m-%d'):
    """Format date object."""
    try:
        return date_obj.strftime(format_string)
    except:
        return str(date_obj)

@register.filter
def is_list(value):
    """Check if value is a list."""
    return isinstance(value, list)

@register.filter
def is_dict(value):
    """Check if value is a dictionary."""
    return isinstance(value, dict)

@register.filter
def get_list_item(lst, index):
    """Get item from list by index."""
    try:
        return lst[index]
    except (IndexError, TypeError):
        return None

@register.filter
def json_length(value):
    """Get length of JSON-like structure."""
    if isinstance(value, (list, tuple)):
        return len(value)
    elif isinstance(value, dict):
        return len(value)
    return 0

@register.filter
def safe_json(value):
    """Convert value to JSON string safely."""
    import json
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)

@register.filter
def analysis_status_color(status):
    """Return color class based on analysis status."""
    colors = {
        'pending': 'warning',
        'processing': 'info',
        'completed': 'success',
        'failed': 'danger'
    }
    return colors.get(status, 'secondary')

@register.filter
def analysis_type_badge(analysis_type):
    """Return badge class for analysis type."""
    badges = {
        'comprehensive': 'primary',
        'statistical': 'secondary',
        'correlation': 'success',
        'trend': 'info',
        'anomaly': 'warning',
        'distribution': 'dark',
        'comparative': 'light',
        'predictive': 'danger'
    }
    return badges.get(analysis_type, 'secondary')
