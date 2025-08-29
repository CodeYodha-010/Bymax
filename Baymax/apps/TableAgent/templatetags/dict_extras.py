# from django import template

# register = template.Library()

# @register.filter
# def get_item(dictionary, key):
#     return dictionary.get(key)

# Create this file: your_app/templatetags/dict_extras.py

from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Get an item from a dictionary using a key"""
    if dictionary and hasattr(dictionary, 'get'):
        return dictionary.get(key)
    elif dictionary and hasattr(dictionary, '__getitem__'):
        try:
            return dictionary[key]
        except (KeyError, IndexError, TypeError):
            return None
    return None