"""
General utility helpers.
"""


def print_dict(data: dict, indent: int = 0, max_str_length: int = 50):
    """
    Pretty print a dictionary with nested structure support.
    Truncates long strings for readability.

    Args:
        data: Dictionary to print
        indent: Current indentation level (for recursion)
        max_str_length: Maximum length for string values before truncation
    """
    indent_str = "  " * indent

    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{indent_str}{key}:")
            print_dict(value, indent + 1, max_str_length)
        elif isinstance(value, str):
            if len(value) > max_str_length:
                truncated = value[:max_str_length]
                print(f"{indent_str}{key}: {truncated}... (length: {len(value)})")
            else:
                print(f"{indent_str}{key}: {value}")
        elif isinstance(value, list):
            print(f"{indent_str}{key}: [{len(value)} items]")
        else:
            print(f"{indent_str}{key}: {value}")
