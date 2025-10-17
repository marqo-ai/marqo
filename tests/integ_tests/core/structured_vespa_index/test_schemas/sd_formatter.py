# Helper GPT script to format Vespa schema files. The script is not perfect but it helps.
import os


def format_vespa_schema(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    formatted_lines = []
    indent_level = 0
    indent_size = 4  # Set the number of spaces for indentation

    for line in lines:
        stripped_line = line.strip()

        # Process and split lines for '{' and '}' handling spaces correctly
        new_line = ""
        for i, char in enumerate(stripped_line):
            if char == '{':
                if new_line and new_line[-1] != ' ':
                    new_line += ' '  # Add space before '{'
                new_line += '{'
                indent_level += 1
            elif char == '}':
                indent_level = max(0, indent_level - 1)
                new_line += '}'
            else:
                if char == ' ' and i < len(stripped_line) - 1 and stripped_line[i + 1] in '{}':
                    continue  # Skip space before '{' or '}'
                new_line += char

        # Handle collapsed empty braces
        new_line = new_line.replace('{ }', '{}').replace('{  }', '{}')

        if new_line:  # Add the processed line with indentation
            formatted_line = (' ' * (indent_size * indent_level)) + new_line + "\n"
            formatted_lines.append(formatted_line)

    # Write formatted content back to the file
    with open(file_path, 'w') as file:
        file.writelines(formatted_lines)


# Apply function to all .sd files in script directory
root_dir = os.path.dirname(os.path.abspath(__file__))
for file in os.listdir(root_dir):
    if file.endswith('.sd'):
        print(f'Formatting {file}')
        format_vespa_schema(os.path.join(root_dir, file))
