# collect_source_files.py
from datetime import datetime
import os

# List of files and directories to exclude
EXCLUDES = ['.git', '__pycache__', 'venv', 'env', '.env', 'build', 'dist', 'setup.py', 'tests',
            'setup_project_structure.py', 'collect_source_files.py', "assets", ".venv", "ddqpro.egg-info",
            ".gitignore", "requirements.txt", "setup.py" "data/utils", "docs"]

def should_exclude(path, excludes):
    for exclude in excludes:
        if exclude in path:
            return True
    return False

def collect_python_files(base_dir, excludes):
    python_files = []
    for root, dirs, files in os.walk(base_dir):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d), excludes)]
        for file in files:
            if file.endswith('.py') and not should_exclude(os.path.join(root, file), excludes):
                full_path = os.path.join(root, file)
                python_files.append(full_path)
    return python_files

def write_collected_files(python_files, output_file):
    with open(output_file, 'w') as outfile:
        # Write the datetime as the first line
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        outfile.write(f"# Files extracted on: {current_datetime}\n\n")
        # Write the list of excluded files/directories at the top
        outfile.write(f"# Excluded files and directories: {EXCLUDES}\n\n")
        for file_path in python_files:
            outfile.write(f"# File: {file_path}\n")
            with open(file_path, 'r') as infile:
                outfile.write(infile.read())
                outfile.write('\n\n')  # Add some spacing between files

if __name__ == "__main__":
    # Set the base directory to the current working directory
    base_directory = os.getcwd()
    output_filename = 'collected_source_code.py'

    python_files = collect_python_files(base_directory, EXCLUDES)
    write_collected_files(python_files, output_filename)
    print(f"Collected {len(python_files)} Python files into {output_filename}")
