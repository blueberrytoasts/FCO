import re
import pyperclip

INPUT_FILE = "testfiledens.txt"

def get_clean_temps(file_path):
    pattern = re.compile(r"\((\d+\.\d+)C\)")
    temps = []
    
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                temps.append(match.group(1))
    
    if not temps:
        print("No temperatures found!")
        return

    # Prepare blocks of 10
    output_blocks = []
    for i in range(0, len(temps), 10):
        group = temps[i:i+10]
        # Joining with \t (Tabs) so they paste horizontally in a row
        output_blocks.append("\t".join(group))

    # Join groups with a single newline so each block starts on its own row
    final_string = "\n".join(output_blocks)
    
    # Push to clipboard
    pyperclip.copy(final_string)
    
    print(f"Success! {len(temps)} temperatures copied.")
    print("Go to Excel and Ctrl+V. They will paste horizontally now.")

if __name__ == "__main__":
    get_clean_temps(INPUT_FILE)