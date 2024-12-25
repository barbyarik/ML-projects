def read_from_file(filename='example_text.txt'):
    with open(f'data/textfiles/{filename}', 'r') as file:
        text = file.read()
    return text

def write_to_file(filename='example_text.txt', text='Hello, World!'):
    with open(f'data/textfiles/{filename}', 'w') as file:
        file.write(text)