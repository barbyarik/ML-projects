'''An auxiliary file containing functions for writing and reading \
informationS'''

def read_from_file(filename='example_text.txt', path='data/textfiles/'):
    '''Reading from a file'''

    with open(f'{path}{filename}', 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def write_to_file(filename='example_text.txt', text='Hello, World!'):
    '''Writing to a file'''

    with open(f'data/textfiles/{filename}', 'w') as file:
        file.write(text)