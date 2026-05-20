# python grep.py [options] pattern [files]

# from argparse import ArgumentParser
from argparser import ArgumentParser
import re
from pathlib import Path

parser = ArgumentParser(description='Grep')

parser.add_argument('pattern', help='regex')
parser.add_argument('files', nargs='*', help='files')

parser.add_argument('-i', '--ignore-case', action='store_true', help='case insensitive')
parser.add_argument('-l', '--files-with-matches', action='store_true', help='print only file name')
parser.add_argument('-n', '--line-number', action='store_true', help='number of line')
parser.add_argument('-w', '--word-regexp', action='store_true', help='match only words')
parser.add_argument('-r', '--recursive', action='store_true', help='recursive search')

args = parser.parse_args()

def main(args):
    pattern = args.pattern
    files = args.files

    if args.word_regexp:
        pattern = r'\b' + pattern + r'\b'

    if args.ignore_case:
        pattern = re.compile(pattern, re.IGNORECASE)
    else:
        pattern = re.compile(pattern)

    if not files:
        while True:
            try:
                line = input()
                if re.search(pattern, line):
                    print(line)
            except:
                return
    
    path = Path('.')
    search_func = path.rglob if args.recursive else path.glob
    searching_files = []
    wildcard = files[0]

    if "*" in wildcard or "?" in wildcard or wildcard == '.':
        wildcard = wildcard if wildcard != '.' else '*'
        searching_files = [file for file in search_func(wildcard) if file.is_file()]

    else:
        for file in files:
            f = Path(file)
            if f.exists() and f.is_file():
                searching_files.append(file)

    if not searching_files:
        print('Not found files')
        return 0

    two_or_more_f = len(searching_files) > 1

    matches = 0
    for file in searching_files:
        with open(file, encoding='utf-8') as f:
            for i, line in enumerate(f, start=1):
                if re.search(pattern, line):
                    matches += 1

                    if args.files_with_matches:
                        print(f.name)
                        break

                    if args.line_number:
                        line = str(i) + ':' + line

                    if two_or_more_f:
                        line = f.name + ':' + line

                    print(line[:-1])
    
    if not matches:
        print('Not matches found')
        return matches
    return 1
                    
main(args)