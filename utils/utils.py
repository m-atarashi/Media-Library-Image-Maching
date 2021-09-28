from glob import glob

def glob_extensions(file_extensions, dir='.'):
    paths = []
    for file_extension in file_extensions:
        paths = paths + sorted(glob(f'{dir}/*.{file_extension}'))
    return paths

# Test Code
if __name__ == '__main__':
    print(glob_extensions(['jpg', 'png'], dir='../shelf/20210607/2x'))