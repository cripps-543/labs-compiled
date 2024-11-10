def readFromFile(filename):
    with open(filename, 'r') as file:
        return file.read()
    
if __name__ == '__main__':
    print(readFromFile("task3.txt"))