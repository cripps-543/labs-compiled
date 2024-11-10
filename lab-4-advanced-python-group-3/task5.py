import csv

def addToCSV():
    while True:
        name = input("Enter name: ")
        if name == 'quit':
            break
        with open('task5.csv', mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([name])
        print("Name added to file")

def readFromCSV():
    with open('task5.csv', mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            print(row[0])



if __name__ == '__main__':
    addToCSV()
    readFromCSV()