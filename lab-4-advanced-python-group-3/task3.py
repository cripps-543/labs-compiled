
# Takes a name as input and adds it to a file called task3.txt
def addName(name):
    with open("task3.txt", "a") as file:
        file.write(name + "\n")

if __name__ == '__main__':
    for i in range(3):
        name = input("Enter your name: ")
        addName(name)
        print("Name added to file")