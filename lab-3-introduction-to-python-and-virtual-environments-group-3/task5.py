print("Please enter two numbers:")
a = input()
b = input()
a, b = int(a), int(b)
if a > b:
    print(f"{a} is greater than {b}.")
elif a < b:
    print(f"{a} is less than {b}.")
else:
    print(f"{a} is equal to {b}.")
