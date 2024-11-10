def calculate(num1, num2, operator):
    if operator == "+":
        return num1 + num2
    elif operator == "-":
        return num1 - num2
    elif operator == "*":
        return num1 * num2
    elif operator == "/":
        if num2 == 0:
            return "Division by zero is not allowed"
        return num1 / num2
    else:
        return "Invalid operator"
    
if __name__ == "__main__":
    while True:
        print("Please enter two numbers:")
        a = input()
        if a == "quit":
            break
        b = input()
        a, b = int(a), int(b)
        print("Please enter an operator (+, -, *, /):")
        operator = input()
        result = calculate(a, b, operator)
        print(result)