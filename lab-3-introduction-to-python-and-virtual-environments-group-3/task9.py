import time

def countdown(sleepTime):
    print("\nStarting timer...")
    for i in range(sleepTime, 0, -1):
        print(i)
        time.sleep(1)

if __name__ == "__main__":
    print("How long is the timer? (seconds)")
    sleeptime = int(input())
    countdown(sleeptime)
