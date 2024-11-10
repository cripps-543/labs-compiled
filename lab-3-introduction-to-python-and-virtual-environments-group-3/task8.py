import time

print("How long is the timer? (seconds)")
sleeptime = int(input())
print("\nStarting timer...")
for i in range(sleeptime, 0, -1):
    print(i)
    time.sleep(1)
