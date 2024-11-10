import csv
import matplotlib.pyplot as plt
from datetime import datetime

date = []
stocks = {}
labels= []

with open("task9.csv", "r") as file:
    reader = csv.reader(file)
    for index, row in enumerate(reader):

        if index == 0:
            labels = row

            for stock in row[1:]:
                stocks[stock] = []
            continue
        date.append(datetime.strptime(row[0], '%d/%m/%Y'))
        for i,stock in enumerate(labels[1:]):
            stocks[stock].append(float(row[i+1]))


# print(stocks)
colors = ["blue","grey","black","red", "green", "yellow"]
for index, stock in enumerate(stocks):
    change = stocks[stock][-1]/stocks[stock][0]*100
    change = round(stocks[stock][-1] / stocks[stock][0] * 100, 2)
    if change < 0:
        sign = "-"
    else:
        sign = "+"
    plt.plot(date, stocks[stock], label=f"{stock}: {sign}{change}%", color=colors[index])


plt.legend()
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("FAANG Performance")
plt.show()