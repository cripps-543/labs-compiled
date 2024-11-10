#!/bin/bash

>task10.txt

while true; do
	read -p "Enter  a name (type 'quit' to exit)" name
	
	if [ "$name" == "quit" ]; then
		echo "Exiting..."
		break
	fi

	echo "$name" >> task10.txt
	echo "Added '$name' to task 10.txt"
done
