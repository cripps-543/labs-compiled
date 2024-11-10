#!/bin/bash


if  [ "$1" -lt "$2" ]; then
	echo "$1 is less than $2"
elif [ "$1" -gt "$2" ]; then
	echo "$1 is greater than $2"
else
	echo "$1 is equal to $2"
fi

if  [ "$3" = "$4" ]; then
        echo "$3 is equal to $4"
else 
	echo "$3 is not equal to $4"
fi

