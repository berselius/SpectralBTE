#!/bin/bash

#cd into the directory where this script is stored
cd "${BASH_SOURCE%/*}" || exit

#then go up one level
cd ..

if [ -z $1 ]; then
    #look at the productivity logs by default
    ref=refs/notes/productivity
else
    #but allow the user to specify a ref, if they like
    ref=$1
fi

#get all of the hash pairs for the ref
hash_string=$( git notes --ref "$ref" )

#split them up into an array
hash_array=( $hash_string ) 

for ((i = 0; i < ${#hash_array[@]}; i++)); do
    #the second hash in each pair corresponds to the commit
    #so use that to access the note
    if (($i % 2 == 1)); then
        #display the note
        git notes --ref $ref show "${hash_array[$i]}"
    fi
done
