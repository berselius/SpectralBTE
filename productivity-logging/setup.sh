#!/bin/bash

#cd into the directory where this script is stored
cd "${BASH_SOURCE%/*}" || exit

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
	#provide help info
	echo "Sets up the logging scripts in the specified repo."
	echo "Intended to be run in the same directory as the logging scripts."
	echo "Usage: setup.sh [main-repo-path] [-h] [--help]"
	echo "main-repo-path [optional] - the path to the top level of the main repo;"
	echo "                            uses the level above the current directory"
	echo "                            by default"
	echo "-h --help - displays this help message"
	exit 0
elif [ -z "$1" ]; then
	#if we don't have a positional argument,
	#use the level above this by default
	MAIN_REPO=..
else
	#if we have a positional argument passed in,
	#interpert it as the path to the main repo
	MAIN_REPO=$1
fi

echo "Starting setup"
#setup the main repo hooks
for hook in commit-msg post-commit pre-push ; do
	echo "Starting setup for $hook"	

	hook_copy=$MAIN_REPO/.git/hooks/$hook

	#move over the main-repo-side hooks
	cp $hook $hook_copy
	printf "\t$hook copied to $hook_copy\n" 
	
	#make sure LOG variable line up in all the hooks
	#sed -i "3iLOG=.commit.log" $hook_copy

	#make sure the hooks are executable
	chmod +x $hook_copy
	printf "\t$hook_copy is now executable\n"
	
	#let the user know that the hook was configured properly
	echo "Setup complete for $hook"
done

echo "Setup complete for all hooks"
