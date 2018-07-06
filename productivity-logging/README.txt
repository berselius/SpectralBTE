Overview:

	This repo contains a few scripts designed to enable productivity
	logging on a git repo. Most are git hooks, which will be automatically
	executed by git after a commit is made.

Dependencies:

	(1) bash - for scripts
	(2) git - this repo extends git's functionality
	(3) git notes - for storing logs (should come by default with newer versions
	                of git)

Setup:

	(1) run setup.sh, optionally passing in the path to the main repo
				setup.sh <main repo path>
  		[by default, <main repo path> is one level above where setup.sh is stored]

Usage:

	Once a repo is configured for logging, (see Setup), just commit like normal!
	After you finialise your commit message, you will be asked if you would like
	to create a commit log. If you answer yes, you will be prompted to answer a 
	few quick questions about the work you just commited. After answering, your
	commit will proceede like normal. Behind the scenes, though, your answers,
	plus a few pieces of metadata which are gathered automatically will be stored 
	as a git note on your commit.

	You can view the log entries for a given commit using the command
		git notes --ref refs/notes/productivity show <commit hash>
	ommiting <commit hash> will show the log for the current HEAD.

	Similiarly, if you ever need to make chages to your logs, (for instance, if you
	entered a value wrong, or there was sensitive information in your commit message)
	you can edit the log file for a specififc commit using the command
		git otes --ref refs/notes/productivity edit <commit hash>
	as above, ommiting <commit hash> will allow you to edit the log for the
	current HEAD.

Customisation:
	You can change the specific areas you will be asked about by editing
	the lists at the top of commit-msg, and then rerunning setup.sh to get
	the new version of that script into the hooks folder.
