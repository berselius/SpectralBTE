import subprocess as sub
from glob import glob
import os.path
import argparse

def rdiff(dir1,dir2):
	fail = 0

	first_paths = glob(os.path.abspath(dir1) + "/*")
	second_paths = glob(os.path.abspath(dir2) + "/*")

	for fp in first_paths:
		for sp in second_paths:
			if(os.path.basename(fp) == os.path.basename(sp)):
				p = sub.Popen(['diff',fp,sp],stdout=sub.PIPE,stderr=sub.PIPE)
				output, errors = p.communicate()
			
				if(output!=''):
					print('error with ' + os.path.basename(fp))
					fail = 1
	
	return fail

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("directory_1")
	parser.add_argument("directory_2")

	args = parser.parse_args()
	
	rdiff(args.directory_1, args.directory_2)
