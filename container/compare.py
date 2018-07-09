import os
import sys
import subprocess
from io import StringIO

count=0
def check_benchmark(dir_to_check):
    global count
    all_dir=os.listdir(dir_to_check)
    for j in all_dir:
        next_dir=dir_to_check+'/'+j
        if os.path.isdir(next_dir):
            check_benchmark(next_dir)
        else:
            if 'smt2' in j:
                count += 1
                try:
                    output = str(subprocess.check_output('./Trau '+next_dir, shell=True))
                    if output[-6:-3] == "SAT":
                        print(count,next_dir,': SAT')
                    else:
                        print(count,next_dir,': NOT')
                except:
                    print(count,next_dir,': ERROR')

check_benchmark(input("input test data location:"))