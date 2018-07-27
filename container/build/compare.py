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
                flag1=0
                flag2=0
                count += 1
                try:
                    #os.system("python2 run.py -f "+j)
                    output1 = str(subprocess.check_output('./Trau '+next_dir, shell=True))
                    if 'UNSAT' in output1:
                        flag1=1
                        print(count,next_dir,': NOT by trau')
                    else:
                        print(count,next_dir,': SAT by trau')
                except:
                    flag1 = 2
                    print(count,next_dir,': ERROR by trau')

                try:
                    output2 = str(subprocess.check_output('python2 run.py -f '+next_dir))
                    if 'UNSAT' in output2:
                        flag2=1
                        print(count,next_dir,': NOT by s3p')
                    else:
                        print(count,next_dir,': SAT by s3p')
                except:
                    print(count,next_dir,': ERROR by s3p')
                    flag2 = 2
                if flag1 == flag2:
                    print(count,next_dir,"SAME")
                else:
                    print(count,next_dir,"DIFF")

check_benchmark(input("input test data location:"))