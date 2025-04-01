# prop-chaos

To use:

1) import numpy, matplotlib, torch, seaborn (I think that was it?)
2) mkdir results
3) Then see example commands in jobs.sh (for practice, make just set the time small and dont use large m's,
   eg. I recommend: python3 main_nn.py -d 16 -mb 128 -ms 3 -t 8 -p He4 -l 0.01 [This will run d=16, m=128, 256, 512, for time t = 8, problem He4, learning rate 0.01.
   You should repeat this command for a few values of d.
   See all possible problems names in main_nn.py if you want to vary the problem
5) run plotting.py to make plots. You need to hand-set the values of d you want in plotting.py in line 197. Example: python3 plotting.py -p He4 -mb 128 -ms 3 -t 8 -k 1. The final k is the the index of the problem, since they are all MIMs.
