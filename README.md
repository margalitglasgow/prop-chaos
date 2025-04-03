# prop-chaos

To use:

1) import numpy, matplotlib, torch, seaborn (I think that was it?)
2) mkdir results, results/folder/data, results/folder/plotdata for some "folder" (which in the example below will be He4_practice)
3) Then see example commands in jobs.sh to run the training (for practice, make just set the time small and dont use large m's,
   eg. I recommend: python3 main_nn.py -d 16 -mb 128 -ms 3 -t 8 -p He4 -l 0.01 -a He4_practice  [This will run d=16, m=128, 256, 512, for time t = 8, problem He4, learning rate 0.01.
   You should repeat this command for a few values of d.
   See all possible problems names in problems.py if you want to vary the problem
4) Perform cleanup to condense files for plotting. python3 main_nn.py -d 16 -mb 128 -ms 3 -p He4 -a He4_practice
   Do this for all ds.
5) run plotting.py to make plots. You need to hand-set the values of d you want in plotting.py in line 197. Example: python3 plotting.py -p He4 -mb 128 -ms 3 -t 8.
   All the .npy files that are needed for this will should be stored in results/folder/plotdata if step 4) was run correctly.
