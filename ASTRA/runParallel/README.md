# Running parallel "jobs" on llrui01 cluster

This is a extremely primitive way to be running computations parallelly, but it worked for me. Bash script *RunParallel.sh* does most of the work, one just needs to prepare files and start it. First of all, source all the libraries on llrui01:

<pre><code> source /opt/intel/oneapi/setvars.sh </pre></code> 

<pre><code> source ~/root/bin/thisroot.sh </pre></code> 

I had root installed in my own directory, I used it only for tolerance analysis. The arguments to run the script go as:

<pre><code> nohup ./RunParallel.sh nameOfDirectory numberOfParallelExecutions /path/to/InputFile.txt nameOfExecutableScript.py /path/to/zipfile & </pre></code>

The name of directory is the base directory where everything is ran and saved. Number of parallel eecutions is how many different jobs are created. Input file should be .txt file where the inputs are on different lines. The number of inputs are more-or-less equally distributed for the different parallel executions. The executable file should be a python script which takes in 1 argument. Finally, one needs to have a .zip file which contains all the important files and the executable to run the process. I usually just zipped my entire directory in which I was working.
Some of the time I used *MergeAndSort.py* to merge all the output files into one, but after some time and running of several different analyses, the outputs differed very much I was too lazy to write a script and I did the merging manually.
