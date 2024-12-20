# Tolerance analysis

The entire code for the analysis is toleranceAnalysis.py. For running it though, one needs several other files- they are all in toleranceAnalysis.zip file. The program is executed with 1 argument:

<pre><code> ./toleranceAnalysis.py inputOffsets.txt </pre></code> 

inputOffsets.txt includes all the different offsets which are to be ran in microns. Before running, the Intel base tool-kit has to be sourced:

<pre><code> source /opt/intel/oneapi/setvars.sh </pre></code> 

when generating new offsets, if you want to see the gradient, skew angle, magnetic centre and the field map, just change argument showPlot to True in *gen.generateFieldMap()*. Some of my data are stored in directory *data/*. For creation of plots of emittance and RMS, I wrote simple C++ code *CreatePlots.C*. Same way can other variables be plotted. 

For analysis of data stored in TTree, I used ROOT's TBrowser. When I opened the TBrowser (not in online mode), I used the TBrowser ability to plot emittance for offset 100:

<pre><code> outputTree->Draw("outEmitNormX>>hist(100, 1.173, 1.12", "offsetInMicrons == 100") </pre></code>

When I wanted to add another histogram with additional condition:

<pre><code> outputTree->Draw("outEmitNormY>>+hist", "offsetInMicrons <= 10 && Q1MCXAmp1 <= 5") </pre></code>

One can add as many conditions very easily- very convenient.