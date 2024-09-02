# Development of focusing and deflecting devices for plasma acceleration

## Astra
*Astra* beam dynamics is a software developed by DESY collaboration to track and simulate space charge. Software can be downloaded from: 

https://www.desy.de/~mpyflo/

On the same page can be found a manual for working with the software. Another source of information can be found:

https://indico.cern.ch/event/528094/contributions/2172891/attachments/1323896/1987279/Bacci_tutorial.pdf

Problem with library *libimf.so* was solved by downloading Intel basekit from the link below and also exporting the library path or simply by sourcing a created script:

<pre><code> export LD_LIBRARY_PATH=/usr/local/lib/ </pre></code>

<pre><code> source /opt/intel/oneapi/setvars.sh </pre></code>


https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2023-0/apt.html#APT-PACKAGES


To visualize results from Astra simulation, one can use *PGPlot* software, which can be obtained by following the description on page:

https://sites.google.com/fqa.ub.edu/robert-estalella-home-page/pgplot_gfortran


explanation of Powell's method for finding function minimum:

https://link.springer.com/referenceworkentry/10.1007/0-306-48332-7_393#:~:text=The%20Powell%20method%20in%20its,free%20%2C%20or%20gradient%20based%20procedures%20.

https://empossible.net/wp-content/uploads/2020/08/Lecture-Powells-Method-1.pdf


