#!/usr/bin/env python


from astra import Astra
from astra.install import install_astra, install_generator


astra_bin = install_astra('.', verbose=True)
generator_bin = install_generator('.', verbose=True)


A = Astra('parallelBeam.in')
A.verbose = True
A.run()
...
A.plot(y=['norm_emit_x', 'norm_emit_y'], y2=['sigma_x', 'sigma_y'])