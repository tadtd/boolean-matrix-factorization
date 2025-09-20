'''
BMF - Boolean Matrix Factorization Library

Import specific algorithms as needed:
    from BMF.models.asso import AssoAlgorithm
    from BMF.models.greedy import GreedyBMF
    
Or import utilities:
    from BMF.utils import BMFResult, boolean_product
    
Or import everything:
    import BMF
    algo = BMF.models.asso.AssoAlgorithm(rank=3, tau=0.8)
'''

# Make submodules available at package level (numpy-style)
from . import models as models, utils as _utils

# Utility access
utils = _utils.utils
BMFResult = _utils.BMFResult
boolean_product = utils.boolean_product
cosine_similarity = utils.cosine_similarity

# For backward compatibility
AssoAlgorithm = models.Asso
GreedyBMF = models.Greedy
BMFAlgorithm = models.BMFAlgorithm

# Package metadata
__version__ = '0.1.0'
__author__ = 'Tien-Dat Do'
__email__ = 'dotiendat1725@gmail.com'

__all__ = [
  # Submodules
  'models',
  'utils',
  
  # Utility functions
  'boolean_product',
  'cosine_similarity',

  # Direct classes (backward compatibility)
  'Asso',
  'Greedy', 
  'GreConD',
  'BMFAlgorithm',
  
  # Metadata
  '__version__',
  '__author__',
  '__email__'
]