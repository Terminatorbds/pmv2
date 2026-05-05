"""Quick sanity check that everything is wired correctly."""
import sys
import pandas as pd
import numpy as np
import matplotlib
import sklearn

print(f"Python      : {sys.version.split()[0]}")
print(f"pandas      : {pd.__version__}")
print(f"numpy       : {np.__version__}")
print(f"matplotlib  : {matplotlib.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print("\n[OK] Environment is ready.")