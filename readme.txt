cj_loader presents a script that should be used for data loading, storing and preprocessing.
Requirements of the main class Storer() are described in the docstring.
Usage:
---
from cj_loader import Storer

st = Storer()
main_frame = st.get_mainframe()
---
main_frame is a pandas DataFrame() with multiindex that consists of 'timestamp', 'coin' and 'parameter' where:
'timestamp' range is from earliest found in data files to latest one
'coin' is set of coins found in all datafiles
'parameter' is name of variable (equals to name of csv data file)

main_frame can be easily sliced by all dimensions and ranges
See examples of handling MultiIndex DataFrames here: https://www.somebits.com/~nelson/pandas-multiindex-slice-demo.html

Nex steps will be in accordance to this pipeline:
https://docs.google.com/document/d/1-5YiCan_yR_M5qlSFD7ZQhbQEcu3TrehkVxjHdM0fVk/edit?usp=sharing
