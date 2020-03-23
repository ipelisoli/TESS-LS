## TESS-LS

Given a TESS TIC name, downloads data for all available sectors and performs a period search using a Lomb-Scargle periodogram.\
Additionally, it searches for the object in Gaia.\
The output contains both the light curve and a Gaia CMD showing the location of the object.\
\
Requires python >= 3.6\
Other required packages can be installed with conda install -r requirements.txt (for conda users) or pip install -r requirements.txt\
\
This code was designed for a personal use quick look, therefore it has plenty of limitations. Improvements will be made eventually, suggestions are welcome.
