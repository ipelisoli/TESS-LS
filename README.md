## TESS-LS

Given a TESS TIC name, downloads data for all available sectors, performs a period search using a Lomb-Scargle periodogram, and generates a quick-look plot.\
Both low (2-min) and fast (20-sec) cadence data are downloaded, and processed separately.\
It will also download the Target Pixel File (TPF) for the first available sector, for visualisation purposes.
Additionally, it searches for objects in Gaia within a 30" radius of the coordinates in the image header (epoch propagation is taken into account).\
\
The output quick-look plot contains (i) an image of the TPF, with the Gaia sources overplotted with sizes proportional to the brightness, (ii) the Gaia colour-magnitude diagram showing the location of the queried object in red, and nearby sources in blue, (iii) the Lomb-Scargle periodogram, (iv) the light curve, (v) a phase plot assuming the period to be the main peak, and (vi) a phase plot assuming twice the main peak as the period.\
\
To run it, simply do 'python TESS-LS.py [TIC number]'\
\
Requires python >= 3.6\
Other required packages can be installed with conda install -r requirements.txt (for conda users) or pip install -r requirements.txt\
\
This code was designed as a personal tool, therefore it has plenty of limitations. Improvements will be made eventually, suggestions are welcome.
