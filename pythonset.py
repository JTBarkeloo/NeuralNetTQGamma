import numpy as np

from numpy.lib.recfunctions import stack_arrays
from root_numpy import root2array, root2rec, testdata, tree2array
#import root_numpy
import glob

fiSig = '/scratch/jbarkelo/20181026Ntuples/user.jbarkelo.410980.aMcAtNloPythia8EvtGen_NNPDF30ME_A14_Q2cgambWmln.deriv.DAOD_TOPQ1.e6908_a875_r9364_p3629_output_root/410980.root'
fiBkg = '/scratch/jbarkelo/20181026Ntuples/user.jbarkelo.410470.PhPy8EG.DAOD_TOPQ1.e6337_s3126_r9364_p3629.Oct18-v2_output_root/user.jbarkelo.15859005._000010.output.root' 
arr = root2array(fiSig,'nominal')

import ROOT
rfile = ROOT.TFile(fiSig)
intree = rfile.Get('nominal')

array = tree2array(intree)
#array = tree2array(intree,
#	branches=['ph_pt','el_pt'],
#	start = 0, stop = 10)

#print arr, array
#print array
print array.dtype.names

#sig = root2array(fi)
#sig
#type(sig)
#sig.shape
#sig.dtype.names

import pandas as pd
def root2pandas(files_path, tree_name, **kwargs):
	'''
	Args:
	-----
	files_path: a string like './data/*.root', for example
        tree_name: a string like 'Collection_Tree' corresponding to the name of the folder inside the root 
        file that we want to open
        kwargs: arguments taken by root2array, such as branches to consider, start, stop, step, etc
	Returns:
	--------    
        output_panda: a pandas dataframe like allbkg_df in which all the info from the root file will be stored
    
	Note:
	-----
	if you are working with .root files that contain different branches, you might have to mask your data
        in that case, return pd.DataFrame(ss.data)
	'''
	# -- create list of .root files to process
	files = glob.glob(files_path)
    
	# -- process ntuples into rec arrays
	ss = stack_arrays([root2array(fpath, tree_name, **kwargs).view(np.recarray) for fpath in files])
	try:
		return pd.DataFrame(ss)
	except Exception:
	        return pd.DataFrame(ss.data)	
## Usage:
## -- usage of root2pandas
##singletop = root2pandas('./files/single_top.root', 'events')

def flatten(column):
    	'''
	    Args:
	    -----
	        column: a column of a pandas df whose entries are lists (or regular entries -- in which case nothing is done)
	                e.g.: my_df['some_variable'] 
	
	    Returns:
	    --------    
	        flattened out version of the column. 
		
	        For example, it will turn:
	        [1791, 2719, 1891]
	        [1717, 1, 0, 171, 9181, 537, 12]
	        [82, 11]
	        ...
	        into:
	        1791, 2719, 1891, 1717, 1, 0, 171, 9181, 537, 12, 82, 11, ...
	    '''
	try:
        	return np.array([v for e in column for v in e])
        except (TypeError, ValueError):
    	        return column




sigDF = root2pandas(fiSig,'nominal')
bkgDF = root2pandas(fiBkg,'nominal')
# -- save a pandas df to hdf5 (better to first convert it back to ndarray, to be fair)
#sigDF.to_pickle('sigDF.h5')
#bkgDF.to_pickle('bkgDF.h5')
sigDF.head()

ph_df =sigDF[[key for key in df.keys() if key.startswith('ph')]]
ph_df.head()
df_flat = pd.DataFrame({k: flatten(c) for k, c in ph_df.iteritems()})
df_flat.head()





