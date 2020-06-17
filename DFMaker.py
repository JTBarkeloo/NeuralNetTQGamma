import numpy as np
import pandas as pd
import glob
from root_numpy import root2array
from numpy.lib.recfunctions import stack_arrays


def root2pandas(files_path, tree_name,Friend=False, **kwargs):
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
        if Friend==True:
                for i in range(len(files)):
                        files[i] = files[i]+'FCNCFriend'
    # -- process ntuples into rec arrays
        ss = stack_arrays([root2array(fpath, tree_name, **kwargs).view(np.recarray) for fpath in files])
        try:
                return pd.DataFrame(ss)
        except Exception:
                return pd.DataFrame(ss.data)

sampleDict ={}
sampleDict['FCNCc']     =  [410980, 410981]
sampleDict['FCNCu']     =  [410984, 410985]
sampleDict['ttbar']     =  [410470]
sampleDict['singleTop'] =  [410658,410659,410644,410645,410646,410647]
sampleDict['ttV']       =  [410081,410155,410156,410157,410218,410219,410220]
sampleDict['Vgam']      =  [364500,364501,364502,364503,364504,364505,364506,364507,364508,364509,364510,364511,364512,364513,364514,364517,364518,364519,364521,364522,364523,364524,364525,364526,364527,364528,364529,364530,364531,364532,364533,364534,364535]
sampleDict['diboson']   =  [364250,364253,364254,364255,363355,363356,363357,363358,363359,363360,363489]
sampleDict['WJets']     =  [364156,364157,364158,364159,364160,364161,364162,364163,364164,364165,364166,364167,364168,364169,364170,364171,364172,364173,364174,364175,364176,364177,364178,364179,364180,364181,364182,364183,364184,364185,364186,364187,364188,364189,364190,364191,364192,364193,364194,364195,364196,364197]
sampleDict['ZJets']     =  [364100,364101,364102,364103,364104,364105,364106,364107,364108,364109,364110,364111,364112,364113,364114,364115,364116,364117,364118,364119,364120,364121,364122,364123,364124,364125,364126,364127,364128,364129,364130,364131,364132,364133,364134,364135,364136,364137,364138,364139,364140,364141]
sampleDict['ttgam']     =  [410389]
#sampleDict['data']      =  ['AllYear']


for key in sampleDict:
	df = pd.DataFrame()
	for dsid in sampleDict[key]:
		for fi in glob.glob('/scratch/jbarkelo/201905Ntuples/user.jbarkelo.'+str(dsid)+'*'+'/*.root'):
			try:
				df = pd.concat([df,root2pandas(fi,'nominalFCNCFriend',Friend=True)],ignore_index=1)
			except IOError:
				"No friend tree for file: ", fi
				continue
			print str(key), " ", str(dsid), " ", df.shape
	df = df[df.loc[:,'nbjets']>=0]
	df.reset_index(drop=True)	
	print str(key), df.shape, " before emu split"			
	df.to_pickle('dataframes/'+str(key)+'.pkl')
	#df[df.loc[:,'ejets']==1].to_pickle('dataframes/'+str(key)+'.el.pkl')		
	#df[df.loc[:,'mujets']==1].to_pickle('dataframes/'+str(key)+'.mu.pkl')

#sig = pd.concat([root2pandas(fiSig,'nominal'),root2pandas(fiSig2,'nominal')],ignore_index=1)
#sigFriend = pd.concat([root2pandas(fiSig,'nominalFCNCFriend',Friend=True),root2pandas(fiSig2, 'nominalFCNCFriend',Friend=True)],ignore_index=1)
#sig_df = pd.concat([sig,sigFriend], axis=1)#,join_axes=[sig.index])
#sig_df = sig_df[sig_df.loc[:,'nbjets']>=0] #ensures in signal region
#sig_df.reset_index(drop=True)

#bkg = pd.concat([root2pandas(fiBkg,'nominal'),root2pandas(fiBkg2,'nominal')],ignore_index=1)
#bkgFriend = pd.concat([root2pandas(fiBkg,'nominalFCNCFriend',Friend=True),root2pandas(fiBkg2, 'nominalFCNCFriend',Friend=True)],ignore_index=1)
#bkg_df = pd.concat([bkg,bkgFriend], axis=1)#,join_axes=[sig.index])
#bkg_df = bkg_df[bkg_df.loc[:,'nbjets']>=0]
#bkg_df.reset_index(drop=True)

