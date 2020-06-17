import numpy as np
import pandas as pd
import glob
import pickle
from root_numpy import root2array
from numpy.lib.recfunctions import stack_arrays
import time

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
def getjetpt(JetPts, bTags, getB):
	# list of jet pts, if it is btagged or not in same length list and then 0 or 1 for getB if you want to return the leading jet pt or leading bjet pt
	a= zip(JetPts,bTags)
	pt = [item for item in a if item[1]==getB][0][0]
	return pt

sampleDict ={}
sampleDict['FCNCc']     =  [410980, 410981]
sampleDict['FCNCu']     =  [410984, 410985]
sampleDict['ttbar']     =  [410470]
sampleDict['singleTop'] =  [410658,410659,410644,410645,410646,410647]
sampleDict['ttV']      =  [410155,410156,410157,410218,410219,410220]#410081
#####sampleDict['Vgam']      =  [364500,364501,364502,364503,364504,364505,364506,364507,364508,364509,364510,364511,364512,364513,364514,364517,364518,364519,364521,364522,364523,364524,364525,364526,364527,364528,364529,364530,364531,364532,364533,364534,364535]
sampleDict['Vgam']      =  [366140,366141,366142,366143,366144,366145,366146,366147,366148,366149,366150,366151,366152,366153,366154,364522,364523,364524,364525,364527,364528,364529,364530,364532,364533,364534,364535]
sampleDict['diboson']   =  [364250,364253,364254,364255,363355,363356,363357,363358,363359,363360,363489]
sampleDict['WJets']     =  [364156,364157,364158,364159,364160,364161,364162,364163,364164,364165,364166,364167,364168,364169,364170,364171,364172,364173,364174,364175,364176,364177,364178,364179,364180,364181,364182,364183,364184,364185,364186,364187,364188,364189,364190,364191,364192,364193,364194,364195,364196,364197]
sampleDict['ZJets']     =  [364100,364101,364102,364103,364104,364105,364106,364107,364108,364109,364110,364111,364112,364113,364114,364115,364116,364117,364118,364119,364120,364121,364122,364123,364124,364125,364126,364127,364128,364129,364130,364131,364132,364133,364134,364135,364136,364137,364138,364139,364140,364141]
sampleDict['ttgam']     =  [410389]
####sampleDict['data']      =  ['AllYear']

listGen = ['weight_mc','weight_pileup','weight_leptonSF','weight_photonSF','weight_jvt','el_pt','el_e','el_topoetcone20','mu_pt','mu_e','mu_topoetcone20','ph_pt','ph_e','jet_pt','jet_e','met_met','ejets_2015','ejets_2016','ejets_2017','ejets_2018','mujets_2015','mujets_2016','mujets_2017','mujets_2018','event_mwt','event_HT','event_ST','event_njets','ph_drlph_leading','ph_mlph_closest','ph_iso_topoetcone40']
list70 = ['event_nbjets70','weight_bTagSF_MV2c10_70','jet_isbtagged_MV2c10_70','ph_drphb70_leading_pt','ph_drphb70_closest','ph_mphb70_closest','ph_drqph_leading_70','ph_mqph_closest_70','ph_drqph_closest_70','ph_mqph_leading_70','event_m_blnu_70','event_m_lnu_70','event_w_chi2_70','event_t_chi2_70','el_mlb70','el_drlb70','mu_mlb70','mu_drlb70']
list77 = ['mu_drlb77','mu_mlb77','el_drlb77','el_mlb77','event_m_blnu_77','event_m_lnu_77','event_w_chi2_77','event_t_chi2_77','ph_drqph_leading_77','ph_mqph_closest_77','ph_drqph_closest_77','ph_mqph_leading_77','ph_mphb77_closest','ph_drphb77_closest','ph_drphb77_leading_pt','jet_isbtagged_MV2c10_77','weight_bTagSF_MV2c10_77','event_nbjets77']
list85 = ['event_nbjets85','weight_bTagSF_MV2c10_85','jet_isbtagged_MV2c10_85','ph_drphb85_leading_pt','ph_drphb85_closest','ph_mphb85_closest','ph_drqph_leading_85','ph_mqph_closest_85','ph_drqph_closest_85','ph_mqph_leading_85','event_m_blnu_85','event_m_lnu_85','event_w_chi2_85','event_t_chi2_85','el_mlb85','el_drlb85','mu_mlb85','mu_drlb85']

xsecDict = pickle.load(open('/export/home/jbarkelo/AnalysisCondor/crossSecInfo.pkl','rb'))
weightDict = pickle.load(open('/export/home/jbarkelo/AnalysisCondor/weightInfo.pkl.data','rb'))
################### change ph_iso in above listGen .... check other vars
for key in sampleDict:
	start = time.time()
	df = pd.DataFrame()
	for dsid in sampleDict[key]:
		xsec,kfactor = float(xsecDict[str(dsid)].split()[0]), float(xsecDict[str(dsid)].split()[1])
		NWeightedEvents15 = weightDict[str(dsid)][1] 
		NWeightedEvents17 = weightDict[str(dsid)][2] 
		NWeightedEvents18 = weightDict[str(dsid)][3]
		for fi in glob.glob('/scratch/jbarkelo/IntermediateDFs/'+str(dsid)+'.77.*.pkl'):
			try:	
				print fi
				tmpdf = pd.read_pickle(fi)[listGen+list77]
				df = pd.concat([df,tmpdf],ignore_index=1)
				df = df[df.loc[:,'event_nbjets77']==1] #ensures only 1 bjet
				df = df.reset_index(drop=True)

			except IOError:
				"Couldnt concat: ", fi
				continue
			print str(key), " ", str(dsid), " ", df.shape
		df['weight'] = df['weight_mc']*df['weight_pileup']*df['weight_leptonSF']*df['weight_photonSF']*df['weight_jvt']*df['weight_bTagSF_MV2c10_77']
		df['EvWeight'] = xsec*kfactor*df['weight']/(NWeightedEvents15*(df['ejets_2015']+df['ejets_2016']+df['mujets_2015']+df['mujets_2016'])+NWeightedEvents17*(df['ejets_2017']+df['mujets_2017'])+NWeightedEvents18*(df['ejets_2018']+df['mujets_2018']))
	#The following isnt dependent on dsid specific information, can be added after the loop to speed things up a tad
	df['YearLumi'] = 36207.66*(df['ejets_2015']+df['ejets_2016']+df['mujets_2015']+df['mujets_2016']) + 43587.3*(df['ejets_2017']+df['mujets_2017']) + 58450.1*(df['ejets_2018']+df['mujets_2018'])
	df=df.drop(columns=['weight_mc','weight_pileup','weight_leptonSF','weight_photonSF','weight_jvt','weight_bTagSF_MV2c10_77'])	
#	df['ph_pt']=df['ph_pt'].str[0] #only first values of ph_pt	
#	df = df.dropna(subset=['ph_pt']) #removes events without photons
#	df = df.reset_index(drop=True)
	end = time.time()
	print "Starting at: ", (end-start)
	df['jet0pt']=df.apply(lambda row: getjetpt(row['jet_pt'],row['jet_isbtagged_MV2c10_77'],0),axis=1)
	end = time.time()
	print "Finished jet0pt: ", (end-start)
	df['bjet0pt']=df.apply(lambda row: getjetpt(row['jet_pt'],row['jet_isbtagged_MV2c10_77'],1),axis=1)
	end = time.time()
	print "Finished bjet0pt: ", (end-start)
	df=df.drop(columns=['jet_pt','jet_isbtagged_MV2c10_77'])
	print str(key), df.shape, " before emu split"			
	df = df.loc[df.apply(lambda row: len(row['ph_e']), axis=1)==1] #Requires 1 
	df = df.reset_index(drop=True)
        df['ejets'] =df['ejets_2015']+df['ejets_2016']+df['ejets_2017']+df['ejets_2018']
        df=df.drop(columns=['ejets_2015','ejets_2016','ejets_2017','ejets_2018'])
        df['mujets'] =df['mujets_2015']+df['mujets_2016']+df['mujets_2017']+df['mujets_2018']
	df=df.drop(columns=['mujets_2015','mujets_2016','mujets_2017','mujets_2018'])


#	df.to_pickle('/scratch/jbarkelo/dataframes70/'+str(key)+'.pkl')
	df.to_pickle('/scratch/jbarkelo/dataframes77/'+str(key)+'.pkl')
#	df.to_pickle('/scratch/jbarkelo/dataframes85/'+str(key)+'.pkl')


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

