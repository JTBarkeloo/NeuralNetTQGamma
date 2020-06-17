from ROOT import *
import glob
gROOT.SetBatch(True)
fiSig = '/scratch/jbarkelo/201903Ntuples/user.jbarkelo.410980*/*FCNCFriend'
fiSig2 = '/scratch/jbarkelo/201903Ntuples/user.jbarkelo.410981*/*FCNCFriend'
fiSig3 = '/scratch/jbarkelo/201903Ntuples/user.jbarkelo.410984*/*FCNCFriend'
fiSig4 = '/scratch/jbarkelo/201903Ntuples/user.jbarkelo.410985*/*FCNCFriend'

files = glob.glob(fiSig)+glob.glob(fiSig2)+glob.glob(fiSig3)+glob.glob(fiSig4)
#vars = ['el_pt','el_e','el_phi','el_eta','el_charge','el_topoetcone20','el_ptvarcone20','ph_pt','ph_eta','ph_phi','ph_e','ph_ptcone20','met_met','met_phi']
vars = ['lepton_e','lepton_pt','lepton_charge','met_met','met_phi','photon0_pt','photon0_e','m_lgam','m_tSM','m_qgam','nu_chi2','sm_chi2','w_chi2','jet0_e','jet0_pt','bjet0_e','bjet0_pt','S_T','deltaRlgam','deltaRjgam','nbjets','njets','MWT']
vars = ['lepton_pt','lepton_charge','met','m_lgam','m_tSM','m_qgam','nu_chi2','photon0_pt','jet0_pt','bjet0_pt','deltaRlgam','deltaRjgam', 'S_T', 'MWT','lepton_iso','photon0_iso','deltaRbl','deltaRjb']
vars = ['lepton_e','lepton_pt','lepton_charge','lepton_iso','met','photon0_pt','photon0_e','photon0_iso','m_lgam','m_tSM','m_qgam','nu_chi2','sm_chi2','w_chi2','jet0_e','jet0_pt','bjet0_e','bjet0_pt','S_T','deltaRlgam','deltaRjgam','deltaRbl','deltaRjb','nbjets','njets','MWT']
vars = ['photon0_iso','photon0_pt','deltaRjgam','deltaRbl','MWT','S_T','njets','w_chi2','jet0_pt','deltaRlgam','lepton_e','met','bjet0_pt','NNejet','NNmujet']
"""
sampleDict ={}
sampleDict['ttbar']     =  [410470]
sampleDict['singleTop'] =  [410658,410659,410644,410645,410646,410647]
sampleDict['ttV']       =  [410081,410155,410156,410157,410218,410219,410220]
sampleDict['Vgam']      =  [364500,364501,364502,364503,364504,364505,364506,364507,364508,364509,364510,364511,364512,364513,364514,364517,364518,364519,364521,364522,364523,364524,364525,364526,364527,364528,364529,364530,364531,364532,364533,364534,364535]
sampleDict['diboson']   =  [364250,364253,364254,364255,363355,363356,363357,363358,363359,363360,363489]
sampleDict['WJets']     =  [364156,364157,364158,364159,364160,364161,364162,364163,364164,364165,364166,364167,364168,364169,364170,364171,364172,364173,364174,364175,364176,364177,364178,364179,364180,364181,364182,364183,364184,364185,364186,364187,364188,364189,364190,364191,364192,364193,364194,364195,364196,364197]
sampleDict['ZJets']     =  [364100,364101,364102,364103,364104,364105,364106,364107,364108,364109,364110,364111,364112,364113,364114,364115,364116,364117,364118,364119,364120,364121,364122,364123,364124,364125,364126,364127,364128,364129,364130,364131,364132,364133,364134,364135,364136,364137,364138,364139,364140,364141]
sampleDict['ttgam']     =  [410389]

files =[]
for key in sampleDict:
	for dsid in sampleDict[key]:
		files+= glob.glob('/scratch/jbarkelo/20181026Ntuples/user.jbarkelo.'+str(dsid)+'*/*.rootFCNCFriend')
"""
#f = TFile.Open(files[0],'read')
tree = TChain('nominalFCNCFriend')
for fi in files:
	tree.Add(fi)
tree.Print()
dict0 = {}
#vars =[]
#for branch in tree.GetListOfBranches():
#	vars.append(branch.GetName()) 
for var in vars:
	dict0[var]={}
	tempvars=list(vars)
#	tempvars.remove(var)
	xbin,ybin = 50,50
	#xmin,xmax = eval('min(tree.'+var+')'),eval('max(tree.'+var+')')
	xmin,xmax = -100,100000000
	ymin,ymax = -100,100000000
	for tempvar in tempvars:
		#ymin,ymax = eval('min(tree.'+tempvar+')'),eval('max(tree.'+tempvar+')')
		b = TH2F('b','',xbin,xmin,xmax,ybin,ymin,ymax)
		tree.Draw(var+':'+tempvar+'>>b')	
		print 'Correlation for ', var, ' and ', tempvar, ": ",b.GetCorrelationFactor()
		dict0[var][tempvar]=b.GetCorrelationFactor()
		del b

import pandas as pd
df = pd.DataFrame.from_dict(dict0)
def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm
    plt.matshow(df)
    plt.xticks(rotation=90)
    plt.xticks(range(len(df.columns)), df.columns)
    plt.yticks(range(len(df.columns)), df.columns)
    plt.colorbar()
#    plt.clim(-1.,1.)
    plt.set_cmap('Spectral_r')
    plt.savefig('correlations.png')
correlation_matrix(df)	

