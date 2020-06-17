from ROOT import *
import glob
gROOT.SetBatch(True)
fiSig = '/scratch/jbarkelo/20181026Ntuples/user.jbarkelo.410980*/410980.root'
#fiSig2 = '/scratch/jbarkelo/20181026Ntuples/user.jbarkelo.410980*/410981.root'
files = glob.glob(fiSig)#+glob.glob(fiSig2)
#vars = ['el_pt','el_e','el_phi','el_eta','el_charge','el_topoetcone20','el_ptvarcone20','ph_pt','ph_eta','ph_phi','ph_e','ph_ptcone20','met_met','met_phi']
vars = ['lepton_e','lepton_pt','lepton_charge','met_met','met_phi','photon0_pt','photon0_e','m_lgam','m_tSM','m_qgam','nu_chi2','sm_chi2','w_chi2','jet0_e','jet0_pt','bjet0_e','bjet0_pt','S_T','deltaRlgam','deltaRjgam','nbjets','njets','MWT']
vars = ['lepton_pt','lepton_charge','met_met','m_lgam','m_tSM','m_qgam','nu_chi2','photon0_pt','jet0_pt','bjet0_pt','deltaRlgam','deltaRjgam', 'S_T', 'MWT','lepton_iso','photon0_iso','deltaRbl','deltaRjb']



f = TFile.Open(files[0],'read')
tree = f.Get('nominal')
tree.ls()
dict0 = {}
for var in vars:
	dict0[var]={}
	tempvars=list(vars)
#	tempvars.remove(var)
	xbin,ybin = 50,50
	#xmin,xmax = eval('min(tree.'+var+')'),eval('max(tree.'+var+')')
	xmin,xmax = -2,100000000
	ymin,ymax = -2,100000000
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
