import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from keras.models import model_from_json
import math
import glob
from root_numpy import root2array
from numpy.lib.recfunctions import stack_arrays
from keras.models import Model
from keras.layers import Dense, Dropout, Input
import pickle 

lepchannel = 'ejets'
lepchannel = 'mujets'
print lepchannel

def root2pandas(files_path, tree_name,Friend=False, **kwargs):
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

def flatten(column):
        try:
            return np.array([v for e in column for v in e])
        except (TypeError, ValueError):
            return column
#### Load Specific Models
if lepchannel == 'ejets':
	scaler = joblib.load('/export/home/jbarkelo/kerasTutorial/tutorialVirtEnv/models/ejetsbothscaler.pkl')
	ej_json_file = open('/export/home/jbarkelo/kerasTutorial/tutorialVirtEnv/models/ejetsboth2hidnpart0.json','r')
	ej_json = ej_json_file.read()
	ej_json_file.close()
	model = model_from_json(ej_json)
	model.load_weights('/export/home/jbarkelo/kerasTutorial/tutorialVirtEnv/models/ejetsboth2hidnpart0.h5')
	history = pickle.load(open('/export/home/jbarkelo/kerasTutorial/tutorialVirtEnv/modelouts/ejetsboth2hidnpart0-history.pkl','rb'))

elif lepchannel == 'mujets':
	scaler = joblib.load('/export/home/jbarkelo/kerasTutorial/tutorialVirtEnv/models/mujetsbothscaler.pkl')
	mj_json_file = open('/export/home/jbarkelo/kerasTutorial/tutorialVirtEnv/models/mujetsboth2hidnpart0.json','r')
	mj_json = mj_json_file.read()
	mj_json_file.close()
	model = model_from_json(mj_json)
	model.load_weights('/export/home/jbarkelo/kerasTutorial/tutorialVirtEnv/models/mujetsboth2hidnpart0.h5')
#	model.load_weights('/export/home/jbarkelo/kerasTutorial/tutorialVirtEnv/models/mujetsboth1hidnpart1-progress.h5')
	history = pickle.load(open('/export/home/jbarkelo/kerasTutorial/tutorialVirtEnv/modelouts/mujetsboth2hidnpart0-history.pkl','rb'))


bkgs = ['ttbar','singleTop','ttV','Vgam','diboson','WJets','ZJets','ttgam']
bkgs = ['ttbar','singleTop','ttV',       'diboson','WJets','ZJets']


sig_df = pd.read_pickle('dataframes/FCNCc.pkl')
# For Combined Neural Network:
sig_df = pd.concat([sig_df, pd.read_pickle('dataframes/FCNCu.pkl')],ignore_index=1)
sig_df = sig_df[sig_df.loc[:,lepchannel]>0]
sig_df = sig_df[sig_df.loc[:,'nbjets']==1] #>0#ensures in signal region
sig_df = sig_df[sig_df.loc[:,'photon0_e']>=15000]
sig_df.reset_index(drop=True)

ix = range(sig_df.shape[0])
sig_train,sig_test,ix_train,ix_test = train_test_split(sig_df,ix,test_size=0.2)
sig_train,sig_val,ix_train,ix_val  = train_test_split(sig_train,ix_train,test_size=0.2)
bkg_train,bkg_test,bkg_val = {},{},{}
bg_df =pd.DataFrame()
bg_train,bg_test,bg_val = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
X_train,X_test,X_val=sig_train,sig_test,sig_val
for bk in bkgs:
        tmpdf = pd.read_pickle('dataframes/'+str(bk)+'.pkl')
        tmpdf = tmpdf[tmpdf.loc[:,lepchannel]>0]
        tmpdf = tmpdf[tmpdf.loc[:,'nbjets']==1]#>0
        tmpdf.reset_index(drop=True)
        bg_df= pd.concat([bg_df,tmpdf],ignore_index=1)
        tmpix =range(len(ix),len(ix)+tmpdf.shape[0])
        bkg_train[bk],bkg_test[bk],tmpix_train,tmpix_test = train_test_split(tmpdf,tmpix,test_size=0.2)
        bkg_train[bk],bkg_val[bk],tmpix_train,tmpix_val  = train_test_split(bkg_train[bk],tmpix_train,test_size=0.2)
        print bk, bkg_train[bk].shape,bkg_test[bk].shape,bkg_val[bk].shape
        bg_train=pd.concat([bg_train,bkg_train[bk]],ignore_index=1)
        bg_test=pd.concat([bg_test,bkg_test[bk]],ignore_index=1)
        bg_val=pd.concat([bg_val,bkg_val[bk]],ignore_index=1)
        ix_train.extend(tmpix_train)
        ix_test.extend(tmpix_test)
        ix_val.extend(tmpix_val)
X_train,X_test,X_val=pd.concat([X_train,bg_train],ignore_index=1),pd.concat([X_test,bg_test],ignore_index=1),pd.concat([X_val,bg_val],ignore_index=1)
w_train,w_test,w_val = X_train['EvWeight'],X_test['EvWeight'],X_val['EvWeight']
y=[]
for _df, ID in [(sig_df,1),(bg_df,0)]:
        y.extend([ID] * _df.shape[0])
y_train,y_test,y_val=[],[],[]
for _df, ID in [(sig_train,1),(bg_train,0)]:
        y_train.extend([ID]*_df.shape[0])
for _df, ID in [(sig_test,1),(bg_test,0)]:
        y_test.extend([ID]*_df.shape[0])
for _df, ID in [(sig_val,1),(bg_val,0)]:
        y_val.extend([ID]*_df.shape[0])

npart = ['photon0_iso','photon0_pt','m_qgam','m_lgam','m_tSM','deltaRjgam','deltaRbl','MWT','S_T','nbjets','njets','w_chi2','jet0_pt','nu_chi2','sm_chi2','deltaRlgam','lepton_e','met','lepton_iso','bjet0_pt'] #npart0
npart = ['photon0_iso','photon0_pt','m_qgam','m_lgam','m_tSM','deltaRjgam','deltaRbl','MWT','S_T','njets','nbjets','w_chi2','jet0_pt','deltaRlgam','lepton_e','met','bjet0_pt']#npart1
npart = ['photon0_iso','photon0_pt','m_qgam','m_lgam','m_tSM','deltaRjgam','deltaRbl','MWT','S_T','njets','w_chi2','jet0_pt','deltaRlgam','lepton_e','met','bjet0_pt'] 
#npart=['photon0_iso','photon0_pt','deltaRjgam','deltaRbl','MWT','S_T','njets','w_chi2','jet0_pt','deltaRlgam','lepton_e','met','bjet0_pt'] #minimal Set
X_train,X_test,X_val=X_train[npart],X_test[npart],X_val[npart]
y_train,y_test,y_val=np.asarray(y_train),np.asarray(y_test),np.asarray(y_val)

print "Scaling \n"
#scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# Visualization of model history
#history = model.history.history
print "history keys: ", history.keys()

#Accuracy plot
plt.plot(100 * np.array(history['acc']), label='training')
plt.plot(100 * np.array(history['val_acc']), label='validation')
plt.xlim(0)
plt.xlabel('epoch')
plt.ylabel('accuracy %')
plt.legend(loc='lower right', fontsize=20)
plt.savefig('accuarcy.png')
plt.close()
#loss plot
plt.plot(np.array(history['loss']), label='training')
#plt.plot(100 * np.array(history['val_loss']), label='validation')
plt.xlim(0)
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right', fontsize=20)
# the line indicate the epoch corresponding to the best performance on the validation set
# plt.vlines(np.argmin(history['val_loss']), 45, 56, linestyle='dashed', linewidth=0.5)
plt.savefig('loss.png')
print "accuracy.png and loss.png Saved"
plt.close()
print 'Loss estimate on unseen examples (from validation set) = {0:.3f}'.format(np.min(history['val_loss']))
############################################################
###############

print 'Testing...'
yhat = model.predict(X_test, verbose = True, batch_size = 512)
print "yhat: ", yhat
yhat_cls = np.argmax(yhat, axis=1)

plt.clf()
bins = np.linspace(0, 1, 20)
#For normalization
wes = np.ones_like(yhat[y_test==1])/len(yhat[y_test==1])
web = np.ones_like(yhat[y_test==0])/len(yhat[y_test==0])
_ = plt.hist(yhat[y_test==1], histtype='stepfilled', alpha=0.5, color='red', label=r"Signal", bins=bins, weights=wes)
_ = plt.hist(yhat[y_test==0], histtype='stepfilled', alpha=0.5, color='blue', label=r'Background', bins=bins, weights=web)
#_ = plt.hist(yhat[y_test==1], histtype='stepfilled', alpha=0.5, color='red', label=r"Signal", bins=bins)
#_ = plt.hist(yhat[y_test==0], histtype='stepfilled', alpha=0.5, color='blue', label=r'Background', bins=bins)
plt.legend(loc='upper center')
plt.xlabel('P(signal) assigned by the model')
plt.tight_layout()
plt.savefig('sigbkg.png')
#plt.xscale('log')
#plt.xlim(0.7,1.)
#plt.savefig('sigbkglog.png')
plt.close('all')

print "Sum of weights of first layer mapped to input variable: "
we = model.layers[1].get_weights()
for i in range(len(we[0])):
        print npart[i], " : ", sum(we[0][i])


print "Making ROC Curves. . ."
from sklearn.metrics import roc_curve,roc_auc_score
#fpr = false positive, tpr = true positive
fpr, tpr,thresholds = roc_curve(y_test,yhat)
auc = roc_auc_score(y_test,yhat)
plt.figure(figsize=(10,10))
#plt.grid(b = True, which = 'minor')
#plt.grid(b = True, which = 'major')
_=plt.plot(tpr,1.-fpr, label='Model: AUC=%.3f' %auc)
plt.legend()
plt.xlabel(r'True Positive Rate $\epsilon_{signal}$')
plt.ylabel(r'1-False Positive Rate $1-\epsilon_{bkg}$')
plt.xlim(0.8,1.1)
plt.ylim(0.8,1.2)
#plt.yscale('log')
plt.savefig('roc.png')
plt.clf()
print "ROC Saved"
#sd=scaler.transform(sig_df[npart])
#bd=scaler.transform(bg_df[npart])
shat=model.predict(scaler.transform(sig_df[npart]))
bhat=model.predict(scaler.transform(bg_df[npart]))
wesi=np.ones_like(shat)/len(shat)
wesi=sig_df['EvWeight']*sig_df['YearLumi']
webi=np.ones_like(bhat)/len(bhat)
webi=bg_df['EvWeight']*bg_df['YearLumi']
bins =np.linspace(0,1,500)
a,b,c = plt.hist(shat, histtype='stepfilled', alpha=0.5, color='red', label=r"Signal", bins=bins, weights=wesi)
d,e,f = plt.hist(bhat, histtype='stepfilled', alpha=0.5, color='blue', label=r'Background', bins=bins, weights=webi)
plt.legend(loc='upper center')
plt.xlabel('P(signal) assigned by the model')
plt.tight_layout()
plt.savefig('WeightedAllsigbkg.png')
#plt.xlim(0.7,1.)
#plt.xscale('log')
#plt.savefig('WeightedAllsigbkglog.png')
plt.close('all')
plt.clf()
print "WeightedAllSigBkg Saved"

def sigplot(signum,bgnum,bins,BR,name='',setlog=False):
        sig={}
        plt.figure(figsize=(15,10))
        xval= [0.5*(b[i]+b[i+1]) for i in range(len(a))]
        for br in BR:
                sig[str(br)]=[]
                for i in range(len(signum)):
                        sig[str(br)].append(sum(br*831.76/21.61/(2)*signum[i:])/math.sqrt(sum(br*831.76/21.61/(2)*signum[i:])+sum(bgnum[i:])))
                _=plt.scatter(xval,sig[str(br)],label='BR: '+str(br))
        plt.legend(fontsize='x-small')
        if setlog == True:
                plt.yscale('log')
                minim,maxim =10000000.,0.
                for key in sig:
                        if min(sig[key])<minim:
                                minim=min(sig[key])
                        if max(sig[key])>maxim:
                                maxim=max(sig[key])
                plt.ylim(minim*.95,maxim*1.05)
	if name == '2':
		plt.hlines(5,0., 1, linestyle='dashed', linewidth=0.5)
		plt.hlines(2,0., 1, linestyle='dashed', linewidth=0.5)		
        plt.xlabel('Cut on P(signal) assigned by the model')
        plt.ylabel('Significance s/sqrt(s+b)')
        plt.tight_layout()
        plt.savefig('significance'+name+'.png')
        print "Significance Plot Made"
	plt.clf()
	plt.close('all')
#       print sig
#plt.yscale('log')
#################################################################
######  NN Cut with Max Sig vs Branching Ratio      #############
######  Max Significance vs Branching Ratio         #############
#################################################################
BR=np.linspace(1e-7,2e-3,1000)
sig,maxes,cut={},[],[]
for br in BR:
        sig[str(br)]=[]
        for i in range(len(a)):
                sig[str(br)].append(sum(br*831.76/21.61/(2)*a[i:])/math.sqrt(sum(br*831.76/21.61/(2)*a[i:])+sum(d[i:])))
        maxes.append(max(sig[str(br)]))
        cut.append(b[max(xrange(len(sig[str(br)])),key=sig[str(br)].__getitem__)])
sig5index=(np.abs(np.asarray(maxes)-5.)).argmin()
sig2index=(np.abs(np.asarray(maxes)-2.)).argmin()
### Fitting Info for SigVsBR
logx,logy=np.log(BR),np.log(maxes)
coeffs = np.polyfit(logx,logy,deg=3)
poly = np.poly1d(coeffs)
yfit = lambda x: np.exp(poly(np.log(x)))
#Reverse fit, can get BR for specific significance
co2 = np.polyfit(logy,logx,deg=3)
p2 = np.poly1d(co2)
y2 = lambda x: np.exp(p2(np.log(x)))
#####################################


_=plt.scatter(BR,cut)
plt.vlines(BR[sig5index],0.85, 1, linestyle='dashed', linewidth=0.5)
plt.vlines(BR[sig2index],0.85, 1, linestyle='dashed', linewidth=0.5)
plt.xlabel('Branching Ratio')
plt.ylabel('NN Cut with max Significance')
plt.tight_layout()
plt.text(min(BR)*1.05,0.99, 'Cut with Sig=5: %.2f, Cut with Sig=2: %.2f'%(cut[sig5index],cut[sig2index]))
plt.xscale('log')
plt.xlim(min(BR)*0.95,max(BR)*1.1)
plt.ylim(0.85,1.0)
plt.savefig('CutVsBR.png')
print "CutVsBR Made"
plt.clf()


_=plt.scatter(BR,maxes)
_=plt.plot(BR,yfit(BR))
plt.xlabel('Branching Ratio')
plt.ylabel('Significance s/sqrt(s+b)')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.tight_layout()
textstr='\n'.join((r'Sig at BR %.2e: 2.0'%(y2(2.0e0),),r'Sig at BR %.2e: 5.0'%(y2(5.0e0),),r'Sig at BR 5.00e-4: %.2f'%(yfit(5e-4),)))
plt.text(min(BR)*1.05,max(maxes)*0.5, textstr)
plt.yscale('log')
plt.ylim(min(maxes)*0.95,max(maxes)*1.1)
plt.xlim(min(BR)*0.95,max(BR)*1.1)
plt.xscale('log')
plt.savefig('SigVsBR.png')
print "SigVsBR Made"
plt.clf()
plt.close('all')
#signum = a  bgnum = d
####################################3
BR = [0.001, 0.005,0.01, 0.05, 0.1]#%s come out
sigplot(a,d,bins,BR,name='1',setlog=True)
BR=[1e-5,y2(2.0),y2(5.0),5e-4,1e-3,5e-3] #% of nominal 21.61pb-1
sigplot(a,d,bins,BR,name='2',setlog=True)

print "Donezo"

