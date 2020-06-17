import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import glob
from root_numpy import root2array
from numpy.lib.recfunctions import stack_arrays
from keras.models import Model,Sequential
from keras.layers import Dense, Dropout, Input
from sklearn.metrics import roc_curve,roc_auc_score

import argparse
import math
import pickle
#run with 
#python 02multiModels.py -lep ejets -q up
#python 02multiModels.py -lep mujets -q up 
#python 02multiModels.py -lep ejets -q charm
#python 02multiModels.py -lep mujets -q charm

parser = argparse.ArgumentParser(description='Argument Parser for condor stuffs')
parser.add_argument('-lep', '--lepton',  type=str, required=True, dest='lepchannel',    help='lepton channel to use: ejets or mujets, string')
parser.add_argument('-q', '--quark',  type=str, required=False,default='both', dest='qchannel',    help='quark channel to use: up or charm or both, string')
# parse the arguments, throw errors if missing any
args = parser.parse_args()

lepchannel=args.lepchannel
qchannel = args.qchannel

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

def DNNmodel(Input_shape=(10,), n_hidden=1, n_nodesHidden=20, dropout=0.2, optimizer='adam'):
        inputs=Input(shape=Input_shape)
        i=0
        if n_hidden>0:
                hidden=Dense(n_nodesHidden, activation='relu')(inputs)
                hidden=Dropout(dropout)(hidden)
                i+=1
        while i<n_hidden:
                hidden=Dense(n_nodesHidden, activation='relu')(hidden)
                hidden=Dropout(dropout)(hidden)
                i+=1
        outputs = Dense(1,activation='sigmoid')(hidden)
        model = Model(inputs,outputs)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        return model


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
        plt.xlabel('Cut on P(signal) assigned by the model')
        plt.ylabel('Significance s/sqrt(s+b)')
        plt.tight_layout()
        plt.savefig('modelouts/significance'+name+'.png')
	plt.close('all')

        print "Significance Plot Made"

########################################################################################
bkgs = ['ttbar','singleTop','ttV','Vgam','diboson','WJets','ZJets','ttgam']
bkgs = ['ttbar','singleTop','ttV',       'diboson','WJets','ZJets']
#bkgs = ['ttbar','ttgam','singleTop']
#bkgs = ['ttbar']
### Changeable parameters
#lepchannel='ejets'
#lepchannel='mujets'
#qchannel  ='charm'
#qchannel ='up'
print "Lep Channel: ", lepchannel, type(lepchannel)
print "Quark Channel: ", qchannel, type(qchannel)
if qchannel == 'charm':
	sig_df = pd.read_pickle('dataframes/FCNCc.pkl')
elif qchannel == 'up':
	sig_df = pd.read_pickle('dataframes/FCNCu.pkl')
elif qchannel == 'both':
	sig_df = pd.read_pickle('dataframes/FCNCc.pkl')
	sig_df = pd.concat([sig_df, pd.read_pickle('dataframes/FCNCu.pkl')],ignore_index=1)

else:
	print 'Qchannel set incorrectly.  Must be "charm" or "up"'
sig_df = sig_df[sig_df.loc[:,lepchannel]>0]
sig_df = sig_df[sig_df.loc[:,'nbjets']==1] #>0#ensures in signal region
sig_df = sig_df[sig_df.loc[:,'photon0_e']>=15000]
BrRatio = 0.02
sig_df['EvWeight'] *= BrRatio*831.76/(2*21.608)

#if lepchannel == 'ejets':
#	sig_df = sig_df[sig_df.loc[:,'NNejet']>0.8]
#elif lepchannel == 'mujets':
#	sig_df = sig_df[sig_df.loc[:,'NNmujet']>0.8]

#sig_df = sig_df[sig_df.loc[:,'EvWeight']>=0]
#sig_df['EvWeight'] = abs(sig_df['EvWeight'])
sig_df.reset_index(drop=True)
#sig_df = sig_df['EvWeight'] = 1.


ix = range(sig_df.shape[0])
sig_train,sig_test,ix_train,ix_test = train_test_split(sig_df,ix,test_size=0.2)
sig_train,sig_val,ix_train,ix_val  = train_test_split(sig_train,ix_train,test_size=0.2)
X_train,X_test,X_val=sig_train,sig_test,sig_val
bkg_train,bkg_test,bkg_val = {},{},{}
bg_df =pd.DataFrame()
bg_train,bg_test,bg_val = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
print sig_train.shape, sig_test.shape, sig_val.shape
for bk in bkgs:
	tmpdf = pd.read_pickle('dataframes/'+str(bk)+'.pkl')
	tmpdf = tmpdf[tmpdf.loc[:,lepchannel]>0]
	tmpdf = tmpdf[tmpdf.loc[:,'nbjets']==1] #>0
	tmpdf = tmpdf[tmpdf.loc[:,'photon0_e']>=15000]
 #       if lepchannel == 'ejets':
 #               tmpdf = tmpdf[tmpdf.loc[:,'NNejet']>0.8]
 #       elif lepchannel == 'mujets':
 #               tmpdf = tmpdf[tmpdf.loc[:,'NNmujet']>0.8]
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
### Scale Factor
#EvWScale =sum(sig_df['EvWeight'])/sum(bg_df['EvWeight'])
#X_train['EvWeight'],X_test['EvWeight'],X_val['EvWeight']=EvWScale*X_train['EvWeight'],EvWScale*X_test['EvWeight'],EvWScale*X_val['EvWeight']
##
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
#import sys
#sys.exit()

##can run something like  b = root2pandas(fiSig,'nominal', selection = 'ejets_2015 >0||ejets_2016>0') for a selection like in http://scikit-hep.org/root_numpy/start.html#a-quick-tutorial


allvars= ['lepton_e','lepton_eta','lepton_phi','lepton_pt','lepton_charge','lepton_iso','met','photon0_phi','photon0_pt','photon0_eta','photon0_e','photon0_iso','m_lgam','m_tSM','m_qgam','nu_chi2','sm_chi2','w_chi2','jet0_e','jet0_pt','jet0_eta','jet0_phi','bjet0_e','bjet0_pt','bjet0_eta','bjet0_phi','S_T','deltaRlgam','deltaRjgam','deltaRbl','deltaRjb','nbjets','njets','MWT'] #AllVars

################################################
#npart0 = ['lepton_pt','lepton_charge','met','m_lgam','m_tSM','m_qgam','nu_chi2','photon0_pt','jet0_pt','bjet0_pt','deltaRlgam','deltaRjgam', 'S_T', 'MWT','photon0_iso','deltaRbl','deltaRjb'] #Mix2
#npart0 = ['photon0_iso','photon0_pt','m_qgam','m_lgam','m_tSM','deltaRjgam','deltaRbl','MWT','S_T','nbjets','njets','w_chi2','jet0_pt','nu_chi2','sm_chi2','deltaRlgam','deltaRjb','lepton_e','met','lepton_iso','bjet0_pt']  #based on separation alone
#npart0 = ['photon0_iso','photon0_pt','m_qgam','m_lgam','m_tSM','deltaRjgam','deltaRbl','MWT','S_T','njets','w_chi2','jet0_pt','nu_chi2','sm_chi2','deltaRlgam','lepton_e','met','lepton_iso','bjet0_pt']
#npart0 = ['photon0_pt','m_qgam','m_lgam','m_tSM','deltaRjgam','deltaRbl','MWT','S_T','njets','w_chi2','deltaRlgam','met']
#npart0 for models with various branching ratios May 21
npart0 = ['photon0_iso','photon0_pt','m_qgam','m_lgam','m_tSM','deltaRjgam','deltaRbl','MWT','S_T','njets','w_chi2','jet0_pt','deltaRlgam','lepton_e','met','bjet0_pt'] #removed nbjets after njets
#npart1 = ['photon0_iso','photon0_pt','m_qgam','m_lgam','m_tSM','deltaRjgam','deltaRbl','MWT','S_T','nbjets','njets','w_chi2','jet0_pt','deltaRlgam','lepton_e','met','bjet0_pt'] #Removing some Correlated variables/underpreforming variables/variables that shouldnt mean much ('nu_chi2','sm_chi2', 'deltaRjb', 'lepton_iso'), lepton info should be similar, both S and B coming from t to bW to lep
npart1 = ['photon0_iso','photon0_pt','deltaRjgam','deltaRbl','MWT','S_T','njets','w_chi2','jet0_pt','deltaRlgam','lepton_e','met','bjet0_pt']
#npart1 = ['photon0_pt','deltaRjgam','deltaRbl','MWT','S_T','njets','w_chi2','jet0_pt','deltaRlgam','lepton_e','met','bjet0_pt']


#for npart in inputvars:

print "Scaling \n"
from sklearn.preprocessing import StandardScaler, RobustScaler
#scaler = StandardScaler()
scaler = RobustScaler()
#X_train = scaler.fit_transform(X_train)
#X_val = scaler.transform(X_val)
#X_test = scaler.transform(X_test)

#X_train,X_test,X_val=X_train[npart],X_test[npart],X_val[npart]
y_train,y_test,y_val=np.asarray(y_train),np.asarray(y_test),np.asarray(y_val)


modelDict={}
modelDict['1hidnpart0']=DNNmodel(Input_shape=(len(npart0),),n_hidden=1)
modelDict['1hidnpart1']=DNNmodel(Input_shape=(len(npart1),),n_hidden=1)
modelDict['2hidnpart0']=DNNmodel(Input_shape=(len(npart0),),n_hidden=2)
modelDict['2hidnpart1']=DNNmodel(Input_shape=(len(npart1),),n_hidden=2)
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import Counter
print "NEvents to train over: ", Counter(y_train) 
print "NEvents to test over:  ", Counter(y_test)
print "Training: "
from sklearn.utils import class_weight

for key in modelDict:
	print "Strating fit on Model: ", key
	try:
	    if 'npart0' in key:
		Xtrain,Xtest,Xval=X_train[npart0],X_test[npart0],X_val[npart0]	
	    elif 'npart1' in key:
		Xtrain,Xtest,Xval=X_train[npart1],X_test[npart1],X_val[npart1]
	    Xtrain = scaler.fit_transform(Xtrain)
	    Xval = scaler.transform(Xval)
            Xtest = scaler.transform(Xtest)
	    modelname=key
	    model=modelDict[key]
	    model.fit(
        	Xtrain, y_train, sample_weight=abs(w_train),
	        callbacks = [
        	    EarlyStopping(verbose=True, patience=50, monitor='val_loss'),
	            ModelCheckpoint('./models/'+lepchannel+qchannel+key+'-progress.h5', monitor='val_loss', verbose=True, save_best_only=True)
        	],
	        epochs=200, 
		batch_size=100,#200,
		validation_data=(Xval, y_val)) 
	except KeyboardInterrupt:
	    print 'Training ended early.'
        #################
        # Visualization of model history
        history = model.history.history
        pickle.dump(history,open('./modelouts/'+lepchannel+qchannel+key+'-history.pkl','wb'))
        #print "history keys: ", history.keys()
	if 'npart0' in key:
                print "Saving Scaler: " + './models/'+lepchannel+qchannel+'npart0scaler.pkl'
        	joblib.dump(scaler,'./models/'+lepchannel+qchannel+'npart0scaler.pkl')
        elif 'npart1' in key:
		print "Saving Scaler: " + './models/'+lepchannel+qchannel+'npart1scaler.pkl'
	        joblib.dump(scaler,'./models/'+lepchannel+qchannel+'npart1scaler.pkl')	

fprDict,tprDict,threshDict,aucDict={},{},{},{}
for key in modelDict:
	print "Model: ", key
	model=modelDict[key]
	model.load_weights('./models/'+lepchannel+qchannel+key+'-progress.h5')

        if 'npart0' in key:
            Xtrain,Xtest,Xval=X_train[npart0],X_test[npart0],X_val[npart0]
        elif 'npart1' in key:
            Xtrain,Xtest,Xval=X_train[npart1],X_test[npart1],X_val[npart1]
        Xtrain = scaler.fit_transform(Xtrain)
        Xval = scaler.transform(Xval)
        Xtest = scaler.transform(Xtest)

	#Accuracy plot
	plt.plot(100 * np.array(history['acc']), label='training')
	plt.plot(100 * np.array(history['val_acc']), label='validation')
	plt.xlim(0)
	plt.xlabel('epoch')
	plt.ylabel('accuracy %')
	plt.legend(loc='lower right', fontsize=20)
	plt.savefig('./modelouts/'+lepchannel+qchannel+key+'accuarcy.png')
	plt.close()
	
	#loss plot
	plt.plot(np.array(history['loss']), label='training')
	#plt.plot(100 * np.array(history['val_loss']), label='validation')
	plt.xlim(0)
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.ticklabel_format(axis='y',style='sci')
	plt.legend(loc='upper right', fontsize=20)
	# the line indicate the epoch corresponding to the best performance on the validation set
	# plt.vlines(np.argmin(history['val_loss']), 45, 56, linestyle='dashed', linewidth=0.5)
	plt.savefig('./modelouts/'+lepchannel+qchannel+key+'loss.png')
	plt.close()
	
	print 'Loss estimate on unseen examples (from validation set) = {0:.3f}'.format(np.min(history['val_loss']))
	############################################################
	###############
	# -- Save network weights and structure
	print 'Saving model...'
	model.save_weights('./models/'+lepchannel+qchannel+key+'.h5', overwrite=True)
	json_string = model.to_json()
	open('./models/'+lepchannel+qchannel+key+'.json', 'w').write(json_string)
	print 'Done'


	print 'Testing...'
	yhat = model.predict(Xtest, verbose = True, batch_size = 512) 
	print "yhat: ", yhat
	
	yhat_cls = np.argmax(yhat, axis=1)

	## -- events that got assigned to class 0
	#predicted_sig = df_full.iloc[np.array(ix_test)[yhat_cls == 1]]
	#predicted_sig['true'] = y_test[yhat_cls == 1]#Changed from 0to 1 Feb 07

	#print predicted_sig.head()
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
	plt.savefig('./modelouts/'+lepchannel+qchannel+key+'sigbkg.png')
	print 'Saved ./modelouts/'+lepchannel+qchannel+key+'sigbkg.png'
	plt.close('all')



	print "Sum of weights of first layer mapped to input variable: "
	we = model.layers[1].get_weights()
	for i in range(len(we[0])):
		if 'npart0' in key:
			print npart0[i], " : ", sum(we[0][i])
		elif 'npart1' in key:
                        print npart1[i], " : ", sum(we[0][i])

	print "Getting ROC Curve Info. . ."
	from sklearn.metrics import roc_curve,roc_auc_score
	#fpr = false positive, tpr = true positive
	fprDict[key],tprDict[key],threshDict[key]= roc_curve(y_test,yhat)
	aucDict[key] = roc_auc_score(y_test,yhat)

plt.figure(figsize=(10,10))
for key in fprDict:
	_=plt.plot(tprDict[key],1.-fprDict[key], label='%s: AUC=%.3f' %(key,aucDict[key]))
plt.legend()
plt.xlabel(r'True Positive Rate $\epsilon_{signal}$')
plt.ylabel(r'1-False Positive Rate $1-\epsilon_{bkg}$')
plt.xlim(0.,1.2)
plt.ylim(0.,1.4)
#plt.yscale('log')
plt.savefig('modelouts/'+lepchannel+qchannel+'roc.png')
print "ROC Curve Saved"
plt.clf()
plt.close('all')

sig_df['EvWeight'] /= BrRatio*831.76/(2*21.608)

for key in modelDict:
        print "Model: ", key
        model=modelDict[key]
        model.load_weights('./models/'+lepchannel+qchannel+key+'.h5')
        if 'npart0' in key:
            Xtrain,Xtest,Xval=X_train[npart0],X_test[npart0],X_val[npart0]
            npart = npart0
	elif 'npart1' in key:
            Xtrain,Xtest,Xval=X_train[npart1],X_test[npart1],X_val[npart1]
	    npart= npart1
	scaler.fit_transform(Xtrain)
	shat=model.predict(scaler.transform(sig_df[npart]))
	bhat=model.predict(scaler.transform(bg_df[npart]))
	wesi=np.ones_like(shat)/len(shat)
	wesi=sig_df['EvWeight']*sig_df['YearLumi']
	webi=np.ones_like(bhat)/len(bhat)
	webi=bg_df['EvWeight']*bg_df['YearLumi']
	bins =np.linspace(0,1,100)
	a,b,c = plt.hist(shat, histtype='stepfilled', alpha=0.5, color='red', label=r"Signal", bins=bins, weights=wesi)
	d,e,f = plt.hist(bhat, histtype='stepfilled', alpha=0.5, color='blue', label=r'Background', bins=bins, weights=webi)
	plt.legend(loc='upper center')
	plt.xlabel('P(signal) assigned by the model')
	plt.tight_layout()
	plt.savefig('modelouts/'+lepchannel+qchannel+key+'WeightedAllsigbkg.png')
	plt.clf()
	plt.close('all')
	
#################################################################
######  NN Cut with Max Sig vs Branching Ratio      #############
######  Max Significance vs Branching Ratio         #############
#################################################################
	BR=np.linspace(1e-5,2e-3,50)
	sig,maxes,cut={},[],[]
	for br in BR:
		sig[str(br)]=[]
		for i in range(len(a)):
        		sig[str(br)].append(sum(br*831.76/21.61/(2)*a[i:])/math.sqrt(sum(br*831.76/21.61/(2)*a[i:])+sum(d[i:])))
		maxes.append(max(sig[str(br)]))
		cut.append(b[max(xrange(len(sig[str(br)])),key=sig[str(br)].__getitem__)])
	sig5index=(np.abs(np.asarray(maxes)-5.)).argmin()
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
	plt.xlabel('Branching Ratio')
	plt.ylabel('NN Cut with max Significance')
        plt.tight_layout()
	plt.text(min(BR)*1.05,0.99, 'BR With Sig=5: (%.2e,%.2f)'%(BR[sig5index],cut[sig5index]))
	plt.xscale('log')
        plt.xlim(min(BR)*0.95,max(BR)*1.1)
	plt.ylim(.85,1.0)
        plt.savefig('modelouts/'+lepchannel+qchannel+key+'CutVsBR.png')
        print "CutVsBR Made"
        plt.clf()
        plt.close('all')

	
	_=plt.scatter(BR,maxes)
	_=plt.plot(BR,yfit(BR))
	plt.xlabel('Branching Ratio')
	plt.ylabel('Significance s/sqrt(s+b)')
	plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	plt.tight_layout()
	textstr='\n'.join((r'Sig at BR %.2e: 2.0'%(y2(2.0e0 ),),r'Sig at BR %.2e: 5.0'%(y2(5.0e0),),r'Sig at BR 5.00e-4: %.2f'%(yfit(5e-4),)))
	plt.text(min(BR)*1.05,max(maxes)*0.5, textstr)
	plt.yscale('log')
	plt.ylim(min(maxes)*0.95,max(maxes)*1.1)
	plt.xlim(min(BR)*0.95,max(BR)*1.1)
	plt.xscale('log')
	plt.savefig('modelouts/'+lepchannel+qchannel+key+'SigVsBR.png')
	print "SigVsBR Made"
	plt.clf()
	#plt.close('all')
	
	#signum = a  bgnum = d
####################################3
	BR = [0.001, 0.005,0.01, 0.05, 0.1]#%s come out
	sigplot(a,d,bins,BR,name=lepchannel+qchannel+key+'1',setlog=True)
	BR=[1e-5,y2(2.0),y2(5.0),5e-4,1e-3,5e-3] #% of nominal 21.61pb-1
	sigplot(a,d,bins,BR,name=lepchannel+qchannel+key+'2',setlog=True)
	print "Donezo"


