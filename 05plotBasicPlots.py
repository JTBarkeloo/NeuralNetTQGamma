import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

import glob
from root_numpy import root2array
from numpy.lib.recfunctions import stack_arrays
from keras.models import Model
from keras.layers import Dense, Dropout, Input

lepchannel='ejets'
lepchannel='mujets'

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



def DNNmodel(Input_shape=(10,), n_hidden=1, n_nodesHidden=10, dropout=0.2, optimizer='adam'):
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

########################################################################################

fiSig = '/scratch/jbarkelo/20181026Ntuples/user.jbarkelo.41098*/410980.root'
#fiBkg = '/scratch/jbarkelo/20181026Ntuples/user.jbarkelo.410470.PhPy8EG.DAOD_TOPQ1.e6337_s3126_r9364_p3629.Oct18-v2_output_root/user.jbarkelo.15859005._000010.output.root'
fiBkg = '/scratch/jbarkelo/20181026Ntuples/user.jbarkelo.410470.PhPy8EG.DAOD_TOPQ1.e6337_s3126_r9364_p3629.Oct18-v2_output_root/user.jbarkelo.15859005._00001*.root'
fiSig2 = '/scratch/jbarkelo/20181026Ntuples/user.jbarkelo.41098*/410981.root'
#fiBkg2 = '/scratch/jbarkelo/20181026Ntuples/user.jbarkelo.410389.MadGraphPythia8EvtGen.DAOD_TOPQ1.e6155_s3126_r9364_p3602.Oct18-v2_output_root/user.jbarkelo.15859244._000010.output.root'
fiBkg2 = '/scratch/jbarkelo/20181026Ntuples/user.jbarkelo.410389.MadGraphPythia8EvtGen.DAOD_TOPQ1.e6155_s3126_r9364_p3602.Oct18-v2_output_root/user.jbarkelo.15859244._00001*.root'

bkgs = ['ttbar','singleTop','ttV','Vgam','diboson','WJets','ZJets','ttgam']
bkgs = ['ttbar','singleTop','ttV',       'diboson','WJets','ZJets']
#bkgs = ['ttbar','ttgam','singleTop']
#bkgs = ['ttbar']

#sig = root2pandas(fiSig,'nominal',selection = 'ejets_2015 >0||ejets_2016>0||ejets_2017>0')
#sigFriend = root2pandas(fiSig+'FCNCFriend','nominalFCNCFriend')
#SRSelect = '(ejets_2015||ejets_2016||ejets_2017)&&(ph_pt[0]>50000)&&(len(jet_e)>=2)'
#sig = pd.concat([root2pandas(fiSig,'nominal'),root2pandas(fiSig2,'nominal')])
#sigFriend = pd.concat([root2pandas(fiSig+'FCNCFriend','nominalFCNCFriend'),root2pandas(fiSig2+'FCNCFriend', 'nominalFCNCFriend')])
#sig_df = pd.concat([sig,sigFriend], axis=1,join_axes=[sig.index])

#sig = pd.concat([root2pandas(fiSig,'nominal'),root2pandas(fiSig2,'nominal')],ignore_index=1)
#sigFriend = pd.concat([root2pandas(fiSig,'nominalFCNCFriend',Friend=True),root2pandas(fiSig2, 'nominalFCNCFriend',Friend=True)],ignore_index=1)
#sig_df = pd.concat([sig,sigFriend], axis=1)#,join_axes=[sig.index])
#sig_df = pd.concat([root2pandas(fiSig,'nominalFCNCFriend',Friend=True),root2pandas(fiSig2, 'nominalFCNCFriend',Friend=True)],ignore_index=1)
sig_df = pd.read_pickle('dataframes/FCNCc.pkl')
# For Combined Neural Network:
sig_df = pd.concat([sig_df, pd.read_pickle('dataframes/FCNCu.pkl')],ignore_index=1)
# Might want to combine the two separate networks in this case instead though instead of training on a combined dataset.  Train networks for NN
sig_df = sig_df[sig_df.loc[:,lepchannel]>0]
BrRatio = 0.02
sig_df['EvWeight'] *= BrRatio*831.76/(2*21.608)
sig_df = sig_df[sig_df.loc[:,'nbjets']==1] #>0#ensures in signal region
sig_df = sig_df[sig_df.loc[:,'photon0_e']>=15000]
#sig_df = sig_df[sig_df.loc[:,'EvWeight']>=0]
#sig_df['EvWeight'] = abs(sig_df['EvWeight'])
#if lepchannel == 'ejets':
#        sig_df = sig_df[sig_df.loc[:,'NNejet']>0.8]
#elif lepchannel == 'mujets':
#        sig_df = sig_df[sig_df.loc[:,'NNmujet']>0.8]

sig_df.reset_index(drop=True)
#sig_df = sig_df['EvWeight'] = 1.

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
	tmpdf = tmpdf[tmpdf.loc[:,'photon0_e']>=15000]
#	if lepchannel == 'ejets':
#        	tmpdf = tmpdf[tmpdf.loc[:,'NNejet']>0.8]
#	elif lepchannel == 'mujets':
#	        tmpdf = tmpdf[tmpdf.loc[:,'NNmujet']>0.8]
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
import sys
#sys.exit()
"""
####bkg_df = pd.concat([root2pandas(fiBkg,'nominalFCNCFriend',Friend=True),root2pandas(fiBkg2, 'nominalFCNCFriend',Friend=True)],ignore_index=1)
bkg_df = pd.DataFrame()
for bg in bkgs:
	bkg_df = pd.concat([bkg_df,pd.read_pickle('dataframes/'+str(bg)+'.pkl')],ignore_index=1)
bkg_df = bkg_df[bkg_df.loc[:,'ejets']>0]
bkg_df = bkg_df[bkg_df.loc[:,'nbjets']>=0]
bkg_df.reset_index(drop=True)

print "Sig Shape: ", sig_df.shape, " Bkg Shape: ", bkg_df.shape

##w = pd.concat((sig_df['weight_mc'],bkg_df['weight_mc']),ignore_index=True).values
##w = pd.concat((sig_df['weight'],bkg_df['weight']),ignore_index=True).values
w = pd.concat((sig_df['EvWeight'],bkg_df['EvWeight']),ignore_index=True).values
"""

##can run something like  b = root2pandas(fiSig,'nominal', selection = 'ejets_2015 >0||ejets_2016>0') for a selection like in http://scikit-hep.org/root_numpy/start.html#a-quick-tutorial


##print sig_df.keys()
## Names of some event-level branches
#npart = ['el_e','el_eta','el_phi','el_pt','el_charge','el_ptvarcone20','el_topoetcone20','met_met','met_phi','m_lgam','m_tSM','m_qgam','nu_chi2','sm_chi2','w_chi2','ph_phi0','ph_pt0','ph_eta0','ph_e0']
#npart = ['el_pt','el_charge','el_ptvarcone20','el_topoetcone20','met_met','m_lgam','m_tSM','m_qgam','nu_chi2','ph_pt0', 'jet0_pt', 'bjet0_pt', 'deltaRlgam','deltaRjgam']

#npart = ['el_pt','el_charge','el_ptvarcone20','el_topoetcone20','met_met','m_lgam','m_tSM','m_qgam','nu_chi2','photon0_pt', 'jet0_pt', 'bjet0_pt', 'deltaRlgam','deltaRjgam']

allvars= ['lepton_e','lepton_eta','lepton_phi','lepton_pt','lepton_charge','lepton_iso','met','photon0_phi','photon0_pt','photon0_eta','photon0_e','photon0_iso','m_lgam','m_tSM','m_qgam','nu_chi2','sm_chi2','w_chi2','jet0_e','jet0_pt','jet0_eta','jet0_phi','bjet0_e','bjet0_pt','bjet0_eta','bjet0_phi','S_T','deltaRlgam','deltaRjgam','deltaRbl','deltaRjb','nbjets','njets','MWT'] #AllVars
#npart = ['el_pt','el_charge','el_ptvarcone20','el_topoetcone20','met_met','m_lgam','m_tSM','m_qgam','nu_chi2','photon0_pt', 'jet0_pt', 'bjet0_pt', 'deltaRlgam','deltaRjgam']# Mix1

#npart = ['lepton_pt','lepton_charge','el_ptvarcone20','el_topoetcone20','met_met','m_lgam','m_tSM','m_qgam','nu_chi2','photon0_pt','jet0_pt','bjet0_pt','deltaRlgam','deltaRjgam'] #'S_T', 'MWT'] #SomeNewVars

#npart = ['lepton_pt','lepton_charge','met','m_lgam','m_tSM','m_qgam','nu_chi2','photon0_pt','jet0_pt','bjet0_pt','deltaRlgam','deltaRjgam', 'S_T', 'MWT','lepton_iso','photon0_iso','deltaRbl','deltaRjb'] #AllNewVars, this works well, better for 3 hidden than 2 (Looks better, worse AUC)

#################333
npart = ['lepton_pt','lepton_charge','met','m_lgam','m_tSM','m_qgam','nu_chi2','photon0_pt','jet0_pt','bjet0_pt','deltaRlgam','deltaRjgam', 'S_T', 'MWT','photon0_iso','deltaRbl','deltaRjb'] #Mix2

#npart = ['photon0_iso','photon0_pt','m_qgam','m_lgam','m_tSM','deltaRjgam','deltaRbl','MWT','S_T','nbjets','njets','w_chi2','jet0_pt','nu_chi2','sm_chi2','deltaRlgam','deltaRjb','lepton_e','met','lepton_iso','bjet0_pt']  #based on separation alone
npart = ['photon0_iso','photon0_pt','m_qgam','m_lgam','m_tSM','deltaRjgam','deltaRbl','MWT','S_T','nbjets','njets','w_chi2','jet0_pt','deltaRlgam','lepton_e','met','bjet0_pt'] #Removing some Correlated variables/underpreforming variables/variables that shouldnt mean much ('nu_chi2','sm_chi2', 'deltaRjb', 'lepton_iso'), lepton info should be similar, both S and B coming from t to bW to lep
#npart0 = ['photon0_iso','photon0_pt','m_qgam','m_lgam','m_tSM','deltaRjgam','deltaRbl','MWT','S_T','nbjets','njets','w_chi2','jet0_pt','nu_chi2','sm_chi2','deltaRlgam','lepton_e','met','lepton_iso','bjet0_pt']#npart0
npart = ['photon0_iso','photon0_pt','m_qgam','m_lgam','m_tSM','deltaRjgam','deltaRbl','MWT','S_T','njets','nbjets','w_chi2','jet0_pt','deltaRlgam','lepton_e','met','bjet0_pt']#npart1
######################################
#npart = ['photon0_pt','deltaRjgam','deltaRbl','MWT','S_T','njets','w_chi2','jet0_pt','deltaRlgam','lepton_e','met','bjet0_pt']
#npart = ['photon0_pt','m_qgam','m_lgam','m_tSM','deltaRjgam','deltaRbl','MWT','S_T','njets','nbjets','w_chi2','jet0_pt','deltaRlgam','lepton_e','met','bjet0_pt']
npart = ['photon0_iso','photon0_pt','m_qgam','m_lgam','m_tSM','deltaRjgam','deltaRbl','MWT','S_T','njets','w_chi2','jet0_pt','deltaRlgam','lepton_e','met','bjet0_pt']
npart = ['photon0_iso','photon0_pt','m_qgam','m_lgam','deltaRjgam','deltaRbl','MWT','S_T','njets','jet0_pt','deltaRlgam','lepton_e','met','bjet0_pt']

def separation(sig,bkg):
        sep = 0
        for i in range(len(sig)):
                num = (sig[i]-bkg[i])**2
                denom = (sig[i]+bkg[i])
                sep+= num/denom
        return sep*100/2 

for key in allvars: # loop through the event-level branches and plot them on separate histograms 
    # -- set font and canvas size (optional)
    matplotlib.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(8,8), dpi=100)
    # -- declare common binning strategy (otherwise every histogram will have its own binning)
    nbins=30
    if len(set(sig_df[key]))<30:
	nbins=len(set(sig_df[key]))
#    bins = np.linspace(min(sig_df[key]), max(sig_df[key]), nbins)
    # plot!
#    _ = plt.hist(sig_df[key], histtype='step', normed=True, bins=bins, label=r'FCNC', linewidth=2)
#    _ = plt.hist(bg_df[key], histtype='step', normed=True, bins=bins, label=r'Background')
#    plt.xlabel(key)
#    plt.yscale('log')
#    plt.legend(loc='best')
#    plt.savefig('varplots/'+str(key)+'.png')
#    plt.close('all') #changed from clf()
#    a,b,c=plt.hist(sig_df[key], histtype='step',weights=sig_df['EvWeight'], bins=bins, label=r'FCNC', linewidth=2)
#    d,e,f=plt.hist(bg_df[key], histtype='step',weights=bg_df['EvWeight'], bins=bins, label=r'Background')
#    plt.close('all') #cnaged from clf()
#    print key, separation(a/sum(a),d/sum(d))
#print "Done Plotting Physics Vars used in npart"
#sys.exit()
'''
#ThisBlock
df_full = pd.concat((sig_df,bkg_df), ignore_index=True)
df = pd.concat((sig_df[npart],bkg_df[npart]),ignore_index=True)
X=df.values#as_matrix()
type(X)
X.shape
#w=pd.concat((sig_df['ejets_2015'],sig_df['ejets_2016'],sig_df['ejets_2017'],bkg_df['ejets_2015'],bkg_df['ejets_2016'],bkg['ejets_2017']),ignore_index=True).values
type(w)

#Generate an array of truth labels yo distinguish among different classes in the problem
y=[]
for _df, ID in [(sig_df,1),(bkg_df,0)]:
	y.extend([ID] * _df.shape[0])
y=np.array(y)
y.shape

ix = range(X.shape[0]) # array of indices, just to keep track of them for safety reasons and future checks

#X_train, X_test, y_train, y_test, w_train, w_test, ix_train, ix_test = train_test_split(X, y, w, ix, train_size=0.8)
######## 80% Train+Validate, 20% test
#X_train, X_test, \
#	y_train, y_test, \
#	ix_train, ix_test\
#	= train_test_split(X, y, ix, test_size=0.2)
## 64% train, 16% validate, 20% of original 80%
#X_train, X_val,\
#	y_train, y_val,\
#	ix_train, ix_val\
#	=train_test_split(X_train,y_train,ix_train,test_size=0.2)

#This
X_train, X_test, \
        y_train, y_test, \
        ix_train, ix_test,\
	w_train, w_test\
        = train_test_split(X, y, ix, w, test_size=0.2)
# 64% train, 16% validate, 20% of original 80%
X_train, X_val,\
        y_train, y_val,\
        ix_train, ix_val,\
	w_train, w_val\
        =train_test_split(X_train,y_train,ix_train, w_train,test_size=0.2)
'''



X_train,X_test,X_val=X_train[npart],X_test[npart],X_val[npart]
y_train,y_test,y_val=np.asarray(y_train),np.asarray(y_test),np.asarray(y_val)

print "Scaling \n"
from sklearn.preprocessing import StandardScaler, RobustScaler
#scaler = StandardScaler()
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
#output scaler as pickle file? to be read in to nn situation

inputs = Input(shape=(X_train.shape[1], )) # placeholder
n = X_train.shape[1]
import math
hnodes1 = 20#int(math.ceil((n+1)/2.))
hnodes2 = 20#int(math.ceil((hnodes1+1)/2.))
hnodes3 = int(math.ceil((hnodes2+1)/2.))
hidden = Dense(hnodes1, activation='relu')(inputs) #n+1
hidden = Dropout(0.2)(hidden)
hidden = Dense(hnodes2, activation='relu')(hidden) #2*n+2
hidden = Dropout(0.2)(hidden)
#hidden = Dense(hnodes3, activation='relu')(hidden)
#hidden = Dropout(0.2)(hidden)
outputs = Dense(1, activation='sigmoid')(hidden)
#outputs = Dense(2, activation='softmax')(hidden) #needs as many 
# last layer has to have the same dimensionality as the number of classes we want to predict, here 2

model = Model(inputs, outputs)
model.summary()

from keras.utils.vis_utils import plot_model
#plot_model(model, 'temp.png', show_shapes=True)

#model.compile('adam','sparse_categorical_crossentropy', metrics=['acc'])
model.compile('adam','binary_crossentropy', metrics=['acc'])
#model.compile('adagrad','binary_crossentropy', metrics=['acc'])


from keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import Counter
print "NEvents to train over: ", Counter(y_train) 
print "NEvents to test over:  ", Counter(y_test)
print "Training: "
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
class_weight_dict = dict(enumerate(class_weights))

try:
    model.fit(
        X_train, y_train, sample_weight=abs(w_train),# class_weight= class_weight_dict,# class_weight={ # rebalance class representation
#            0 : 0.70 * (float(len(y)) / (y == 0).sum()),
#            1 : 0.30 * (float(len(y)) / (y == 1).sum()) #These are some sort of weights.  seems weird to have to do this, basically what youre training on I think
###            2 : 0.40 * (float(len(y)) / (y == 2).sum())
#        },
        callbacks = [
            EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
            ModelCheckpoint('./models/tutorial-progress.h5', monitor='val_loss', verbose=True, save_best_only=True)
        ],
        epochs=400, 
	batch_size=200,#200,
	validation_data=(X_val, y_val)
#        validation_split = 0.3,
#        verbose=True
) 
except KeyboardInterrupt:
    print 'Training ended early.'

model.load_weights('./models/tutorial-progress.h5')

#################
# Visualization of model history
history = model.history.history
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
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right', fontsize=20)
# the line indicate the epoch corresponding to the best performance on the validation set
# plt.vlines(np.argmin(history['val_loss']), 45, 56, linestyle='dashed', linewidth=0.5)
plt.savefig('loss.png')
plt.close()
print 'Loss estimate on unseen examples (from validation set) = {0:.3f}'.format(np.min(history['val_loss']))
############################################################
###############


# -- Save network weights and structure
print 'Saving model...'
model.save_weights('./models/tutorial.h5', overwrite=True)
json_string = model.to_json()
open('./models/tutorial.json', 'w').write(json_string)
print 'Done'


print 'Testing...'
yhat = model.predict(X_test, verbose = True, batch_size = 512) 
print "yhat: ", yhat

yhat_cls = np.argmax(yhat, axis=1)

import itertools
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

'''
#compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat_cls, sample_weight=w_test)
np.set_printoptions(precision=4)

plot_confusion_matrix(cnf_matrix, classes=['Sig', 'Bkg'],
                      normalize=True,
                      title='Normalized confusion matrix')

# signal eff = weighted tpr --> out of all signal events, what % for we classify as signal?
print 'Signal efficiency:', w_test[(y_test == 1) & (yhat_cls == 1)].sum() / w_test[y_test == 1].sum()

# bkg eff = weighted fpr --> out of all bkg events, what % do we classify as signal?
b_eff = w_test[(y_test != 0) & (yhat_cls == 0)].sum() / w_test[y_test != 0].sum()
print 'Background efficiency:', b_eff
print 'Background rej:', 1 / b_eff
'''
## -- events that got assigned to class 0
#predicted_sig = df_full.iloc[np.array(ix_test)[yhat_cls == 1]]
#predicted_sig['true'] = y_test[yhat_cls == 1]#Changed from 0to 1 Feb 07

#print predicted_sig.head()

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
plt.xlim(0.,1.2)
plt.ylim(0.,1.4)
#plt.yscale('log')
plt.savefig('roc.png')
plt.clf()

#sd=scaler.transform(sig_df[npart])
#bd=scaler.transform(bg_df[npart])
shat=model.predict(scaler.transform(sig_df[npart]))
bhat=model.predict(scaler.transform(bg_df[npart]))
Lumi=36207+43587+58450 #pb
wesi=np.ones_like(shat)/len(shat)
sig_df['EvWeight'] /= BrRatio*831.76/(2*21.608)
wesi=sig_df['EvWeight']*sig_df['YearLumi']
webi=np.ones_like(bhat)/len(bhat)
webi=bg_df['EvWeight']*bg_df['YearLumi']
bins =np.linspace(0,1,100)
a,b,c = plt.hist(shat, histtype='stepfilled', alpha=0.5, color='red', label=r"Signal", bins=bins, weights=wesi)
d,e,f = plt.hist(bhat, histtype='stepfilled', alpha=0.5, color='blue', label=r'Background', bins=bins, weights=webi)
plt.legend(loc='upper center')
plt.xlabel('P(signal) assigned by the model')
plt.tight_layout()
plt.savefig('WeightedAllsigbkg.png')
plt.close('all')
plt.clf()

BR=[.1,1.,10.,50.,100.] #% of nominal 21.61pb-1
BR=[0.01,0.05,0.1,0.5,1.,5.]
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
	plt.savefig('significance'+name+'.png')
	print "Significance Plot Made"
#	print sig
#plt.yscale('log')
#################################################################
######  Max Significance vs Branching Ratio         #############
#################################################################
BR=np.linspace(1e-5,2e-3,50)
sig,maxes={},[]
for br in BR:
	sig[str(br)]=[]
	for i in range(len(a)):
        	sig[str(br)].append(sum(br*831.76/21.61/(2)*a[i:])/math.sqrt(sum(br*831.76/21.61/(2)*a[i:])+sum(d[i:])))
	maxes.append(max(sig[str(br)]))

logx,logy=np.log(BR),np.log(maxes)
coeffs = np.polyfit(logx,logy,deg=3)
poly = np.poly1d(coeffs)
yfit = lambda x: np.exp(poly(np.log(x)))
#Reverse fit, can get BR for specific significance
co2 = np.polyfit(logy,logx,deg=3)
p2 = np.poly1d(co2)
y2 = lambda x: np.exp(p2(np.log(x)))
#####################################

_=plt.scatter(BR,maxes)
_=plt.plot(BR,yfit(BR))
plt.xlabel('Branching Ratio')
plt.ylabel('Significance s/sqrt(s+b)')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.tight_layout()
textstr='\n'.join((r'Sig at BR 5.00e-5: %.2f'%(yfit(5e-5),),r'Sig at BR %.2e: 5.0'%(y2(5.0e0),),r'Sig at BR 5.00e-4: %.2f'%(yfit(5e-4),)))
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

