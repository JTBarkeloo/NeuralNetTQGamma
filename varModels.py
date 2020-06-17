import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

import glob
from root_numpy import root2array
from numpy.lib.recfunctions import stack_arrays


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


########################################################################################

fiSig = '/scratch/jbarkelo/20181026Ntuples/user.jbarkelo.410980.aMcAtNloPythia8EvtGen_NNPDF30ME_A14_Q2cgambWmln.deriv.DAOD_TOPQ1.e6908_a875_r9364_p3629_output_root/410980.root'
fiBkg = '/scratch/jbarkelo/20181026Ntuples/user.jbarkelo.410470.PhPy8EG.DAOD_TOPQ1.e6337_s3126_r9364_p3629.Oct18-v2_output_root/user.jbarkelo.15859005._000010.output.root'
fiBkg = '/scratch/jbarkelo/20181026Ntuples/user.jbarkelo.410470.PhPy8EG.DAOD_TOPQ1.e6337_s3126_r9364_p3629.Oct18-v2_output_root/user.jbarkelo.15859005._00001*.root'

#sig = root2pandas(fiSig,'nominal',selection = 'ejets_2015 >0||ejets_2016>0||ejets_2017>0')
#sigFriend = root2pandas(fiSig+'FCNCFriend','nominalFCNCFriend')

sig = root2pandas(fiSig,'nominal')
sigFriend = root2pandas(fiSig+'FCNCFriend','nominalFCNCFriend')

sig_df = pd.concat([sig,sigFriend], axis=1,join_axes=[sig.index])
#Now to go through and do a specific selection on the data frame itself, this is like a selection for ejets
sig_df = sig_df[sig_df.loc[:,'ejets_2015']+ sig_df.loc[:,'ejets_2016']+sig_df.loc[:,'ejets_2017'] ==1]
sig_df.reset_index(drop=True)
sig_df = sig_df.assign(ph_e0=pd.Series([i[0] for i in sig_df['ph_e']],index=sig_df.index))
sig_df = sig_df.assign(ph_pt0=pd.Series([i[0] for i in sig_df['ph_pt']],index=sig_df.index))
sig_df = sig_df.assign(ph_eta0=pd.Series([i[0] for i in sig_df['ph_eta']],index=sig_df.index))
sig_df = sig_df.assign(ph_phi0=pd.Series([i[0] for i in sig_df['ph_phi']],index=sig_df.index))
#Not an ideal way to do this but it will work for now, grabs leading photon information

bkg = root2pandas(fiBkg,'nominal',selection = 'ejets_2015 >0||ejets_2016>0||ejets_2017>0')
bkgFriend = root2pandas(fiBkg+'FCNCFriend','nominalFCNCFriend')

bkg_df = pd.concat([bkg,bkgFriend], axis=1,join_axes=[bkg.index])
#Now to go through and do a specific selection on the data frame itself, this is like a selection for ejets
bkg_df = bkg_df[bkg_df.loc[:,'ejets_2015']+ bkg_df.loc[:,'ejets_2016']+bkg_df.loc[:,'ejets_2017'] ==1]
bkg_df.reset_index(drop=True)
bkg_df = bkg_df.assign(ph_e0=pd.Series([i[0] for i in bkg_df['ph_e']],index=bkg_df.index))
bkg_df = bkg_df.assign(ph_pt0=pd.Series([i[0] for i in bkg_df['ph_pt']],index=bkg_df.index))
bkg_df = bkg_df.assign(ph_eta0=pd.Series([i[0] for i in bkg_df['ph_eta']],index=bkg_df.index))
bkg_df = bkg_df.assign(ph_phi0=pd.Series([i[0] for i in bkg_df['ph_phi']],index=bkg_df.index))



#Weight placeholders
#bkgw =bkg_df.loc[:,'mujets_2015']+ bkg_df.loc[:,'mujets_2016']+bkg_df.loc[:,'mujets_2017'] #mu just place holder to set bkgweights to 0 for testing!!!  Barkeloo
#sigw =sig_df.loc[:,'ejets_2015']+ sig_df.loc[:,'ejets_2016']+sig_df.loc[:,'ejets_2017']
#w=pd.concat((sigw,bkgw),ignore_index=True).values
w = pd.concat((sig_df['weight_mc'],bkg_df['weight_mc']),ignore_index=True).values



##can run something like  b = root2pandas(fiSig,'nominal', selection = 'ejets_2015 >0||ejets_2016>0') for a selection like in http://scikit-hep.org/root_numpy/start.html#a-quick-tutorial


print sig.keys()
## Names of some event-level branches
npart = ['el_e','el_eta','el_phi','el_pt','el_charge','el_ptvarcone20','el_topoetcone20','met_met','met_phi','m_lgam','m_tSM','m_qgam','nu_chi2','sm_chi2','w_chi2','ph_phi0','ph_pt0','ph_eta0','ph_e0']#,'m_lgam','m_tSM','m_qgam','nu_chi2','sm_chi2','w_chi2']
#npart = ['el_e','el_eta','el_phi','el_pt','el_charge','el_ptvarcone20','el_topoetcone20','met_met','met_phi','ph_phi0','ph_pt0','ph_eta0','ph_e0']

'''
for key in npart: # loop through the event-level branches and plot them on separate histograms 
    # -- set font and canvas size (optional)
    matplotlib.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(8,8), dpi=100)
    # -- declare common binning strategy (otherwise every histogram will have its own binning)
    bins = np.linspace(min(sig_df[key]), max(sig_df[key]) + 1, 30)
    # plot!
    _ = plt.hist(sig_df[key], histtype='step', normed=False, bins=bins, label=r'FCNC', linewidth=2)
    _ = plt.hist(bkg_df[key], histtype='step', normed=False, bins=bins, label=r'ttbar')
    plt.xlabel(key)
    plt.yscale('log')
    plt.legend(loc='best')
    plt.savefig(str(key)+'.png')
    plt.clf()
'''
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
X_train, X_test, \
	y_train, y_test, \
	ix_train, ix_test\
	= train_test_split(X, y, ix, test_size=0.2)
# 64% train, 16% validate, 20% of original 80%
X_train, X_val,\
	y_train, y_val,\
	ix_train, ix_val\
	=train_test_split(X_train,y_train,ix_train,test_size=0.2)




print "Scaling \n"
from sklearn.preprocessing import StandardScaler, RobustScaler
scaler = StandardScaler()
#scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

from keras.models import Model
from keras.layers import Dense, Dropout, Input

inputs = Input(shape=(X_train.shape[1], )) # placeholder
n = X_train.shape[1]
hidden = Dense(n+1, activation='relu')(inputs)
hidden = Dropout(0.2)(hidden)
hidden = Dense(2*n+2, activation='relu')(hidden)
hidden = Dropout(0.2)(hidden)
#hidden = Dense(4*n+4, activation='relu')(hidden)
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

def DenseNNModel(inputNodes=10,nlayers=0,n_nodesHidden=10,dropout=0.2, optimizer='adam',loss='binary_crossentropy'):
	from keras.models import Model, Sequential
	from keras.layers import Dense, Dropout, Input
	model = Sequential()
	model.add(Dense(inputNodes,activation='relu',input_shape=(inputNodes,)))
	model.add(Dropout(dropout))
	i = 0
	while i<=nlayers:
		model.add(Dense(n_nodesHidden,activation='relu'))
		model.add(Dropout(dropout))
		i+=1
	model.add(Dense(2,activation='sigmoid'))
	model.compile(optimizer=optimizer, loss=loss,metrics=['accuracy'])
	return model
#############################
models=[]
for i in [0,1]:
	models.append(DenseNNModel(inputNodes = X_train.shape[1]),nlayers=i)
names =["1Hidden",'2Hidden']

for i in range(len(models)):
	model =models[i]
	name  =names[i]
	try:
	    model.fit(
	        X_train, y_train,# class_weight= class_weight_dict,# class_weight={ # rebalance class representation
#            0 : 0.70 * (float(len(y)) / (y == 0).sum()),
#            1 : 0.30 * (float(len(y)) / (y == 1).sum()) #These are some sort of weights.  seems weird to have to do this, basically what youre training on I think
###            2 : 0.40 * (float(len(y)) / (y == 2).sum())
#        },
	        callbacks = [
	            EarlyStopping(verbose=True, patience=15, monitor='val_loss'),
        	    ModelCheckpoint('./models/'+name+'.h5', monitor='val_loss', verbose=True, save_best_only=True)
	        ],
	        epochs=200,
	        validation_data=(X_val, y_val)
	#        validation_split = 0.3,
	#        verbose=True
	)
	except KeyboardInterrupt:
	    print 'Training ended early.'
	
#############################


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
plt.plot(100 * np.array(history['loss']), label='training')
plt.plot(100 * np.array(history['val_loss']), label='validation')
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
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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
# -- events that got assigned to class 0
predicted_sig = df_full.iloc[np.array(ix_test)[yhat_cls == 0]]
predicted_sig['true'] = y_test[yhat_cls == 0]

print predicted_sig.head()

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


print "Donezo"
