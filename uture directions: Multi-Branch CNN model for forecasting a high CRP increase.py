
############################################
# Multi-Branch 1-D CNN

def one_branch_net(Xtrain,
         filters_CNN= 8, 
         kernel_size_CNN= 5, 
         poolsize_CNN= 2,
         DropOut_CNN= 50e-2,
         strides_CNN= 1, 
         activation= 'relu', 
         stddev_pred = 1e-1, 
         padding = 'valid', 
         bias = True, 
         activation_pred= 'sigmoid', 
         deepness_CNN= 1, 
         dense_units= 1, 
         second_dense_layer = False, 
         second_dense_units = 1,
         epochs= 1000, 
         batch_size= 500, 
         verbose= 0,
         lr = 1e-3, 
         rate_decay = 1e-7, 
         winit = 1e-1):
    
    #### kernel inital weights
    Winit = tf.keras.initializers.RandomUniform(minval=-winit, maxval= winit)
    Wnorms = np.ones(filters_CNN)/(filters_CNN)
    
    ### input
    Inputs = tf.keras.Input(shape=(Xtrain.shape[1], 1))
    
    ### ### ### ### ###
    ### 1st_layer ## ##
    Wkernel =  tf.keras.constraints.min_max_norm(min_value=.95*Wnorms[0], 
                                                 max_value=1.05*Wnorms[0], 
                                                 rate=.95, axis=0)
    
    X = tf.keras.layers.Conv1D(filters= filters_CNN, 
                          kernel_size= kernel_size_CNN,
                          activation= 'linear', 
                          strides = strides_CNN, 
                          kernel_initializer= Winit,
                          kernel_constraint= Wkernel,     
                          padding = padding, 
                          use_bias = bias)(Inputs)
    X = tf.keras.layers.Activation(activation)(X)
    X = tf.keras.layers.MaxPool1D(pool_size = poolsize_CNN, padding= 'same')(X)
    X = tf.keras.layers.Dropout(rate = DropOut_CNN)(X)
        
    for jj in range(1, deepness_CNN):
        Wkernel =  tf.keras.constraints.min_max_norm(min_value=.95*Wnorms[jj], 
                                                      max_value=1.05*Wnorms[jj], 
                                                      rate=.95, axis=0)
        X = tf.keras.layers.Conv1D(filters= filters_CNN, 
                        kernel_size= kernel_size_CNN,
                        activation= 'linear', 
                        strides = strides_CNN, 
                        padding = padding, 
                        kernel_initializer= Winit,
                        kernel_constraint= Wkernel,           
                        use_bias = bias)(X)
        X = tf.keras.layers.Activation(activation)(X)
        X = tf.keras.layers.MaxPool1D(pool_size = poolsize_CNN, padding= 'same')(X)
        X = tf.keras.layers.Dropout(rate = DropOut_CNN)(X)

    ### flattern   
    output = tf.keras.layers.Flatten()(X)
    
    ##define model
    mymodel = tf.keras.models.Model(Inputs, output)
    
    return mymodel
  
  
  def multi_branch_CNN(Xtrain, Ytrain, Xtest, Ytest, 
                      filters_CNN= 8, 
                      kernel_size_CNN= 5, 
                      poolsize_CNN= 2,
                      DropOut_CNN= 50e-2,
                      strides_CNN= 1, 
                      activation= 'relu', 
                      stddev_pred = 1e-1, 
                      padding = 'valid', 
                      bias = False, 
                      activation_pred= 'sigmoid', 
                      pred_Gdropout_rate = .5, 
                      deepness_CNN= 1, 
                       dense_units= 1, 
                       epochs= 1000, 
                       batch_size= 500, 
                       verbose= 0,
                       lr = 1e-3, 
                       rate_decay = 1e-7, 
                       patience = 0, 
                       winit = 1e-1):

    ### definisci tutti i modelli
    listmodels = []
    for nfeat in range(len(Xtrain)):
        listmodels.append(one_branch_net(Xtrain[nfeat],
                                         filters_CNN= filters_CNN, 
                                         kernel_size_CNN= kernel_size_CNN, 
                                         poolsize_CNN= poolsize_CNN,
                                         DropOut_CNN= DropOut_CNN,
                                         strides_CNN= strides_CNN, 
                                         activation= activation, 
                                         padding = padding,
                                         bias = bias, 
                                         activation_pred= activation_pred, 
                                         deepness_CNN= deepness_CNN, 
                                         dense_units= dense_units, 
                                         winit = winit))
    ## define flattern model
    inputs = [model.inputs for model in listmodels]
    outputs = [model.outputs for model in listmodels]
    flatten_model = tf.keras.models.Model(inputs, outputs)
    
    ## Concatenate and predict
    X = tf.keras.layers.Concatenate()(flatten_model.outputs)
    X = tf.keras.layers.GaussianDropout(rate = pred_Gdropout_rate)(X)
    prediction = tf.keras.layers.Dense(units = dense_units, activation= activation_pred, use_bias= True)(X)
    
    ############################
    ### DEFINE multi brach model
    mymodel = tf.keras.models.Model(inputs, prediction)
    
    ############################
    adam = tf.keras.optimizers.Adam(lr= lr)
    bce = tf.keras.losses.BinaryCrossentropy()
    mymodel.compile(optimizer= adam, loss = bce)
    
    
    
    ###################################
    ## FIT 
    ### callbacks ###
    
    ###lr_scheduler
    def LRScheduler(epoch, lr= lr, rate_decay= rate_decay):
        ### linear decreasing
        lrate = lr -rate_decay 
        return lrate
    
    ## define lrate 
    lrate = tf.keras.callbacks.LearningRateScheduler(LRScheduler)
    
    ###early_stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                    min_delta=0, 
                                    patience= patience)
    ### call fit
    mymodel.fit(Xtrain, Ytrain, 
              validation_data = (Xtest, Ytest),
              epochs= epochs, 
              batch_size= batch_size, 
              verbose= verbose,
              callbacks=[lrate, early_stopping])
    
    return mymodel
  
  
  ################################################################
  
  def group_hidden_layers (X, mymodel, nfeats= nfeats, layer_per_block= 4, deepness= 8):

    ### X--> data
    ### mymodel --> modello gia fittato
    ### nfeats --> len(X)
    ### layer_per_block --> how deep is a convolutinal block
    ### deepness --> how deep the cnn is
    
    ## compute all layers
    list_layers = [layers.output for layers in mymodel.layers]
    unrolled_model = tf.keras.models.Model(mymodel.input, list_layers)

    ### get only the hidden layers (convolutinal blocks)
    pred_unrolled = unrolled_model.predict(X)
    conv_blocks = pred_unrolled[nfeats:nfeats*layer_per_block*deepness]
    total_number_of_layers = len(conv_blocks)
    
    ww = mymodel.get_weights()

    ### ### ### ### ###
    ### each list has the layers per feature (branch)
    ## LC --> convoltional
    ## LA --> activation
    ## LM --> Maxpooling 
    ## LD --> Dropout
    LC, LA, LM, LD, LW = [], [], [], [], []
    for Nfeat in range(nfeats):
        index_ = np.arange(Nfeat, total_number_of_layers, nfeats).astype(int)
        
        ### ### ### ###
        index_lc = index_[np.arange(0, index_.size, layer_per_block)]
        index_la = index_[np.arange(1, index_.size, layer_per_block)]
        index_lm = index_[np.arange(2, index_.size, layer_per_block)]
        index_ld = index_[np.arange(3, index_.size, layer_per_block)]
        
        ### ### ### ###
        LC.append([conv_blocks[kk] for kk in index_lc])
        LA.append([conv_blocks[kk] for kk in index_la])
        LM.append([conv_blocks[kk] for kk in index_lm])
        LD.append([conv_blocks[kk] for kk in index_ld])
        
        ### ### ### ###
        LW.append([ww[kk] for kk in np.arange(Nfeat, deepness*nfeats, nfeats)])
        
    return LC, LA, LM, LD, LW
  
  
#### #### #### #### #### #### #### #### ####  
### Make reconstrunction from the deepset hiddenlayer
RECS = []
Xinf = Xtest[Ytest==1]
for deepness_ in np.arange(3, 2, -1): ### the reconstruntion starts from the deepest layer
    print(deepness_)
    missing_counter = 0
    Xrecinf = np.zeros(Xinf.shape)
    leftover = Xinf.shape[0]
    for inst in range(Xinf.shape[0]):   
        leftover += -1
        xx = [xinf[kk][inst].reshape(1, -1) for kk in range(0, nfeats)]
        LC, LA, LM, LD, LW = group_hidden_layers(xx, 
                                                 mymodel, 
                                                 nfeats= nfeats, 
                                                 layer_per_block= 4, 
                                                 deepness= deepness_)
        
        #print(leftover)
        for kfeat in range(nfeats):
            Rec, hidden_rec = zei.deconv(feat_map_conv = LC[kfeat],
                                         feat_map_before = LA[kfeat],
                                         feat_map_after = LM[kfeat],
                                         weights = LW[kfeat],
                                         strides = 1,
                                         poolsize = 2,
                                         actv= 'relu', 
                                         phi_factor = 1)

            ### ###
            m0, s0 = Xinf[inst, :, kfeat].mean(), Xinf[inst, :, kfeat].std()
            zz = Rec.ravel()
            theta_bias = m0-s0*np.mean(zz)/np.std(zz)
            erg_bias = s0/np.std(zz)
            zz = zz*erg_bias+theta_bias
            try:
                Xrecinf[inst, :, kfeat] = zz.ravel()
            except:
                 missing_counter += 1
                    
    RECS.append(Xrecinf)
                
   #####################################################################
                
    def correlation_per_inst(Xtrue, Xrec):
    
    ninst, ntime, nfeats = Xtrue.shape
    Pmtx = np.zeros((ninst, nfeats))
    for ii in range(ninst):
        for kk in range(nfeats):
            Pmtx[ii, kk] = pearsonr(Xtrue[ii, :, kk],  Xrec[ii, :, kk])[0]
            
    return Pmtx

def R2_per_inst(Xtrue, Xrec):
    
    ninst, ntime, nfeats = Xtrue.shape
    Pmtx = np.zeros((ninst, nfeats))
    for ii in range(ninst):
        for kk in range(nfeats):
            try:
                Pmtx[ii, kk] = explained_variance_score(Xtrue[ii, :, kk],  Xrec[ii, :, kk])
            except:
                Pmtx[ii, kk] = np.nan
    return Pmtx
