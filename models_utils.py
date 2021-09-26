import numpy as np
from models import E_MNL, EL_MNL
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model



def cats2ints(Q_df_train):

    UNIQUE_CATS= sorted(list(set(Q_df_train.values.reshape(-1))))
    cat2index={}

    for i in range(len(UNIQUE_CATS)):

        cat2index[UNIQUE_CATS[i]]=i

    return cat2index



def cats2ints_transform(Q_df, cat2index):

    Q=[]

    for obs in Q_df.values:

        input_i=[cat2index[cat] for cat in obs]
        Q.append(input_i)


    return np.array(Q)



def E_MNL_train(X_train, Q_train, y_train,
                N_EPOCHS= 10, VERBOSE= 1,
                save_model=1, model_filename= str()):

    NUM_CHOICES= X_train.shape[2]
    NUM_CONT_VARS= X_train.shape[1]
    NUM_EMB_VARS= Q_train.shape[-1]

    UNIQUE_CATS= sorted(list(set(Q_train.reshape(-1))))
    NUM_UNIQUE_CATS= len(UNIQUE_CATS)
    
    mnl_model = E_MNL(NUM_CONT_VARS, NUM_EMB_VARS,
                      NUM_CHOICES, NUM_UNIQUE_CATS)

    optimizer = Adam(clipnorm= 50.)
    mnl_model.compile(optimizer= optimizer, metrics= ["accuracy"], loss= 'categorical_crossentropy')

    Callback = EarlyStopping(monitor= 'loss', min_delta= 0, patience= 20)

    if VERBOSE:

        print(mnl_model.summary())

        mnl_model.fit([X_train, Q_train], y_train, epochs= N_EPOCHS,
                   steps_per_epoch= 50, shuffle= 'batch',
                   verbose= VERBOSE, callbacks=[Callback])


        pred_prob_train= mnl_model.predict(x= {'Features': X_train, 'input_categories': Q_train})


        LL_train= sum([np.log(x) for x in np.multiply(pred_prob_train.reshape(-1), y_train.reshape(-1)) if x!= 0.0])
        print('LL train:', LL_train)

    else:

        mnl_model.fit([X_train, Q_train], y_train, epochs= N_EPOCHS,
                   steps_per_epoch= 50, shuffle= 'batch',
                   verbose= VERBOSE, callbacks=[Callback])

    if save_model:

        if not model_filename:

            print("Model file name not provided. Keras model is saved as 'temp_model'")

            model_filename= 'temp_model'

        mnl_model.save(model_filename)

    return mnl_model



def EL_MNL_train(X_train, Q_train, y_train,
                 n_extra_emb_dims= 2, N_NODES= 15,
                 N_EPOCHS= 1, VERBOSE= 1,
                 save_model= 1, model_filename= str()):

    NUM_CHOICES= X_train.shape[2]
    NUM_CONT_VARS= X_train.shape[1]
    NUM_EMB_VARS= Q_train.shape[-1]

    UNIQUE_CATS= sorted(list(set(Q_train.reshape(-1))))
    NUM_UNIQUE_CATS= len(UNIQUE_CATS)

    XTRA_EMB_DIMS= n_extra_emb_dims

    mnl_model = EL_MNL(NUM_CONT_VARS, NUM_EMB_VARS,
                       NUM_CHOICES, NUM_UNIQUE_CATS,
                       XTRA_EMB_DIMS, N_NODES)


    optimizer = Adam(clipnorm= 50.)
    mnl_model.compile(optimizer= optimizer, metrics= ["accuracy"], loss= 'categorical_crossentropy')

    Callback = EarlyStopping(monitor= 'loss', min_delta= 0, patience= 20)

    if VERBOSE:

        print(mnl_model.summary())

        mnl_model.fit([X_train, Q_train], y_train, epochs= N_EPOCHS,
                       steps_per_epoch= 50, shuffle= 'batch',
                       verbose= VERBOSE, callbacks=[Callback])

        pred_prob_train= mnl_model.predict(x= {'Features': X_train, 'input_categories': Q_train})


        LL_train= sum([np.log(x) for x in np.multiply(pred_prob_train.reshape(-1), y_train.reshape(-1)) if x!= 0.0])
        print('LL train:', LL_train)

    else:

        mnl_model.fit([X_train, Q_train], y_train, epochs= N_EPOCHS,
                   steps_per_epoch= 50, shuffle= 'batch',
                   verbose= VERBOSE, callbacks=[Callback])

    if save_model:

        if not model_filename:

            print("Model file name not provided. Keras model is saved as 'temp_model'")

            model_filename= 'temp_model'

        mnl_model.save(model_filename)

    return mnl_model



def model_load_and_predict(X, Q, y, model_filename= str()):

    mnl_model= load_model(model_filename)

    pred_prob= mnl_model.predict(x={'Features': X, 'input_categories': Q})

    LL= sum([np.log(x) for x in np.multiply(pred_prob.reshape(-1), y.reshape(-1)) if x!=0.0])

    return LL



def model_predict(X, Q, y, trained_model):

    pred_prob= trained_model.predict(x={'Features': X, 'input_categories': Q})

    LL= sum([np.log(x) for x in np.multiply(pred_prob.reshape(-1), y.reshape(-1)) if x!=0.0])

    return LL
