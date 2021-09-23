from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Activation, Dropout, Flatten, concatenate, Lambda
from keras.layers import Conv2D, Add, Reshape
from keras import regularizers
import tensorflow as tf
#from keras.utils import plot_model


def E_MNL(cont_vars_num, emb_vars_num, choices_num,
          unique_cats_num, emb_size, pos_constraint= True,
          logits_activation= 'softmax'):

    """ E-MNL: Multinomial Logit model as a CNN
               with interpretable embeddings """

    main_input= Input((cont_vars_num, choices_num, 1), name= 'Features')
    emb_input= Input(shape= (emb_vars_num,), name= 'input_categories')

    hidden= Embedding(output_dim= emb_size, name= 'embeddings',
                      embeddings_regularizer= regularizers.l2(0.01),
                      input_dim= unique_cats_num, trainable= True)(emb_input)

    emb_dropout= Dropout(0.2, name= 'dropout_layer')(hidden)
    emb_final= Reshape([emb_vars_num, choices_num,1], name= 'reshape_embs')(emb_dropout)

    if pos_constraint==True:

        utilities1= Conv2D(filters= 1, kernel_size= [emb_vars_num,1],
                           kernel_constraint= tf.keras.constraints.NonNeg(),
                           strides= (1,1), padding= 'valid', name= 'Utilities_embs',
                           use_bias= False, trainable= True)(emb_final)

        utilities2= Conv2D(filters= 1, kernel_size= [cont_vars_num, 1],
                           strides= (1,1), padding= 'valid', name= 'Utilities_exog',
                           use_bias= False, trainable= True)(main_input)

        utilities= Add(name= 'add_Utilities')([utilities1, utilities2])

    else:

        final_data= concatenate([emb_final]+[main_input], name= 'concat_embs_and_exogenous', axis= 1)


        utilities= Conv2D(filters= 1, kernel_size= [cont_vars_num + (emb_vars_num),1],
                          strides= (1,1), padding= 'valid', name= 'Utilities',
                          use_bias= False, trainable= True)(final_data)

    utilitiesR= Reshape([choices_num], name= 'Flatten_Dim')(utilities)

    logits=  Activation(logits_activation, name= 'Choice')(utilitiesR)
    model= Model(inputs= [main_input,emb_input], outputs= logits, name= 'Choice')

    return model


def EL_MNL(cont_vars_num, emb_vars_num, choices_num,
           unique_cats_num, emb_size, n_nodes, pos_constraint=True,
           logits_activation = 'softmax'):

    """ EL-MNL: Multinomial Logit model as a CNN
                with interpretable embeddings
                plus representation learning term R
                according to (Sifringer et al. 2020)"""


    main_input= Input((cont_vars_num, choices_num,1), name= 'Features')
    emb_input= Input(shape= (emb_vars_num,), name= 'input_categories')

    hidden= Embedding(output_dim= emb_size, name= 'embeddings',
                       embeddings_regularizer= regularizers.l2(0.01),
                       input_dim= unique_cats_num, trainable= True)(emb_input)

    emb_dropout= Dropout(0.2, name= 'dropout_layer')(hidden)


    emb1= Lambda(lambda z: z[:,:,:choices_num], name= 'get_emb_utilities')(emb_dropout)
    emb2= Lambda(lambda z: z[:,:,choices_num:], name= 'get_extra_dims')(emb_dropout)
    emb2= Reshape([emb_vars_num*(emb_size-choices_num),1,1], name= 'reshape_extra')(emb2)

    dense= Conv2D(filters= n_nodes, kernel_size= [emb_vars_num*(emb_size-choices_num), 1],
                   activation='relu', padding='valid', name= 'Dense_NN_per_frame')(emb2)

    new_feature= Dense(units= choices_num, name= 'Output_new_feature')(dense)

    new_featureR= Reshape([choices_num], name= 'Remove_Dim')(new_feature)

    emb_final= Reshape([emb_vars_num,choices_num,1], name= 'reshape_embs')(emb1)

    if pos_constraint==True:

        utilities1=  Conv2D(filters= 1, kernel_size= [emb_vars_num,1],
                            kernel_constraint= tf.keras.constraints.NonNeg(),
                            strides= (1,1), padding= 'valid', name= 'Utilities_embs',
                            use_bias= False, trainable= True)(emb_final)

        utilities2 = Conv2D(filters= 1, kernel_size= [cont_vars_num, 1],
                            strides= (1,1), padding= 'valid', name= 'Utilities_exog',
                            use_bias= False, trainable= True)(main_input)

        utilities= Add(name='add_Utilities')([utilities1, utilities2])

    else:

        final_data= concatenate([emb_final] + [main_input], name= 'concat_embs_and_exogenous', axis=1)


        utilities= Conv2D(filters= 1, kernel_size= [cont_vars_num + (emb_vars_num),1],
                          strides= (1,1), padding= 'valid', name= 'Utilities',
                          use_bias= False, trainable= True)(final_data)


    utilitiesR= Reshape([choices_num], name= 'Flatten_Dim')(utilities)
    final_utilities= Add(name= 'New_Utility_functions')([utilitiesR, new_featureR])


    logits= Activation(logits_activation, name= 'Choice')(final_utilities)
    model= Model(inputs= [main_input, emb_input], outputs=logits, name= 'Choice')


    return model
