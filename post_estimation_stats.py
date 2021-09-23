import numpy as np
from scipy.stats import norm
from keras import backend as K
import pandas as pd



def create_index(alfabet): # alphabet-->number of unique categories

    """ Maps categories (strings) to integers and creates
        a dictionary with look up index. """

    index2alfa={}
    alfa2index={}

    for i in range(len(alfabet)):
        index2alfa[i]=alfabet[i]
        alfa2index[alfabet[i]]=i
    return index2alfa, alfa2index



def get_betas_and_embeddings(trained_model, Q_df_train):


    UNIQUE_CATS= sorted(list(set(Q_df_train.values.reshape(-1))))

    DICT={}

    DICT['index2alfa_from'], DICT['alfa2index_from']=create_index(UNIQUE_CATS)

    DICT['index2alfa_from'], DICT['alfa2index_from']=create_index(UNIQUE_CATS)
    betas_embs = trained_model.get_layer('Utilities_embs').get_weights()[0].reshape(-1)
    betas_exog = trained_model.get_layer('Utilities_exog').get_weights()[0].reshape(-1)
    embeddings= trained_model.get_layer('embeddings').get_weights()[0]

    DICT['embeddings']= embeddings
    DICT['betas_embs']= betas_embs
    DICT['betas_exog']= betas_exog

    return DICT



def get_inverse_Hessian(model, model_inputs, labels, layer_name='Utilities'):

    """ This function was copied from: https://github.com/BSifringer/EnhancedDCM
    and was modified to handle singular matrix cases."""

    data_size = len(model_inputs[0])

# Get layer and gradient w.r.t. loss
    beta_layer = model.get_layer(layer_name)
    beta_gradient = K.gradients(model.total_loss, beta_layer.weights[0])[0]

# Get second order derivative operators (linewise of Hessian)
    Hessian_lines_op = {}
    for i in range(len(beta_layer.get_weights()[0])):
        Hessian_lines_op[i] = K.gradients(beta_gradient[i], beta_layer.weights[0])

# Define Functions that get operator values given inputed data
    input_tensors= model.inputs + model.sample_weights + model.targets + [K.learning_phase()]
    get_Hess_funcs = {}
    for i in range(len(Hessian_lines_op)):
        get_Hess_funcs[i] = K.function(inputs=input_tensors, outputs=Hessian_lines_op[i])

# Line by line Hessian average multiplied by data length (due to automatic normalization)
    Hessian=[]
    func_inputs=[*[inputs for inputs in model_inputs], np.ones(data_size), labels, 0]
    for j in range(len(Hessian_lines_op)):
        Hessian.append((np.array(get_Hess_funcs[j](func_inputs))))
    Hessian = np.squeeze(Hessian)*data_size

# The inverse Hessian:
    try:
        invHess = np.linalg.inv(Hessian)
    except np.linalg.LinAlgError:
        print(LinAlgError('Singular matrix'))
        return np.nan

    return invHess



def get_stds(model, model_inputs, labels, layer_name='Utilities'):

    """ Gets the diagonal of the inverse Hessian, square rooted
        This function was copied from: https://github.com/BSifringer/EnhancedDCM
        and was modified to handle singular matrix cases."""

    inv_Hess = get_inverse_Hessian(model, model_inputs, labels, layer_name)

    if isinstance(inv_Hess, float):

        return np.nan

    else:

        stds = [inv_Hess[i][i]**0.5 for i in range(inv_Hess.shape[0])]

        return np.array(stds).flatten()



def model_summary(trained_model, X_train, Q_train, y_train,
                  X_vars_names=[], Q_vars_names=[]):


    emb_betas_stds= get_stds(trained_model, [X_train, Q_train], y_train, layer_name='Utilities_embs')
    exog_betas_stds= get_stds(trained_model, [X_train, Q_train], y_train, layer_name='Utilities_exog')

    betas_embs= trained_model.get_layer('Utilities_embs').get_weights()[0].reshape(-1)
    betas_exog= trained_model.get_layer('Utilities_exog').get_weights()[0].reshape(-1)

    if not isinstance(emb_betas_stds, float) and  not isinstance(exog_betas_stds, float):

        z_embs= betas_embs/emb_betas_stds

        p_embs = (1-norm.cdf(abs(z_embs)))*2

        z_exog= betas_exog/exog_betas_stds

        p_exog = (1-norm.cdf(abs(z_exog)))*2

        stats_exog=np.array(list(zip(X_vars_names, betas_exog, exog_betas_stds, z_exog, p_exog)))

        stats_embs=np.array(list(zip(Q_vars_names, betas_embs,emb_betas_stds,z_embs, p_embs)))

        stats_all=np.vstack([stats_exog,stats_embs])

        df_stats=pd.DataFrame(index=[i[0] for i in stats_all],
                         data=np.array([[np.float(i[1]) for i in stats_all],[np.float(i[2]) for i in stats_all],
                                        [np.float(i[3]) for i in stats_all],
                                        [np.round(np.float(i[4]),4) for i in stats_all]]).T,
                         columns=['Betas','St errors', 't-stat','p-value'])

        return df_stats

    else:

        return np.nan
