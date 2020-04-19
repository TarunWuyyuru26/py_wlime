
# Loading Required Packages
import numpy as np

# REGRESSION METRICS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

EPSILON = 1e-10


def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error
    Note: result is NOT multiplied by 100
    """
    return _error(actual, predicted) / (actual + EPSILON)


def _naive_forecasting(actual: np.ndarray, seasonality: int = 1):
    """ Naive forecasting method which just repeats previous samples """
    return actual[:-seasonality]


def _relative_error(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Relative Error """
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark
        return _error(actual[seasonality:], predicted[seasonality:]) /\
               (_error(actual[seasonality:], _naive_forecasting(actual, seasonality)) + EPSILON)

    return _error(actual, predicted) / (_error(actual, benchmark) + EPSILON)


def _bounded_relative_error(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Bounded Relative Error """
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark

        abs_err = np.abs(_error(actual[seasonality:], predicted[seasonality:]))
        abs_err_bench = np.abs(_error(actual[seasonality:], _naive_forecasting(actual, seasonality)))
    else:
        abs_err = np.abs(_error(actual, predicted))
        abs_err_bench = np.abs(_error(actual, benchmark))

    return abs_err / (abs_err + abs_err_bench + EPSILON)


def _geometric_mean(a, axis=0, dtype=None):
    """ Geometric mean """
    if not isinstance(a, np.ndarray):  # if not an ndarray object attempt to convert it
        log_a = np.log(np.array(a, dtype=dtype))
    elif dtype:  # Must change the default dtype allowing array type
        if isinstance(a, np.ma.MaskedArray):
            log_a = np.log(np.ma.asarray(a, dtype=dtype))
        else:
            log_a = np.log(np.asarray(a, dtype=dtype))
    else:
        log_a = np.log(a)
    return np.exp(log_a.mean(axis=axis))


def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))


def nrmse(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Root Mean Squared Error """
    return rmse(actual, predicted) / (actual.max() - actual.min())


def me(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Error """
    return np.mean(_error(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(_error(actual, predicted)))


mad = mae  # Mean Absolute Deviation (it is the same as MAE)


def gmae(actual: np.ndarray, predicted: np.ndarray):
    """ Geometric Mean Absolute Error """
    return _geometric_mean(np.abs(_error(actual, predicted)))


def mdae(actual: np.ndarray, predicted: np.ndarray):
    """ Median Absolute Error """
    return np.median(np.abs(_error(actual, predicted)))


def mpe(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Percentage Error """
    return np.mean(_percentage_error(actual, predicted))


def mape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    Note: result is NOT multiplied by 100
    """
    return np.mean(np.abs(_percentage_error(actual, predicted)))


def mdape(actual: np.ndarray, predicted: np.ndarray):
    """
    Median Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.median(np.abs(_percentage_error(actual, predicted)))


def smape(actual: np.ndarray, predicted: np.ndarray):
    """
    Symmetric Mean Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.mean(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON))


def smdape(actual: np.ndarray, predicted: np.ndarray):
    """
    Symmetric Median Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.median(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON))


def maape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Arctangent Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.mean(np.arctan(np.abs((actual - predicted) / (actual + EPSILON))))


def mase(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    """
    Mean Absolute Scaled Error
    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    return mae(actual, predicted) / mae(actual[seasonality:], _naive_forecasting(actual, seasonality))


def std_ae(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Absolute Error """
    __mae = mae(actual, predicted)
    return np.sqrt(np.sum(np.square(_error(actual, predicted) - __mae))/(len(actual) - 1))


def std_ape(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Absolute Percentage Error """
    __mape = mape(actual, predicted)
    return np.sqrt(np.sum(np.square(_percentage_error(actual, predicted) - __mape))/(len(actual) - 1))


def rmspe(actual: np.ndarray, predicted: np.ndarray):
    """
    Root Mean Squared Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.sqrt(np.mean(np.square(_percentage_error(actual, predicted))))


def rmdspe(actual: np.ndarray, predicted: np.ndarray):
    """
    Root Median Squared Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.sqrt(np.median(np.square(_percentage_error(actual, predicted))))


def rmsse(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    """ Root Mean Squared Scaled Error """
    q = np.abs(_error(actual, predicted)) / mae(actual[seasonality:], _naive_forecasting(actual, seasonality))
    return np.sqrt(np.mean(np.square(q)))


def inrse(actual: np.ndarray, predicted: np.ndarray):
    """ Integral Normalized Root Squared Error """
    return np.sqrt(np.sum(np.square(_error(actual, predicted))) / np.sum(np.square(actual - np.mean(actual))))


def rrse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Relative Squared Error """
    return np.sqrt(np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual))))


def mre(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Mean Relative Error """
    return np.mean(_relative_error(actual, predicted, benchmark))


def rae(actual: np.ndarray, predicted: np.ndarray):
    """ Relative Absolute Error (aka Approximation Error) """
    return np.sum(np.abs(actual - predicted)) / (np.sum(np.abs(actual - np.mean(actual))) + EPSILON)


def mrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Mean Relative Absolute Error """
    return np.mean(np.abs(_relative_error(actual, predicted, benchmark)))


def mdrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Median Relative Absolute Error """
    return np.median(np.abs(_relative_error(actual, predicted, benchmark)))


def gmrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Geometric Mean Relative Absolute Error """
    return _geometric_mean(np.abs(_relative_error(actual, predicted, benchmark)))


def mbrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Mean Bounded Relative Absolute Error """
    return np.mean(_bounded_relative_error(actual, predicted, benchmark))


def umbrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Unscaled Mean Bounded Relative Absolute Error """
    __mbrae = mbrae(actual, predicted, benchmark)
    return __mbrae / (1 - __mbrae)


def mda(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Directional Accuracy """
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))
    
    
def nmae(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Mean Absolute Error """
    return mae(actual, predicted) / (actual.max() - actual.min())


def nmse(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Mean Squared Error """
    return mse(actual, predicted) / (actual.max() - actual.min())


def nmape(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Mean Absolute Percentage Error """
    return mape(actual, predicted) / (actual.max() - actual.min())


def gini(predicted: np.ndarray):
    """ Gini Coefficient """
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(predicted, predicted)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(predicted)
    # Gini coefficient
    g = 0.5 * rmad
    return g
    

def ngini(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Gini Coefficient """
    ng = gini(predicted)/gini(actual)
    return ng


REGR_METRICS = {
    'mse': mse,
    'rmse': rmse,
    'nrmse': nrmse,
    'me': me,
    'mae': mae,
    'mad': mad,
    'gmae': gmae,
    'mdae': mdae,
    'mpe': mpe,
    'mape': mape,
    'mdape': mdape,
    'smape': smape,
    'smdape': smdape,
    'maape': maape,
    'mase': mase,
    'std_ae': std_ae,
    'std_ape': std_ape,
    'rmspe': rmspe,
    'rmdspe': rmdspe,
    'rmsse': rmsse,
    'inrse': inrse,
    'rrse': rrse,
    'mre': mre,
    'rae': rae,
    'mrae': mrae,
    'mdrae': mdrae,
    'gmrae': gmrae,
    'mbrae': mbrae,
    'umbrae': umbrae,
    'mda': mda,
    'nmae':nmae,
    'nmse':nmse,
    'nmape':nmape,
    'gini_idx':gini,
    'ngini_idx':ngini
}


def evaluate(actual: np.ndarray, predicted: np.ndarray, metrics=('mae', 'mse', 'smape', 'umbrae')):
    results = {}
    for name in metrics:
        try:
            results[name] = REGR_METRICS[name](actual, predicted)
        except Exception as err:
            results[name] = np.nan
            print('Unable to compute metric {0}: {1}'.format(name, err))
    return results


def evaluate_all(actual: np.ndarray, predicted: np.ndarray):
    return evaluate(actual, predicted, metrics=set(REGR_METRICS.keys()))


# qt = np.array([1,2,3,3,4,5])
# qp = np.array([2,3,3,2,5,6])

# evaluate(actual=qt, predicted=qp)    
# evaluate_all(actual=qt, predicted=qp)    

    
# CLASSIFICATION METRICS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
def confusion_matrix(actual, predicted, pos_class = None, ret_obj = False):
    
    # Loading the Libraries
    import pandas as pd
    import seaborn as sn
    import matplotlib.pyplot as plt
    from collections import Counter
    
    # Creating Confusion Matrix
    data = {'y_Actual' : actual,'y_Predicted' : predicted}
    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
    conf_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    
    print("\nConfusion Matrix ==============================================\n")
    print(conf_matrix)
    
    # Metrics Calculations
    print("\nOverall Statistics ============================================\n")
    
    # Intermediate Processing ----------------------------------------------------------------
    
    # Accuracy
    tot = []
    for i in range(0,conf_matrix.shape[0]):
        tot.append(sum(conf_matrix.iloc[i,:]))
        
    # Values ---------------------------------------------------------------------------------
    
    acc=(sum(np.diag(conf_matrix))/sum(tot))*100
    ci_95_perc=0
    nir=0
    p_val=0
    kappa=0
    mcnemar_test=0
    
    ov_stat_dict = {
        '':['Accuracy','95% CI','No Information Rate','P-Value [Acc > NIR]','Kappa',"Mcnemar's Test P-Value"],
        'Value':[acc,ci_95_perc,nir,p_val,kappa,mcnemar_test]}
    
    ov_stat_df = pd.DataFrame(ov_stat_dict)
    ov_stat_df = ov_stat_df.set_index('')
    print(ov_stat_df)
    
    # Metrics Calculations
    print("\nClass Statistics ==============================================\n")
    
    cls_stat_df = pd.DataFrame({'':['# of Records (Actuals)','TP: True Positive','TN: True Negative','FP: False Positive',
    'FN: False Negative','TPR: (Sensitivity, hit rate, recall)','TNR: (Specificity)','PPV: Pos Pred Value (Precision)',
    'NPV: Neg Pred Value','FPR: False-out','FDR: False Discovery Rate','FNR: Miss Rate','ACC: Accuracy','F1 score',
    'MCC: Matthews correlation coefficient','Informedness','Markedness','Prevalence','LR+: Positive likelihood ratio',
    'LR-: Negative likelihood ratio','DOR: Diagnostic odds ratio','FOR: False omission rate']})
    
    col_list = list(np.unique(actual))
        
    # Calculating the Values
    for i in col_list:
        # No of Records
        c = Counter(actual)
        no_recs = c[i]
        
        # TP,TN,FP,FN
        tp = conf_matrix.loc[str(i),str(i)]
        tn = sum(list(np.sum(conf_matrix))) - (sum(conf_matrix.loc[:,str(i)]) + sum(conf_matrix.loc[str(i),:]))
        fp = sum(conf_matrix.loc[:,str(i)]) - conf_matrix.loc[str(i),str(i)]
        fn = sum(conf_matrix.loc[str(i),:]) - conf_matrix.loc[str(i),str(i)]
    
        # Actual Metrics
        tpr = tp/(tp+fn)
        tnr = tn/(tn+fp)
        ppv = tp/(tp+fp)
        npv = tn/(tn+fn)
        fpr = fp/(tn+fp)
        fdr = fp/(tp+fp)
        fnr = fn/(tp+fn)
        cls_acc = (tp+tn)/(tp+fp+fn+tn)
        f1_score = (2*tpr*ppv)/(tpr+ppv)
        mcc = ((tp*tn)-(fp*fn))/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
        informed = tpr+tnr-1
        mrdns = ppv+npv-1
        prev = (tp+fn)/(tp+fp+tn+fn)
        pllr = tpr/fpr
        nllr = tnr/fnr
        dor = pllr/nllr
        For = fn/(fn+tn)
        
        cls_stat_df[i] = list([no_recs,tp,tn,fp,fn,tpr,tnr,ppv,npv,fpr,fdr,fnr,cls_acc,f1_score,mcc,informed,mrdns,prev,pllr,nllr,dor,For])

    cls_stat_df = cls_stat_df.set_index('')
    print(cls_stat_df)

    # Returning the Object
    if(ret_obj == True):
        return({'conf_mat':conf_matrix, 'overall_stats':ov_stat_df, 'class_stats':cls_stat_df})
    
    
# actual = ["A", "B", "C", "B", "A", "A", "B", "A", "B", "C", "A", "B", "C","B","A","A"]
# predicted = ["A", "B", "B", "A", "C", "A", "A", "B", "B", "C", "A", "A", "C","B","C","A"]

# confusion_matrix(actual=actual, predicted=predicted)





















    


