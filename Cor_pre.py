import statsmodels.formula.api as smf
from sklearn import linear_model
#from sklearn.cross_validation import cross_val_predict
from sklearn.model_selection import cross_val_predict
import pandas as pd
#from scipy import stats

def training_test(training, test, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """

    from scipy import stats

    y = training.LogBIO
    X = training.drop([response], axis=1)
    remaining = set(X.columns)
    #remaining.remove(response)
    n = len(X)
    p = 0
    for i in remaining:
        p = p + 1
    lr1 = linear_model.LinearRegression()
    lr1.fit(X, y)
    predicted = lr1.predict(X)
    lr0 = linear_model.LinearRegression(fit_intercept=False)
    lr0.fit(X, y)
    predicted0 = lr0.predict(X)
    data = pd.DataFrame({'y':y, 'y1':predicted, 'y0':predicted0})
    ols = smf.OLS(y, X).fit()
    train = pd.DataFrame({'y':y, 'y1':predicted, 'y0':predicted0})
    train.to_csv('Train.csv')
    
    r2 = ols.rsquared
    r2a = ols.rsquared_adj
    r2a_m12 = 1 - (n-1)/(n-1.2*p-1)*(1-r2)
    r2a_m15 = 1 - (n-1)/(n-1.5*p-1)*(1-r2)
    r02 = smf.ols('y~y0',data).fit().rsquared
    print("r2a (tr):", r2a)
    a_ic = ols.aic
    b_ic = ols.bic
    aicc = a_ic + 2*(p+2)*(p+3) / (n-p-3)
    aic_m15 = ols.aic - (2-1.5)*p
    print("AIC (tr):", a_ic)
    print("AICc (tr):", aicc)
    print("BIC (tr):", b_ic)
    #print("AIC_m1.5 (tr):", aic_m15)
    rou = stats.spearmanr(y, predicted)
    print("rou (tr):", rou.correlation)
    slope, intercept, r_value, p_value, std_err = stats.linregress(y, predicted)
    #print("slope (training):", slope)
    #print("intercept (training):", intercept)
    import math
    rm2 = r2 * math.sqrt(abs(1-abs(r2-r02)))
    print("rm2 (tr):", rm2)
    qm2(training, 'LogBIO')
    q2_ven()

    print("----------------------------------")

    y = test.LogBIO
    X = test.drop([response], axis=1)
    predicted = lr1.predict(X)
    predicted0 = lr0.predict(X)
    data = pd.DataFrame({'y':y, 'y1':predicted, 'y0':predicted0})
    data.to_csv('Test.csv')
    r2 = smf.ols('y~y1', data).fit().rsquared
    print("r2(test):", r2)
    slope, intercept, r_value, p_value, std_err = stats.linregress(y, predicted)
    print("slope (test):", abs(slope-1))
    print("intercept (test):", intercept)
    r02 = smf.ols('y~y0',data).fit().rsquared
    import math
    rm2 = r2 * math.sqrt(abs(1-abs(r2-r02)))
    rou = stats.spearmanr(y, predicted)
    print("rm2(test):", rm2)
    print("rou(test):", rou.correlation)
    
def qm2(data, response):
    n = len(data)
    y = data.LogBIO
    X = data.drop([response], axis=1)
    lr = linear_model.LinearRegression()
    predicted = cross_val_predict(lr, X, y, cv = 5)
    #lr = linear_model.LinearRegression(fit_intercept=False)
    #predicted0 = cross_val_predict(lr, X, y, cv = n)
    #cv = pd.DataFrame({'y':y, 'y1':predicted, 'y0':predicted0})
    cv = pd.DataFrame({'y':y, 'y1':predicted})
    r2 = smf.ols('y~y1', cv).fit().rsquared
    print("q2(RAN):", r2)
    #r02 = smf.ols('y~y0',cv).fit().rsquared
                
    #import math
    #rm2 = r2 * math.sqrt(abs(1-abs(r2-r02)))
                        
    #return rm2

def q2_ven():
    cor = pd.read_csv("Correct_2.csv")

    result = pd.DataFrame(columns=['y_exp', 'y_pre'])
    
    group = 0
    va = cor[cor.index%4==group]
    tr = cor[cor.index%4!=group]
    y = tr.LogBIO
    X = tr.drop(["LogBIO"], axis=1)
    lr = linear_model.LinearRegression()
    lr.fit(X, y)
    y = va.LogBIO
    X = va.drop(["LogBIO"], axis=1)
    y_pre = lr.predict(X)
    cv = pd.DataFrame({'y_exp':y, 'y_pre':y_pre})
    result=result.append(cv)
    
    group = 1
    va = cor[cor.index%4==group]
    tr = cor[cor.index%4!=group]
    y = tr.LogBIO
    X = tr.drop(["LogBIO"], axis=1)
    lr = linear_model.LinearRegression()
    lr.fit(X, y)
    y = va.LogBIO
    X = va.drop(["LogBIO"], axis=1)
    y_pre = lr.predict(X)
    cv = pd.DataFrame({'y_exp':y, 'y_pre':y_pre})
    result=result.append(cv)
    
    group = 2
    va = cor[cor.index%4==group]
    tr = cor[cor.index%4!=group]
    y = tr.LogBIO
    X = tr.drop(["LogBIO"], axis=1)
    lr = linear_model.LinearRegression()
    lr.fit(X, y)
    y = va.LogBIO
    X = va.drop(["LogBIO"], axis=1)
    y_pre = lr.predict(X)
    cv = pd.DataFrame({'y_exp':y, 'y_pre':y_pre})
    result=result.append(cv)
    
    group = 3
    va = cor[cor.index%4==group]
    tr = cor[cor.index%4!=group]
    y = tr.LogBIO
    X = tr.drop(["LogBIO"], axis=1)
    lr = linear_model.LinearRegression()
    lr.fit(X, y)
    y = va.LogBIO
    X = va.drop(["LogBIO"], axis=1)
    y_pre = lr.predict(X)
    cv = pd.DataFrame({'y_exp':y, 'y_pre':y_pre})
    result=result.append(cv)

    r2 = smf.ols('y_exp~y_pre', result).fit().rsquared
    print("q2(VEN):", r2)

train_file = "Correct_2.csv"
training = pd.read_csv(train_file)
test_file = "Predict_2.csv"
test = pd.read_csv(test_file)

training_test(training, test, 'LogBIO')