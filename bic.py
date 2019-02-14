import statsmodels.formula.api as smf
from sklearn import linear_model
#from sklearn.cross_validation import cross_val_predict
import pandas as pd

def q_squared(data, response):
    n = len(data)
    y = data.LogBIO
    X = data.drop([response], axis=1)
    lr = linear_model.LinearRegression()
    predicted = cross_val_predict(lr, X, y, cv = n)
    """lr = linear_model.LinearRegression(fit_intercept=False)
    predicted0 = cross_val_predict(lr, X, y, cv = n)"""
    cv = pd.DataFrame({'y':y, 'y1':predicted})
    """cv = pd.DataFrame({'y':y, 'y1':predicted, 'y0':predicted0})
    cv = pd.DataFrame({'y':y, 'y0':predicted0})"""
    r2 = smf.ols('y~y1', cv).fit().rsquared_adj
    """r02 = smf.ols('y~y0',cv).fit().rsquared_adj"""
                
    """import math
    rm2 = r2 * math.sqrt(abs(1-abs(r2-r02)))"""
                        
    return r2

def r0_squared(data, response):
    y = data.LogBIO
    X = data.drop([response], axis=1)
    """lr1 = linear_model.LinearRegression()
    lr1.fit(X, y)
    predicted = lr1.predict(X)"""
    lr0 = linear_model.LinearRegression(fit_intercept=False)
    lr0.fit(X, y)
    predicted0 = lr0.predict(X)
    cv = pd.DataFrame({'y':y, 'y0':predicted0})
    """cv = pd.DataFrame({'y':y, 'y1':predicted, 'y0':predicted0})
    r2 = smf.ols('y~y1', cv).fit().rsquared"""
    r02 = smf.ols('y~y0',cv).fit().rsquared
                
    """import math
    rm2 = r2 * math.sqrt(abs(1-abs(r2-r02)))"""
                        
    return r02

def r_squared(data, response):
    y = data.LogBIO
    X = data.drop([response], axis=1)
    #lr1 = linear_model.LinearRegression()
    #lr1.fit(X, y)
    #predicted = lr1.predict(X)
    """lr0 = linear_model.LinearRegression(fit_intercept=False)
    lr0.fit(X, y)
    predicted0 = lr0.predict(X)
    cv = pd.DataFrame({'y':y, 'y0':predicted0})
    cv = pd.DataFrame({'y':y, 'y1':predicted, 'y0':predicted0})"""
    #cv = pd.DataFrame({'y':y, 'y1':predicted})
    r2 = smf.OLS(y,X).fit().rsquared
    """r2 = smf.ols('y~y1', cv).fit().rsquared
    radj_2 = smf.ols('y~y0',cv).fit().rsquared_adj"""
                
    """import math
    rm2 = r2 * math.sqrt(abs(1-abs(r2-r02)))"""
                        
    return r2

def ols(data, response):
    y = data.LogBIO
    X = data.drop([response], axis=1)
    #lr1 = linear_model.LinearRegression()
    #lr1.fit(X, y)
    #predicted = lr1.predict(X)
    """lr0 = linear_model.LinearRegression(fit_intercept=False)
    lr0.fit(X, y)
    predicted0 = lr0.predict(X)
    cv = pd.DataFrame({'y':y, 'y0':predicted0})
    cv = pd.DataFrame({'y':y, 'y1':predicted, 'y0':predicted0})"""
    #cv = pd.DataFrame({'y':y, 'y1':predicted})
    ols = smf.OLS(y,X).fit()
    """r2 = smf.ols('y~y1', cv).fit().rsquared
    radj_2 = smf.ols('y~y0',cv).fit().rsquared_adj"""
                
    """import math
    rm2 = r2 * math.sqrt(abs(1-abs(r2-r02)))"""
                        
    return ols

def Spearman_Correlation(data, response):
    y = data.LogBIO
    X = data.drop([response], axis=1)
    lr1 = linear_model.LinearRegression()
    lr1.fit(X, y)
    predicted = lr1.predict(X)
    import scipy.stats as stats
    corr = stats.spearmanr(y, predicted)
    return corr.correlation

def forward_selected(data, pre, response):
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
    n = len(data)
    remaining = set(data.columns)
    remaining.remove(response)
    """k = 0
    for i in remaining:
        k = k + 1"""
    #import math
    #kinit = math.log10(kinit)
    selected = ['LogBIO']
    p = 0
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates  = []
        for candidate in remaining:
            feature_cols = selected + [candidate]
            p = len(feature_cols)
            cv = pd.DataFrame(data[feature_cols])
            pre_feature = pd.DataFrame(pre[feature_cols])

            """r02 = r0_squared(cv, 'LogBIO')
            r2 = r_squared(cv, 'LogBIO')"""
            fit = ols(cv, 'LogBIO')
            #radj2 = radj2 * n
            #r = fit.rsquared
            #print("aic=",aic)
            '''r2a = fit.rsquared_adj
            rou = Spearman_Correlation(cv, 'LogBIO')
            rou = 1 - (1-rou)*(n-1)/(n-p-1)'''
            ic = fit.bic
            '''if aic > 10000:
                aic = 10000'''
            #sse = fit.ssr
            #print("r=",r)
            #aicc = aic(cv, 'LogBIO')
            #aicc = aicc + 2*(p+2)*(p+3) / (n-p-3)
            #q2 = q_squared(cv, 'LogBIO')

            """r2_avg = (r02+r2+q2) / 3
            r2_min = min(r02, r2, q2)

            d1 = abs(r02-r2)
            d2 = abs(r02-q2)
            d3 = abs(r2-q2)
            d_avg = (d1+d2+d3)/3
            d_max = max(d1, d2, d3)"""

            #import math
            #rm2 = r2 * math.sqrt(abs(1-abs(r2-r02)))
            score = 100000-ic
            #print(radj2,aicc)
            """score = r2_avg * math.sqrt(1-d_avg)
            score = r2_min * math.sqrt(1-d_max)"""
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            #delta = best_new_score - current_score
            print(100000-current_score)
            #print(score)
            cv.to_csv("correct_2.csv", index=False)
            pre_feature.to_csv("predict_2.csv", index=False)

file1 = "Correct_1.csv"
data1 = pd.read_csv(file1)
file2 = "Predict_1.csv"
data2 = pd.read_csv(file2)

"""
cut_off = [0.0006,0.0005,0.0004,0.0003,0.0002,0.0001]

result = pd.DataFrame(columns=['cut_off', 'rm2'])

for i in cut_off:
    forward_selected(data1, data2, 'LogBIO')
    tra_file = "Correct_2.csv"
    tes_file = "Predict_2.csv"
    tra_da = pd.read_csv(tra_file)
    tes_da = pd.read_csv(tes_file)
    rm2 = test(tra_da, tes_da, 'LogBIO')
    print(i,j,round(rm2,4))
    rst = pd.DataFrame({'cut_off':[i],'fold':[j],'rm2':[rm2]})
    result=result.append(rst)

result.to_csv("Result.csv")
"""

model = forward_selected(data1, data2, 'LogBIO')