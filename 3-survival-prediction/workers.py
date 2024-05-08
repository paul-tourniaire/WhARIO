from statistics import NormalDist

from lifelines.fitters.coxph_fitter import CoxPHFitter
from lifelines.statistics import pairwise_logrank_test
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def confidence_interval(data, confidence=0.95):
    dist = NormalDist.from_samples(data)
    z = NormalDist().inv_cdf((1 + confidence) / 2.)
    h = dist.stdev * z / ((len(data) - 1) ** .5)
    left = round(dist.mean - h, 4)
    right = round(dist.mean + h, 4)
    return left, right


def assign_risk_groups(df, quantile=0.5):
    thresh = df['predicted_risk'].quantile(q=quantile)
    df['risk_group'] = df['predicted_risk'].map(
        lambda x: 'low' if x < thresh else 'high'
    )
    return df


def log_rank_pvalue(df):
    event_durations = df['os_months']
    groups = df['risk_group']
    event_observed = df['event_observed']
    statsresults = pairwise_logrank_test(
        event_durations, groups, event_observed
    )
    return statsresults.summary.loc['high', 'p'].item()


def fit_model_cv(
        df, covariates, quantile=0.5, seed=0,
        **model_kwargs):
    '''
    Fit a CoxPH model on a 5-fold CV of df
    '''
    columns = covariates + ['os_months', 'event_observed']

    kf = KFold(5, shuffle=True, random_state=seed)
    c_index = []
    predicted_risks = []
    summary = {}
    models = []
    for i, (train_index, test_index) in enumerate(kf.split(df)):
        x_train, x_test = (
            df.iloc[train_index], df.iloc[test_index]
        )
        x_train_, x_test_ = (
            x_train.loc[:, columns], x_test.loc[:, columns]
        )
        model = CoxPHFitter(**model_kwargs)
        model.fit(
            x_train_, duration_col='os_months',
            event_col='event_observed'
        )
        c_index.append(
            model.score(
                x_test_, scoring_method='concordance_index'
            )
        )
        predicted_risks.append(
            model.predict_log_partial_hazard(x_test_)
        )
        models.append(model)

    # Predict risk on validation
    predicted_risks = pd.concat(predicted_risks).sort_index()
    df = df.assign(predicted_risk=predicted_risks)
    df = assign_risk_groups(df, quantile)

    # log-rank test of the group separation
    p_value = log_rank_pvalue(df)
    summary['c_index_mean'] = [np.mean(c_index)]
    summary['c_index_std'] = [np.std(c_index, ddof=1)]
    summary['c_index_CI'] = [confidence_interval(c_index)]
    summary['log-rank pvalue'] = [p_value]
    summary = pd.DataFrame(summary)
    
    # fold-wise results dataframe
    results = dict(
        zip(
            (f'fold {i}' for i in range(5)),
            c_index
        )
    )
    results = pd.DataFrame.from_dict(results, orient='index')

    return df, results, summary, models