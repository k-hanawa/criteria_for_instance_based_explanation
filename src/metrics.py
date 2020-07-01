#!/usr/bin/env python3 -u

import numpy as np

def cossim(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom > 0:
        return np.dot(a, b) / denom
    else:
        return 0


def l2sim(a, b):
    # a = a.astype(np.float64)
    # b = b.astype(np.float64)
    return -np.linalg.norm(a - b)


def influence_dot(target_data, train, model, convert):
    predicted_loss_diffs = model.get_relevance_by_grad(target_data,
                                                            train,
                                                            convert,
                                                            force_refresh=False,
                                                            approx_params={'damping': 1e-10},
                                                            matrix='inv-hessian')
    return predicted_loss_diffs

def influence_cos(target_data, train, model, convert):
    predicted_loss_diffs = model.get_relevance_by_grad(target_data,
                                                            train,
                                                            convert,
                                                            force_refresh=False,
                                                            matrix='inv_sqrt-hessian',
                                                            sim_func=cossim)
    return predicted_loss_diffs

def influence_l2(target_data, train, model, convert):
    predicted_loss_diffs = model.get_relevance_by_grad(target_data,
                                                            train,
                                                            convert,
                                                            force_refresh=False,
                                                            matrix='inv_sqrt-hessian',
                                                            sim_func=l2sim)
    return predicted_loss_diffs


def fisher_dot(target_data, train, model, convert):
    predicted_loss_diffs = model.get_relevance_by_grad(target_data,
                                                            train,
                                                            convert,
                                                            force_refresh=False,
                                                            matrix='inv-fisher',
                                                            approx_params={'damping': 1e-10})
    return predicted_loss_diffs

def fisher_cos(target_data, train, model, convert):
    predicted_loss_diffs = model.get_relevance_by_grad(target_data,
                                                            train,
                                                            convert,
                                                            force_refresh=False,
                                                            matrix='inv_sqrt-fisher',
                                                            sim_func=cossim)
    return predicted_loss_diffs


def fisher_l2(target_data, train, model, convert):
    predicted_loss_diffs = model.get_relevance_by_grad(target_data,
                                                            train,
                                                            convert,
                                                            force_refresh=False,
                                                            matrix='inv_sqrt-fisher',
                                                            sim_func=l2sim)
    return predicted_loss_diffs


def grad_dot(target_data, train, model, convert):
    predicted_loss_diffs = model.get_relevance_by_grad(target_data,
                                                            train,
                                                            convert,
                                                            force_refresh=False,
                                                            matrix='none')

    return predicted_loss_diffs

def grad_cos(target_data, train, model, convert):
    predicted_loss_diffs = model.get_relevance_by_grad(target_data,
                                                            train,
                                                            convert,
                                                            force_refresh=False,
                                                            matrix='none',
                                                            sim_func=cossim)
    return predicted_loss_diffs


def grad_l2(target_data, train, model, convert):
    predicted_loss_diffs = model.get_relevance_by_grad(target_data,
                                                            train,
                                                            convert,
                                                            force_refresh=False,
                                                            matrix='none',
                                                            sim_func=l2sim)
    return predicted_loss_diffs


def x_sim_dot(target_data, train, model, convert):
    predicted_loss_diffs = model.get_relevance_by_hsim(target_data,
                                                       train,
                                                       convert,
                                                       'x',
                                                       sim_func=np.dot)
    return predicted_loss_diffs

def x_sim_cos(target_data, train, model, convert):
    predicted_loss_diffs = model.get_relevance_by_hsim(target_data,
                                                       train,
                                                       convert,
                                                       'x',
                                                       sim_func=cossim)
    return predicted_loss_diffs

def x_sim_l2(target_data, train, model, convert):
    predicted_loss_diffs = model.get_relevance_by_hsim(target_data,
                                                       train,
                                                       convert,
                                                       'x',
                                                       sim_func=l2sim)
    return predicted_loss_diffs

def top_h_dot(target_data, train, model, convert):
    predicted_loss_diffs = model.get_relevance_by_hsim(target_data,
                                                       train,
                                                       convert,
                                                       'top_h',
                                                       sim_func=np.dot)
    return predicted_loss_diffs

def top_h_cos(target_data, train, model, convert):
    predicted_loss_diffs = model.get_relevance_by_hsim(target_data,
                                                       train,
                                                       convert,
                                                       'top_h',
                                                       sim_func=cossim)
    return predicted_loss_diffs

def top_h_l2(target_data, train, model, convert):
    predicted_loss_diffs = model.get_relevance_by_hsim(target_data,
                                                       train,
                                                       convert,
                                                       'top_h',
                                                       sim_func=l2sim)
    return predicted_loss_diffs

def all_h_dot(target_data, train, model, convert):
    predicted_loss_diffs = model.get_relevance_by_hsim(target_data,
                                                       train,
                                                       convert,
                                                       'all_h',
                                                       sim_func=np.dot)
    return predicted_loss_diffs

def all_h_cos(target_data, train, model, convert):
    predicted_loss_diffs = model.get_relevance_by_hsim(target_data,
                                                       train,
                                                       convert,
                                                       'all_h',
                                                       sim_func=cossim)
    return predicted_loss_diffs

def all_h_l2(target_data, train, model, convert):
    predicted_loss_diffs = model.get_relevance_by_hsim(target_data,
                                                       train,
                                                       convert,
                                                       'all_h',
                                                       sim_func=l2sim)
    return predicted_loss_diffs


met_names = ['grad_dot',
             'grad_cos',
             'grad_l2',
             'influence_dot',
             'influence_cos',
             'influence_l2',
             'fisher_dot',
             'fisher_cos',
             'fisher_l2',
             'x_sim_dot',
             'x_sim_cos',
             'x_sim_l2',
             'top_h_dot',
             'top_h_cos',
             'top_h_l2',
             'all_h_dot',
             'all_h_cos',
             'all_h_l2'
             ]

met_funcs = [grad_dot,
             grad_cos,
             grad_l2,
             influence_dot,
             influence_cos,
             influence_l2,
             fisher_dot,
             fisher_cos,
             fisher_l2,
             x_sim_dot,
             x_sim_cos,
             x_sim_l2,
             top_h_dot,
             top_h_cos,
             top_h_l2,
             all_h_dot,
             all_h_cos,
             all_h_l2
             ]

