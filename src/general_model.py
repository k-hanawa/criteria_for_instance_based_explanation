#!/usr/bin/env python3 -u

import chainer
import chainer.functions as F
import numpy as np
import scipy.linalg
from scipy.optimize import fmin_ncg

import os
import time
import sys

from tqdm import tqdm

is_print = True
def _print(*args, **kwargs):
    if is_print:
        print(*args, **kwargs, file=sys.stderr)


cashed = {}

class GeneralModel(chainer.Chain):

    def __init__(self, out_dir):
        super(GeneralModel, self).__init__()
        self.out_dir = out_dir

    def get_flat_param(self):
        return F.hstack([F.flatten(p) for p in self.params()])

    def get_flat_param_grad(self):
        return F.hstack([F.flatten(p.grad) for p in self.params()])

    def get_flat_param_grad_var(self):
        return F.hstack([F.flatten(p.grad_var) for p in self.params()])


    def hessian_vector_val(self, v):
        train_iter = chainer.iterators.SerialIterator(self.train_data, batch_size=self.bathsize_for_hvp, repeat=False)
        v_conv = chainer.dataset.to_device(self.device, v)

        hvp = 0
        for batch in train_iter:
            hvp_tmp = self.minibatch_hessian_vector_val(v_conv, batch)
            hvp += (hvp_tmp * len(batch) / len(self.train_data))

        return hvp + self.damping * v

    def minibatch_hessian_vector_val(self, v, batch):
        v = chainer.dataset.to_device(self.device, v)
        loss = self.__call__(*self.convertor(batch, self.device), enable_double_backprop=True)
        hvp = self.get_hvp(loss, v)
        return hvp


    def get_hvp(self, y, v):
        # First backprop
        self.cleargrads()
        y.backward(enable_double_backprop=True)
        grads = self.get_flat_param_grad_var()

        inner_product = F.sum(grads * v)

        # Second backprop
        self.cleargrads()
        grads.cleargrad()
        inner_product.backward()
        hvp = self.get_flat_param_grad()

        return chainer.cuda.to_cpu(hvp.data).astype(np.float64)


    def get_fmin_loss_fn(self, v, scale):
        def get_fmin_loss(x):
            hessian_vector_val = self.hessian_vector_val(x)
            return (0.5 * np.dot(hessian_vector_val, x) - np.dot(v, x)) * scale
        return get_fmin_loss


    def get_fmin_grad_fn(self, v, scale):
        def get_fmin_grad(x):
            hessian_vector_val = self.hessian_vector_val(x)
            return (hessian_vector_val - v) * scale
        return get_fmin_grad


    def get_fmin_hvp_fn(self, scale):
        def get_fmin_hvp(x, p):
            hessian_vector_val = self.hessian_vector_val(p)

            return hessian_vector_val * scale
        return get_fmin_hvp


    def get_cg_callback(self, v, verbose, scale):
        fmin_loss_fn = self.get_fmin_loss_fn(v, scale)

        def cg_callback(x):
            # x is current params

            if verbose:
                # _print('Function value: %s' % fmin_loss_fn(x))
                # quad, lin = fmin_loss_split(x)
                # _print('Split function value: %s, %s' % (quad, lin))
                _print(fmin_loss_fn(x))

        return cg_callback


    def get_inverse_hvp_cg(self, v, verbose, tol=1e-8, maxiter=100, batchsize=100, damping=0., scale=1e20):

        self.bathsize_for_hvp = batchsize
        self.damping = damping

        fmin_loss_fn = self.get_fmin_loss_fn(v, scale)
        fmin_grad_fn = self.get_fmin_grad_fn(v, scale)
        fmin_hvp_fn = self.get_fmin_hvp_fn(scale)
        cg_callback = self.get_cg_callback(v, verbose, scale)

        fmin_results = fmin_ncg(
            f=fmin_loss_fn,
            x0=v,
            fprime=fmin_grad_fn,
            fhess_p=fmin_hvp_fn,
            callback=cg_callback,
            avextol=tol,
            maxiter=maxiter)

        return fmin_results


    def get_hessian(self, y):
        # First backprop
        self.cleargrads()
        y.backward(enable_double_backprop=True)
        grads = self.get_flat_param_grad_var()

        hessian_list = []
        for g in tqdm(grads):
            self.cleargrads()
            g.backward()
            hessian_list.append(chainer.cuda.to_cpu(self.get_flat_param_grad().data))
        hessian = np.vstack(hessian_list)
        return hessian


    def get_inverse_hvp_lissa(self, v,
                              batch_size=None,
                              scale=10, damping=0.0, num_samples=1, recursion_depth=1000):
        """
        This work
        """
        inverse_hvp = 0
        print_iter = 100

        for i in range(num_samples):
            cur_estimate = v
            train_iter = chainer.iterators.SerialIterator(self.train_data, batch_size=batch_size, repeat=True)
            for j, batch in enumerate(train_iter):

                hessian_vector_val = self.minibatch_hessian_vector_val(cur_estimate, batch)
                cur_estimate = v + (1 - damping) * cur_estimate - hessian_vector_val / scale

                # Update: v + (I - Hessian_at_x) * cur_estimate
                if (j % print_iter == 0) or (j == recursion_depth - 1):
                    _print(j, np.linalg.norm(cur_estimate))
                    # _print("Recursion at depth %s: norm is %.8lf" % (j, np.linalg.norm(np.concatenate(cur_estimate))))

                if j >= recursion_depth:
                    break

            inverse_hvp += (cur_estimate / scale)

        inverse_hvp = inverse_hvp / num_samples
        return inverse_hvp


    def get_inverse_hvp(self, v, approx_type='cg', approx_params=None, verbose=True):
        assert approx_type in ['cg', 'lissa']
        if approx_type == 'lissa':
            return self.get_inverse_hvp_lissa(v, **approx_params)
        elif approx_type == 'cg':
            return self.get_inverse_hvp_cg(v, verbose, **approx_params)


    def get_grad_loss(self, test_data):
        loss = self.__call__(*self.convertor([test_data], self.device))
        self.cleargrads()
        loss.backward()
        grad = self.get_flat_param_grad()
        return chainer.cuda.to_cpu(grad.data)

    def get_relevance_by_grad(self, test_data, train_data, convertor, matrix='none',
        approx_type='cg', approx_params={}, force_refresh=False,
        test_description='', sim_func=np.dot):

        self.train_data = train_data
        self.convertor = convertor

        if matrix == 'approx_hessian':
            start_time = time.time()

            test_grad = self.get_grad_loss(test_data)
            test_grad = test_grad.astype(np.float64)

            approx_filename = os.path.join(self.out_dir, '{}-{}.npy'.format(
            approx_type, test_description))
            if os.path.exists(approx_filename) and force_refresh == False:
                test_feature = np.load(approx_filename)
                _print('Loaded from {}'.format(approx_filename))
            else:
                test_feature = self.get_inverse_hvp(
                    test_grad,
                    approx_type,
                    approx_params)
                np.save(approx_filename, test_feature)
                _print('Saved to {}'.format(approx_filename))

            duration = time.time() - start_time
            _print('Inverse HVP took {} sec'.format(duration))

        elif matrix == 'inv-hessian':
            start_time = time.time()

            test_grad = self.get_grad_loss(test_data)
            test_grad = test_grad.astype(np.float64)

            hess_filename = os.path.join(self.out_dir, 'inv-hessian-matrix.npy')
            if hess_filename in cashed:
                inv_hessian = cashed[hess_filename]
            elif os.path.exists(hess_filename) and force_refresh == False:
                inv_hessian = np.load(hess_filename)
                _print('Loaded from {}'.format(hess_filename))
            else:
                # train_iter = chainer.iterators.SerialIterator(train_data, batch_size=len(train_data), repeat=False)
                loss = self.__call__(*convertor(train_data, self.device), enable_double_backprop=True)
                hessian = self.get_hessian(loss)
                damped_hessian = hessian + np.mean(np.abs(hessian)) * approx_params.get('damping', 0) * np.identity(hessian.shape[0])
                inv_hessian = np.linalg.inv(damped_hessian)
                np.save(hess_filename, inv_hessian)
                _print('Saved to {}'.format(hess_filename))

            if hess_filename not in cashed:
                cashed[hess_filename] = inv_hessian

            test_feature = np.matmul(inv_hessian, test_grad)

            duration = time.time() - start_time
            _print('took {} sec'.format(duration))
        elif matrix == 'inv_sqrt-hessian':
            start_time = time.time()

            test_grad = self.get_grad_loss(test_data)
            test_grad = test_grad.astype(np.float64)

            hess_filename = os.path.join(self.out_dir, 'inv_sqrt-hessian-matrix.npy')
            if hess_filename in cashed:
                inv_sqrt_hessian = cashed[hess_filename]
            elif os.path.exists(hess_filename) and force_refresh == False:
                inv_sqrt_hessian = np.load(hess_filename)
                _print('Loaded from {}'.format(hess_filename))
            else:
                # train_iter = chainer.iterators.SerialIterator(train_data, batch_size=len(train_data), repeat=False)
                inv_hessian_fn = os.path.join(self.out_dir, 'inv-hessian-matrix.npy')
                inv_hessian = np.load(inv_hessian_fn)
                inv_sqrt_hessian = scipy.linalg.sqrtm(inv_hessian).real
                np.save(hess_filename, inv_sqrt_hessian)
                inv_sqrt_hessian = np.load(hess_filename)
                _print('Saved to {}'.format(hess_filename))

            if hess_filename not in cashed:
                cashed[hess_filename] = inv_sqrt_hessian

            test_feature = np.matmul(inv_sqrt_hessian, test_grad)

            duration = time.time() - start_time
            _print('took {} sec'.format(duration))
        elif matrix == 'inv-fisher':
            start_time = time.time()

            test_grad = self.get_grad_loss(test_data)
            test_grad = test_grad.astype(np.float64)

            fisher_filename = os.path.join(self.out_dir, 'inv-fisher-matrix.npy')
            if fisher_filename in cashed:
                inv_fisher = cashed[fisher_filename]
            elif os.path.exists(fisher_filename) and force_refresh == False:
                inv_fisher = np.load(fisher_filename)
                _print('Loaded from {}'.format(fisher_filename))
            else:
                # train_iter = chainer.iterators.SerialIterator(train_data, batch_size=len(train_data), repeat=False)
                train_grad_filename = os.path.join(self.out_dir, 'train-grad-on-loss-all.npy')
                if not os.path.exists(train_grad_filename):
                    _print('must calculate train grads first')
                    sys.exit(1)
                train_grads = np.load(train_grad_filename)
                avg_grads = np.average(train_grads, axis=0)
                fisher_matrix = []
                for g in avg_grads:
                    fisher_matrix.append(g * avg_grads)
                fisher = np.vstack(fisher_matrix)
                damped_fisher = fisher + np.mean(np.abs(fisher)) * approx_params.get('damping', 0) * np.identity(fisher.shape[0])
                inv_fisher = np.linalg.inv(damped_fisher)
                np.save(fisher_filename, inv_fisher)
                _print('Saved to {}'.format(fisher_filename))

            if fisher_filename not in cashed:
                cashed[fisher_filename] = inv_fisher

            test_feature = np.matmul(inv_fisher, test_grad)

            duration = time.time() - start_time
            _print('took {} sec'.format(duration))
        elif matrix == 'inv_sqrt-fisher':
            start_time = time.time()

            test_grad = self.get_grad_loss(test_data)
            test_grad = test_grad.astype(np.float64)

            hess_filename = os.path.join(self.out_dir, 'inv_sqrt-fisher-matrix.npy')
            if hess_filename in cashed:
                inv_sqrt_fisher = cashed[hess_filename]
            elif os.path.exists(hess_filename) and force_refresh == False:
                inv_sqrt_fisher = np.load(hess_filename)
                _print('Loaded from {}'.format(hess_filename))
            else:
                # train_iter = chainer.iterators.SerialIterator(train_data, batch_size=len(train_data), repeat=False)
                inv_fisher_fn = os.path.join(self.out_dir, 'inv-fisher-matrix.npy')
                inv_fisher = np.load(inv_fisher_fn)
                inv_sqrt_fisher = scipy.linalg.sqrtm(inv_fisher).real
                np.save(hess_filename, inv_sqrt_fisher)
                inv_sqrt_fisher = np.load(hess_filename)
                _print('Saved to {}'.format(hess_filename))

            if hess_filename not in cashed:
                cashed[hess_filename] = inv_sqrt_fisher

            test_feature = np.matmul(inv_sqrt_fisher, test_grad)

            duration = time.time() - start_time
            _print('took {} sec'.format(duration))
        elif matrix == 'none':
            test_grad = self.get_grad_loss(test_data)
            test_grad = test_grad.astype(np.float64)
            test_feature = test_grad
        else:
            sys.exit(1)

        start_time = time.time()

        predicted_loss_diffs = []

        if matrix == 'inv_sqrt-hessian':
            train_feature_filename = os.path.join(self.out_dir, 'train-grad-on-loss-all_inv_sqrt_hessian.npy')

            if train_feature_filename in cashed:
                train_features = cashed[train_feature_filename]
            elif os.path.exists(train_feature_filename):
                train_features = np.load(train_feature_filename)
                _print('Loaded train grads from {}'.format(train_feature_filename))
            else:
                _train_feature_filename = os.path.join(self.out_dir, 'train-grad-on-loss-all.npy')
                train_features = np.load(_train_feature_filename)
                train_features = np.matmul(inv_sqrt_hessian, train_features.T).T
                np.save(train_feature_filename, train_features)
                _print('Saved train grads to {}'.format(train_feature_filename))

            if train_feature_filename not in cashed:
                cashed[train_feature_filename] = train_features

        elif matrix == 'inv_sqrt-fisher':
            train_feature_filename = os.path.join(self.out_dir, 'train-grad-on-loss-all_inv_sqrt_fisher.npy')

            if train_feature_filename in cashed:
                train_features = cashed[train_feature_filename]
            elif os.path.exists(train_feature_filename):
                train_features = np.load(train_feature_filename)
                _print('Loaded train grads from {}'.format(train_feature_filename))
            else:
                _train_feature_filename = os.path.join(self.out_dir, 'train-grad-on-loss-all.npy')
                train_features = np.load(_train_feature_filename)
                train_features = np.matmul(inv_sqrt_fisher, train_features.T).T
                np.save(train_feature_filename, train_features)
                _print('Saved train grads to {}'.format(train_feature_filename))

            if train_feature_filename not in cashed:
                cashed[train_feature_filename] = train_features

        else:
            train_feature_filename = os.path.join(self.out_dir, 'train-grad-on-loss-all.npy')

            if train_feature_filename in cashed:
                train_features = cashed[train_feature_filename]
            elif os.path.exists(train_feature_filename):
                train_features = np.load(train_feature_filename)
                _print('Loaded train grads from {}'.format(train_feature_filename))
            else:
                train_feature_list = []
                for counter, remove_data in enumerate(tqdm(train_data)):
                    train_feature = self.get_grad_loss(remove_data)
                    train_feature_list.append(train_feature)
                train_features = np.vstack(train_feature_list)
                np.save(train_feature_filename, train_features)
                _print('Saved train grads to {}'.format(train_feature_filename))

            if train_feature_filename not in cashed:
                cashed[train_feature_filename] = train_features

        for counter, train_feature in enumerate(tqdm(train_features, leave=False)):
            predicted_loss_diffs.append(sim_func(test_feature, train_feature))

        duration = time.time() - start_time
        _print('Multiplying by all train examples took {} sec'.format(duration))

        return predicted_loss_diffs


    def get_relevance_by_hsim(self, test_data, train_data, convertor, h_fn, batch_size=100, sim_func=np.dot):
        if h_fn == 'top_h':
            h_func = self.get_top_h
        elif h_fn == 'all_h':
            h_func = self.get_all_h
        elif h_fn == 'x':
            h_func = self.get_x
        # elif h_fn == 'residual':
        #     h_func = self.get_residual
        else:
            sys.exit(1)

        test_feature = h_func(*convertor([test_data], self.device))[0]

        feature_sims = []
        train_feature_filename = os.path.join(self.out_dir, 'train-{}-all.npy'.format(h_fn))

        if train_feature_filename in cashed:
            train_features = cashed[train_feature_filename]
        elif os.path.exists(train_feature_filename):
            train_features = np.load(train_feature_filename)
            _print('Loaded train features from {}'.format(train_feature_filename))
        else:
            train_feature_list = []
            train_iter = chainer.iterators.SerialIterator(train_data, batch_size=batch_size, repeat=False, shuffle=False)
            for batch in tqdm(train_iter):
                features = h_func(*convertor(batch, self.device))
                train_feature_list.append(features)
            train_features = np.vstack(train_feature_list)
            np.save(train_feature_filename, train_features)
            _print('Saved train features to {}'.format(train_feature_filename))
        if train_feature_filename not in cashed:
            cashed[train_feature_filename] = train_features

        for counter, train_feature in enumerate(tqdm(train_features, leave=False)):
            feature_sims.append(sim_func(test_feature, train_feature))

        return feature_sims
