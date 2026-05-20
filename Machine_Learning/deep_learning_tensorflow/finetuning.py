import numpy as np
import tensorflow as tf
import os
import re


class HyperParameters:
    def __init__(self):
        self.values = {}
        self._possible_vals = {}

    def _int_float(self, kind, name, min_value, max_value,
                   step=None, sampling="linear", default=None):
        
        val = self.values.get(name)
        if val is not None:
            return val 

        if min_value > max_value:
            raise ValueError('min_value must be smaller then max_value')

        self._possible_vals[name] = {
            'kind': kind, 
            'min': min_value, 
            'max': max_value, 
            'step': step or 1, 
            'sampling': sampling
            }
        
        dflt = min_value if default is None else default
        self.values[name] = dflt 
        return dflt
        
    def Int(self, name, min_value, max_value, step=None, sampling="linear", default=None):
        return self._int_float('int', name, min_value, max_value, step, sampling, default)
        
    def Float(self, name, min_value, max_value, step=None, sampling="linear", default=None):
        return self._int_float('float', name, min_value, max_value, step, sampling, default)

    def Choice(self, name, values, default=None):
        val = self.values.get(name)
        if val is not None:
            return val
        
        if not isinstance(values, list):
             raise TypeError('values must be a list')
        
        self._possible_vals[name] = {
             'kind': 'choice',
             'values': values
        }
        
        dflt = values[-1] if default is None else default
        self.values[name] = dflt 
        return dflt 



class RandomSearch:
    def __init__(self, hypermodel, objective, max_trials, seed=None, 
                 directory="my_dir", project_name="demo"):
        self.hypermodel = hypermodel
        self.objective = objective
        self.max_trials = max_trials
        self.seed = seed
        self.directory = directory
        self.project_name = project_name
        self._modelfname_score = {}
        self._params_score = {}
        self._buffer = set()
        self._hash_params = lambda x: hash(tuple(x.items()))

    def build(self, hp):
        return self.hypermodel(hp)
    
    def fit(self, hp, model, X, y, **kwargs):
        return model.fit(X, y, **kwargs)
        
    def search(self, X, y, **kwargs):
        np.random.seed(self.seed)
        hp = HyperParameters()
        for t in range(self.max_trials):
            model = self.build(hp)
            history = self.fit(hp, model, X, y, **kwargs)
            scores = history.history.get(self.objective)

            if scores is None:
                raise ValueError(f'Invalid objective: {self.objective}\n\
                                 Valid objective is: {list(history.history.keys())}')
            
            model_fname = f'{self.directory}/{self.project_name}/model_{t+1}.keras'
            self._modelfname_score[scores[-1]] = model_fname 
            model.save(model_fname)
            
            curr_hash_key = self._hash_params(hp.values)
            self._hash_model_params(hp, scores[-1], curr_hash_key)

            while curr_hash_key in self._buffer:
                self._set_random_values(hp)
                curr_hash_key = self._hash_params(hp.values)

    def _hash_model_params(self, hp, score, key=None):
        hash_key = self._hash_params(hp.values) if key is None else key
        self._buffer.add(hash_key)
        self._params_score[score] = hp.values

    def _set_random_values(self, hp):
            for name, params in hp._possible_vals.items():
                match params['kind']:
                    case 'choice': rnd_val = np.random.choice(params['values'])
                   
                    case 'int':
                        possible_vals = np.arange(params['min'], params['max'], params['step'])
                        match params['sampling']:
                            case 'linear': rnd_val = np.random.choice(possible_vals)
                            case 'log': 
                                low_high = np.log([params['min'], params['max']])
                                unf = np.random.uniform(*low_high)
                                x = np.exp(unf)
                                rnd_val = possible_vals[np.abs(possible_vals - x).argmin()]
                            case _: raise ValueError(f"Invalid sampling param: {params['sampling']}")

                        rnd_val = int(rnd_val)

                    case 'float':
                        match params['sampling']:
                            case 'linear': rnd_val = np.random.uniform(params['min'], params['max'])
                            case 'log': 
                                low_high = np.log([params['min'], params['max']])
                                rnd_val = np.exp(np.random.uniform(*low_high))
                            case _: raise ValueError(f"Invalid sampling param: {params['sampling']}")
                        
                        rnd_val = float(rnd_val)

                    case _:
                        raise NotImplementedError()
                    
                hp.values[name] = rnd_val

    def get_best_models(self, num_models=1):
        sorted_keys = sorted(self._modelfname_score.keys())[-num_models:]
        return [tf.keras.models.load_model(self._modelfname_score[k]) for k in sorted_keys]

    def get_best_hyperparameters(self, num_trials=1):
        sorted_keys = sorted(self._params_score.keys())[-num_trials:]
        return [self._params_score[k] for k in sorted_keys]
    



class Hyperband:
    def __init__(self, hypermodel, objective, max_epochs=10, factor=3, 
                 hyperband_iterations=2, seed=None, 
                 directory="my_dir", project_name="demo"):
        self.hypermodel = hypermodel
        self.objective = objective
        self.max_epochs = max_epochs
        self.factor = factor
        self.hyperband_iterations = hyperband_iterations
        self.seed = seed
        self.directory = directory
        self.project_name = project_name
        self._modelfname_score = {}
        self._params_score = {}
        self._buffer = set()
        self._hash_params = lambda x: hash(tuple(x.items()))

    def search(self, X, y, **kwargs): 
        rnd_search = RandomSearch(self.hypermodel, self.objective, self.max_epochs, 
                                  self.seed, self.directory, self.project_name)

        best_models = None
        for i in range(self.hyperband_iterations):
            epochs_i = int(np.ceil(self.max_epochs * (i + 1) / self.hyperband_iterations))

            if best_models is None:
                rnd_search.search(X, y, epochs=epochs_i, **kwargs)
                factor = int(np.ceil(self.max_epochs * (1/self.factor)))
                best_models = rnd_search.get_best_models(factor)
                continue
            
            scores = self._fit_models(best_models, X, y, epochs_i, **kwargs)
            factor = int(np.ceil(len(scores) * (1/self.factor)))
            best_models_inxs = scores.argsort()[-factor:]
            best_models = [best_models[j] for j in best_models_inxs]
            scores = scores[best_models_inxs]

        self.best_model_ = best_models[scores.argmax()]

    def _fit_models(self, models, X, y, epochs, **kwargs):
        scores = np.empty(len(models))

        for i, model in enumerate(models):
            history = model.fit(X, y, epochs=epochs, **kwargs)
            scores[i] = history.history[self.objective][-1]

        return scores


class StatisticCB(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        border_line = '-' * 20
        print(f'\n{border_line}')
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, tf.keras.layers.Dense):
                weights, bias = layer.get_weights()
                print(f'Layer {i}: Dense ({layer.name}) '
                      f'- W_mean: {np.mean(weights):.2f} '
                      f'b_mean: {np.mean(bias):.2f} '
                      f'- W_std: {np.std(weights):.2f} '
                      f'b_std: {np.std(bias):.2f}')
        print(f'{border_line}\n')


class GridSearch:
    def __init__(self, hypermodel, objective, max_trials=None, seed=None, 
                 directory="my_dir", project_name="demo"):
        self.hypermodel = hypermodel
        self.objective = objective
        self.max_trials = max_trials
        self.seed = seed
        self.directory = directory
        self.project_name = project_name
        self.all_combinations = None
        self.hp = None
        self.comb_counter = 0
        self._modelfname_score = {}
        self._params_score = {}
        self._statistic_cb = StatisticCB()

    def build(self, hp):
        return self.hypermodel(hp)
    
    def fit(self, hp, model, X, y, **kwargs):
        if 'callbacks' in kwargs:
            kwargs['callbacks'].append(self._statistic_cb)
        else:
            kwargs['callbacks'] = [self._statistic_cb]

        return model.fit(X, y, **kwargs)
    
    @staticmethod
    def _get_num(min, max, step):
        return int(round((max - min) / step)) + 1
    
    def _make_param_combinations(self, hp):
        from itertools import product 
        self.build(hp)

        self.params_name_list  = list(hp._possible_vals.keys())
        posible_vals_list = []

        for params in hp._possible_vals.values():
            match params['kind']:
                case 'choice': 
                    params_list = params['values']
                
                case 'int':
                    match params['sampling']:
                        case 'linear': 
                            params_list = np.arange(params['min'], 
                                                    params['max'], 
                                                    params['step']).tolist()
                        case 'log': 
                            log_min = np.log10(params['min'])
                            log_max = np.log10(params['max'])
                            params_list = np.logspace(log_min, log_max,
                                                        self._get_num(log_min, log_max,  
                                                                    params['step'])).astype(int)
                            
                        case _: raise ValueError(f"Invalid sampling param: {params['sampling']}")

                case 'float':
                    match params['sampling']:
                        case 'linear': 
                            params_list = np.linspace(params['min'],
                                                        params['max'],
                                                        self._get_num(params['min'], 
                                                                    params['max'], 
                                                                    params['step']))
                        case 'log': 
                            log_min = np.log10(params['min'])
                            log_max = np.log10(params['max'])
                            params_list = np.logspace(log_min, log_max,
                                                        self._get_num(log_min, log_max,  
                                                                    params['step']))
                            
                        case _: raise ValueError(f"Invalid sampling param: {params['sampling']}")
                    
                case _:
                    raise NotImplementedError()
            
            posible_vals_list.append(params_list)

        return list(product(*posible_vals_list))
    
    def search(self, X, y, **kwargs):
        np.random.seed(self.seed)

        if self.all_combinations is None:
            self.hp = HyperParameters()
            self.all_combinations = self._make_param_combinations(self.hp)

        border_line = '=' * 40
        self.max_trials = self.max_trials or len(self.all_combinations)

        if self.comb_counter == 0:
            files = os.listdir(f'{self.directory}/{self.project_name}')
            existing_models = []
            for file in files:
                match = re.match(r'model_(\d+)\.keras', file)
                if match:
                    existing_models.append(int(match.group(1)))

            if existing_models:
                self.comb_counter = np.max(existing_models)

        for i in range(self.comb_counter, self.max_trials):
            new_params = dict(zip(self.params_name_list, self.all_combinations[i]))
            self.hp.values.update(new_params)
            model = self.build(self.hp)
            
            print(f'\n{border_line}\nTrail {i + 1}/{self.max_trials}')
            print('\n'.join(f'{name:<30} {val}' for name, val in new_params.items()))
            print(border_line + '\n')

            history = self.fit(self.hp, model, X, y, **kwargs)
            scores = history.history.get(self.objective)

            if scores is None:
                raise ValueError(f'Invalid objective: {self.objective}\n\
                                 Valid objective is: {list(history.history.keys())}')
            
            model_fname = f'{self.directory}/{self.project_name}/model_{i+1}.keras'
            self._modelfname_score[scores[-1]] = model_fname 
            self._params_score[scores[-1]] = new_params
            model.save(model_fname)

        self.comb_counter = i + 1
        self.max_trials = None

    def get_best_models(self, num_models=1):
        sorted_keys = sorted(self._modelfname_score.keys())[-num_models:]
        return [tf.keras.models.load_model(self._modelfname_score[k]) for k in sorted_keys]

    def get_best_hyperparameters(self, num_trials=1):
        sorted_keys = sorted(self._params_score.keys())[-num_trials:]
        return [self._params_score[k] for k in sorted_keys]