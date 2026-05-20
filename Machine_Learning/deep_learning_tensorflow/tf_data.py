import tensorflow as tf
from abc import ABC, abstractmethod
import glob
from copy import deepcopy
from collections import deque, defaultdict
import os
import pickle
from threading import Thread
from queue import Queue
import math
import pandas as pd


class Dataset:
    def __init__(self, data_gen=None):
        self.data_gen = data_gen
    
    @classmethod
    def from_tensor_slices(cls, tensors):
        if isinstance(tensors, (list, tf.Tensor)):
            return TensorSlice(tf.convert_to_tensor(tensors))

        elif isinstance(tensors, tuple):
            return TupleSlice(tensors)
        
        elif isinstance(tensors, dict):
            tensors = {key: tf.convert_to_tensor(t) for key, t in tensors.items()}
            return DictSlice(tensors)

        elif isinstance(tensors, (pd.Series, pd.DataFrame)):
            tensors = tf.convert_to_tensor(tensors.to_numpy())
            return PandasSlice(tensors)

        raise NotImplementedError()
    
    @classmethod
    def list_files(cls, file_pattern, shuffle=False, seed=None):
        return FileDataset(file_pattern, shuffle, seed)
    
    def repeat(self, count=-1):
        return RepeatDataset(count, self)

    def batch(self, size, drop_remainder=False):
        return BatchDataset(size, drop_remainder, self)

    def take(self, count):
        return TakeDataset(count, self)
    
    def map(self, func):
        return MapDataset(func, self)

    def filter(self, func):
        return FilterDataset(func, self)

    def shuffle(self, buffer_size, seed=None):
        return ShuffleDataset(buffer_size, seed, self)
    
    def interleave(self, func, cycle_length):
        return InterleaveDataset(func, cycle_length, self)

    @classmethod
    def range(cls, *args, **kwargs):
        return RangeDataset(*args, **kwargs)

    def concatenate(self, dataset):
        return ConcatenateDataset(dataset, self)
    
    def window(self, size, shift=None, drop_remainder=False):
        return WindowDataset(size, shift, drop_remainder, self)

    def reduce(self, initial_state, reduce_func):
        # return ReduceDataset(initial_state, reduce_func, self)

        temp_state = deepcopy(initial_state)

        for item in self:
            temp_state = reduce_func(temp_state, item)

        return temp_state
    
    def shard(self, num_shards, index):
        return ShardDataset(num_shards, index, self)

    def apply(self, transformation_func):
        return transformation_func(self)

    def padded_batch(self, batch_size, padded_shapes=None, padding_values=0):
        return PaddedBatchDataset(batch_size, padded_shapes, padding_values, self)

    @classmethod
    def from_generator(cls, generator, output_signature):
        return FromGeneratorDataset(generator, output_signature)

    def prefetch(self, buffer_size):
        return PrefetchDataset(buffer_size, self)

    def cache(self, filename=''):
        return CacheDataset(filename, self)
    
    def skip(self, count):
        return SkipDataset(count, self)

    def stratify(self, n_classes, train_set_ratio=0.8):
        train_set_per_class = []
        test_set_per_class = []

        for cls in range(n_classes):
            cls_tensor = tf.constant(cls, dtype=tf.int32)

            class_ds = self.filter(lambda item, cls=cls_tensor: tf.equal(tf.cast(item[1], tf.int32), cls))
            class_smaples_count = class_ds.reduce(0, lambda state, _: state + 1)
            
            trainset_size = int(class_smaples_count * train_set_ratio)
            train_set_per_class.append(class_ds.take(trainset_size))
            test_set_per_class .append(class_ds.skip(trainset_size))


        train_ds = train_set_per_class[0]
        test_ds  = test_set_per_class[0]

        for i in range(1, n_classes):
            train_ds = train_ds.concatenate(train_set_per_class[i])
            test_ds  = test_ds .concatenate(test_set_per_class[i])

        return train_ds, test_ds

    def flat_map(self, map_func):
        return FlatMapDataset(map_func, self)

    def __iter__(self):
        for item in self.data_gen:
            yield item


class FlatMapDataset(Dataset):
    def __init__(self, map_func, data_gen=None):
        super().__init__(data_gen)

        if not callable(map_func):
            raise TypeError(f'map_func must be callable not: {type(map_func)}')
        
        self.map_func = map_func

    def __iter__(self):
        for item in self.data_gen:
            ds = self.map_func(item)
            
            if not isinstance(ds, Dataset):
                raise TypeError(f'map_func must return Dataset object not: {type(ds)}')
            
            for el in ds:
                yield el

class SkipDataset(Dataset):
    def __init__(self, count, data_gen=None):
        super().__init__(data_gen)
        self.count = count

    def __iter__(self):
        for i, item in enumerate(self.data_gen):
            if i < self.count:
                continue

            yield item

class CacheDataset(Dataset):
    def __init__(self, filename, data_gen=None):
        super().__init__(data_gen)
        self.filename = filename
        self.buffer = []

    def __iter__(self):
        if not self.filename:
            if self.buffer:
                for item in self.buffer:
                    yield item

            else:
                for item in self.data_gen:
                    yield item
                    self.buffer.append(item)

        else:
            if os.path.getsize(self.filename) == 0:
                with open(self.filename, "ab") as f: 
                    for item in self.data_gen:
                        yield item
                        pickle.dump(item, f)

            else:
                with open(self.filename, "rb") as f: 
                    while True:
                        try:
                            yield pickle.load(f)
                        except EOFError:
                            break

class PrefetchDataset(Dataset):
    def __init__(self, buffer_size, data_gen=None):
        super().__init__(data_gen)
        self.buffer_size = buffer_size

    # def __iter__(self):
    #     buffer = deque()

    #     for i, item in enumerate(self.data_gen):
    #         if i < self.buffer_size:
    #             buffer.append(item)
    #             continue

    #         yield buffer.popleft()
    #         buffer.append(item)

    #     while buffer:
    #         yield buffer.popleft()

    def __iter__(self):
        buffer = Queue(self.buffer_size)

        def background_thread():
            for item in self.data_gen:
                buffer.put(item)

            buffer.put(None)

        Thread(target=background_thread, daemon=True).start()

        while True:
            item = buffer.get()

            if item is None:
                break

            yield item

class FromGeneratorDataset(Dataset):
    def __init__(self, generator, output_signature):
        super().__init__()

        if not isinstance(output_signature, (list, tuple)):
            output_signature = [output_signature]

        if not all(isinstance(obj, (tf.TensorSpec, tf.RaggedTensorSpec))
                              for obj in output_signature):
            raise TypeError('output_signature must contains only Spec objects')

        self.generator = generator
        self.output_signature = output_signature

    @staticmethod
    def to_tensor_or_ragged(el, spec):
        if isinstance(el, (tf.Tensor, tf.RaggedTensor)):
            return el
        
        return tf.constant(el, dtype=spec.dtype, shape=spec.shape)

    def __iter__(self):
        for item in self.generator():
            if not isinstance(item, (list, tuple)):
                item = [item]

            next_item = [self.to_tensor_or_ragged(el, spec) 
                         for el, spec in zip(item, self.output_signature)]

            if len(next_item) == 1:
                next_item = next_item[0]

            yield next_item

class PaddedBatchDataset(Dataset):
    def __init__(self, batch_size, padded_shapes=None, padding_values=0, data_gen=None):
        super().__init__(data_gen)
        self.batch_size = batch_size
        self.padded_shapes = padded_shapes
        self.padding_values = padding_values

    def _batch_prepare(self, batch, longest_shape):
        if self.padded_shapes is not None:
            longest_shape = self.padded_shapes

        for i, el in enumerate(batch):
            if el.shape[0] < longest_shape:
                padding_shape = [longest_shape - el.shape[0]]
                padding = tf.fill(padding_shape, self.padding_values)
                batch[i] = tf.concat([el, padding], 0)

        return batch

    def __iter__(self):
        batch = []
        longest_shape = 0

        for item in self.data_gen:
            batch.append(item)
            item_shape = item.shape
            
            if not len(item_shape):
                raise ValueError('not compatible with the shape ()')
            
            longest_shape = item_shape[0] if item_shape[0] > longest_shape else longest_shape 

            if len(batch) == self.batch_size:
                batch = self._batch_prepare(batch, longest_shape)
                yield tf.stack(batch)
                batch = []
                longest_shape = 0

        if batch:
            batch = self._batch_prepare(batch, longest_shape)
            yield tf.stack(batch)

class ShardDataset(Dataset):
    def __init__(self, num_shards, index, data_gen=None):
        super().__init__(data_gen)
        self.num_shards = num_shards
        self.index = index

    def __iter__(self):
        counter = 0
        flag = True

        for i, item in enumerate(self.data_gen):
            if i >= self.index and flag:
                yield item
                flag = False
                counter = 0

            counter += 1
            if counter == self.num_shards:
                flag = True
            
class ReduceDataset(Dataset):
    def __init__(self, initial_state, reduce_func, data_gen=None):
        super().__init__(data_gen)

        if not callable(reduce_func):
            raise TypeError(f'reduce_func must be callable not: {type(reduce_func)}')
        
        self.initial_state = initial_state
        self.reduce_func = reduce_func        

    def __iter__(self):
        temp_state = deepcopy(self.initial_state)

        for item in self.data_gen:
            temp_state = self.reduce_func(temp_state, item)

        yield temp_state

class WindowDataset(Dataset):
    def __init__(self, size, shift=None, drop_remainder=False, data_gen=None):
        super().__init__(data_gen)

        self.shift = size if shift is None else shift
        self.drop_remainder = drop_remainder
        self.size  = size

    def __iter__(self):
        window = []

        for item in self.data_gen:
            window.append(item)

            if len(window) == self.size:
                yield Dataset.from_tensor_slices(window)
                window = window[self.shift:]

        if not self.drop_remainder:
            while window:
                yield Dataset.from_tensor_slices(window)
                window = window[self.shift:]

class ConcatenateDataset(Dataset):
    def __init__(self, dataset, data_gen=None):
        super().__init__(data_gen)

        if not isinstance(dataset, Dataset):
            raise TypeError(f'dataset must be of type Dataset not: {type(dataset)}')
        
        self.dataset = dataset

    def __iter__(self):
        for item in self.data_gen:
            yield item

        for item in self.dataset:
            yield item

class RangeDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return iter(tf.range(*self.args, **self.kwargs))

class RepeatDataset(Dataset):
    def __init__(self, count, data_gen=None):
        super().__init__(data_gen)
        self.count = count

    def __iter__(self):
        for _ in tf.range(self.count):
            for item in self.data_gen:
                yield item
            
class BatchDataset(Dataset): 
    def __init__(self, size, drop_remainder=False, data_gen=None):
        super().__init__(data_gen)
        self.size = size
        self.drop_remainder = drop_remainder

    def stack_batch(self, batch, item):
        if isinstance(item, tf.Tensor):
            return tf.stack(batch)
        
        if isinstance(item, tuple):
            return tuple(tf.stack(trs) for trs in zip(*batch)) 
        
        if isinstance(item, dict):
            out = defaultdict(list)

            for el in batch:
                for key, val in el.items():
                    out[key].append(val)

            for key, vals in out.items():
                out[key] = tf.stack(vals)

            return out 
        
        raise NotImplemented

    def __iter__(self):
        batch = []
        
        for item in self.data_gen:
            batch.append(item)

            if len(batch) == self.size:
                yield self.stack_batch(batch, item)
                batch = []

        if batch and not self.drop_remainder:
            yield self.stack_batch(batch, item)
            batch = []

class TakeDataset(Dataset):
    def __init__(self, count, data_gen=None):
        super().__init__(data_gen)
        self.count = count

    def __iter__(self):
        counter = 0

        for item in self.data_gen:
            yield item

            counter += 1

            if counter == self.count:
                break

class MapDataset(Dataset):
    def __init__(self, func, data_gen=None):
        super().__init__(data_gen)

        if not callable(func):
            raise TypeError(f'Invalid func type: {type(func)}')
        
        self.func = func

    def __iter__(self):
        for item in self.data_gen:
            yield self.func(item)

class FilterDataset(Dataset):
    def __init__(self, func, data_gen=None):
        super().__init__(data_gen)

        if not callable(func):
            raise TypeError(f'Invalid func type: {type(func)}')
        
        self.func = func

    def __iter__(self):
        for item in self.data_gen:
            if self.func(item):
                yield item

class ShuffleDataset(Dataset):
    def __init__(self, buffer_size, seed=None, data_gen=None):
        super().__init__(data_gen)
        self.buffer_size = buffer_size
        tf.random.set_seed(seed)
        self._get_rnd_inx = lambda buffer_len: tf.random.uniform(shape=(), 
                                                                minval=0,
                                                                maxval=buffer_len,
                                                                dtype=tf.int32)

    def __iter__(self):
        buffer = []

        for item in self.data_gen:
            if len(buffer) < self.buffer_size:
                buffer.append(item)
                continue

            inx = self._get_rnd_inx(len(buffer))
            next_item = buffer.pop(inx)
            buffer.append(item)
            yield next_item

        while buffer:
            inx = self._get_rnd_inx(len(buffer))
            yield buffer.pop(inx)
            
class TensorSlice(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __iter__(self):
        for item in self.data:
            yield item
    
class TupleSlice(Dataset):
    def __init__(self, data):
        super().__init__()

        self.data = data
        self.len = None
        self.check_size()

    def __iter__(self):
        for i in tf.range(self.len):
            yield tuple(t[i] for t in self.data)

    def check_size(self):
        self.len = self.data[0].shape[0]

        for tensor in self.data[1:]:
            if (c := tensor.shape[0]) != self.len:
                raise ValueError(f'Invalid data shape expect: {self.len}; receive: {c}')
            
class PandasSlice(Dataset):
    def __init__(self, data):
        super().__init__()

        self.data = data

    def __iter__(self):
        for item in self.data:
            yield item

class DictSlice(Dataset):
    def __init__(self, data):
        super().__init__()

        self.data = data
        self.len = None
        self.check_size()

    def __iter__(self):
        for i in tf.range(self.len):
            yield dict((key, t[i]) for key, t in self.data.items())

    def check_size(self):
        dict_iter = iter(self.data.values())
        self.len = next(dict_iter).shape[0]
        
        for tensor in dict_iter:
            if (c := tensor.shape[0]) != self.len:
                raise ValueError(f'Invalid data shape expect: {self.len}; receive: {c}')

class FileDataset(Dataset):
    def __init__(self, files, shuffle, seed):
        if not isinstance(files, list):
            files = [files]

        self.files = [glob.glob(f) for f in files]
        self.shuffle = shuffle
        tf.random.set_seed(seed)

        self.n_files = len(self.files)
        inxs = tf.range(self.n_files)

        if self.shuffle:
            inxs = tf.random.shuffle(inxs)

        self.inxs = inxs
        self.inx = 0

    def reset_state(self):
        self.inx = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.inx == self.n_files:
            self.reset_state()
            raise StopIteration
        
        res = tf.constant(self.files[self.inxs[self.inx]])
        self.inx += 1
        return res

class InterleaveDataset(Dataset):
    def __init__(self, func, cycle_length, data_gen=None):
        super().__init__(data_gen)
        self.func = func
        self.cycle_length = cycle_length
        self.to_str = lambda filepath: filepath.numpy().item().decode("utf-8")

    def __iter__(self):
        datasets = [self.func(self.to_str(item)) for item, _ in zip(self.data_gen, tf.range(self.cycle_length))]

        while datasets:
            for i, dataset in enumerate(datasets):
                line = next(dataset, None)
                
                if line is None:
                    datasets.pop(i)
                    next_dataset = next(self.data_gen, None)

                    if next_dataset is not None:
                        datasets.append(self.func(self.to_str(next_dataset)))

                    continue

                yield line

class TextLineDataset:
    def __init__(self, filepaths):
        self.filepaths = filepaths

        self._skip = 0
        self._skip_counter = 0
        
    def skip(self, count):
        self._skip = count
        return self

    def __iter__(self):
        return self
    
    def __next__(self):
        if not hasattr(self, 'file'):
            self.file = open(self.filepaths)

        line = self.file.readline()

        while self._skip_counter < self._skip:
            line = self.file.readline()
            self._skip_counter += 1

        if not line:
            self.file.close()
            raise StopIteration

        return line 


class CsvDataset(Dataset):
    def __init__(self, 
                 filenames, 
                 record_defaults,
                 header=False,
                 field_delim=',',
                 use_quote_delim=True,
                 na_value='',
                 select_cols=None,
                 exclude_cols=None
                ):
        
        super().__init__()
        
        if (select_cols is not None or exclude_cols is not None) and not header:
            raise ValueError("Column selection by name requires header=True") 
        
        if select_cols is not None and exclude_cols is not None:
            raise ValueError("select_cols and exclude_cols cannot be used together")

        self.header          = header
        self.na_value        = na_value
        self.filenames       = filenames
        self.field_delim     = field_delim
        self.select_cols     = select_cols
        self.exclude_cols    = exclude_cols 
        self.record_defaults = record_defaults
        self.use_quote_delim = use_quote_delim

        self._col_names = None
        self._selected_cols_inxs = tf.range(len(self.record_defaults))
        self._col_types = [tf.convert_to_tensor(dfv).dtype 
                           for dfv in self.record_defaults]
    
    def _make_row(self, raw_row):
        split_row = tf.strings.split(raw_row, sep=self.field_delim)

        def row_gen():
            for i in self._selected_cols_inxs:
                col = split_row[i]

                if not col or col == self.na_value:
                    yield self.record_defaults[i]
                elif self._col_types[i] == tf.string:
                    yield col
                else:
                    yield tf.strings.to_number(col, self._col_types[i])

        return tuple(row_gen())
    
    def _column_selection(self, header_line):
        if self.select_cols is None and self.exclude_cols is None:
            return 
        
        self._col_names = tf.strings.split(header_line, sep=self.field_delim)
        
        inxs = []
        for i, name in enumerate(self._col_names):
            if self.select_cols is not None and name in self.select_cols:
                inxs.append(i)

            if self.exclude_cols is not None and name not in self.exclude_cols:
                inxs.append(i)

        self._selected_cols_inxs = tf.constant(inxs)

    def __iter__(self):
        filepaths = tf.io.gfile.glob(self.filenames)

        for filepath in filepaths:
            first_row = True

            with open(filepath, 'r') as file:
                for line in file:

                    if first_row:
                        first_row = False

                        if self.header:
                            self._column_selection(line)
                            continue

                    yield self._make_row(line)
 


def make_csv_dataset(file_pattern,
                     batch_size,
                     column_names=None,
                     column_defaults=None,
                     label_name=None,
                     select_columns=None,
                     field_delim=',',
                     use_quote_delim=True,
                     na_value='',
                     header=True,
                     num_epochs=None,
                     shuffle=True,
                     shuffle_buffer_size=10000,
                     shuffle_seed=None,
                     encoding='utf-8'):
    
    if column_names is None and not header:
        raise ValueError('if header=False must input column_names')
    
    filepaths = tf.io.gfile.glob(file_pattern)

    ds_stack = None
    for filepath in filepaths:
        with open(filepath, 'r', encoding=encoding) as file:
            header_line = file.readline()

            if column_names is None:
                column_names = tf.strings.split(header_line, sep=field_delim)

            row = file.readline()

            if column_defaults is None:
                row_split = tf.strings.split(row, sep=field_delim)

                column_defaults = []
                for col in row_split:
                    is_int = tf.strings.regex_full_match(col, r"-?\d+")
                    is_float = tf.strings.regex_full_match(col, r"-?\d*\.\d+")

                    if is_int:
                        val = 0
                    elif is_float:
                        val = 0.0
                    else:
                        val = ''

                    column_defaults.append(val)

        ds = CsvDataset(filepath, 
                        record_defaults=column_defaults,
                        header=header,
                        field_delim=field_delim,
                        use_quote_delim=use_quote_delim,
                        na_value=na_value,
                        select_cols=select_columns)
        
        if num_epochs is not None:
            ds = ds.repeat(num_epochs)

        if shuffle:
            ds = ds.shuffle(shuffle_buffer_size, shuffle_seed)

        ds = ds.batch(batch_size)
        ds = ds.map(lambda row: dict((name.numpy().decode("utf-8"), col) 
                                     for name, col in zip(column_names, row)))

        def return_X_y(row):
            y = row[label_name]
            del row[label_name]
            return (row, y)

        if label_name is not None:
            ds = ds.map(return_X_y)

        if ds_stack is None:
            ds_stack = ds
        else:
            ds_stack.concatenate(ds)

    return ds_stack





def timeseries_dataset_from_array(data, targets, 
                                  sequence_length, 
                                  batch_size=32, 
                                  shuffle=False, 
                                  seed=None):
    data = tf.convert_to_tensor(data)

    time_data = []
    data_len = len(data)

    for i in tf.range(data_len - sequence_length):
        start = i
        end   = start + sequence_length

        part = data[start:end]
        time_data.append(part)

    time_data = tf.convert_to_tensor(time_data)

    if targets is not None:
        targets = tf.convert_to_tensor(targets)
        time_data = (time_data, targets)

    ds = Dataset.from_tensor_slices(time_data)

    if shuffle:
        ds = ds.shuffle(data_len / 2, seed=seed)

    return ds.batch(batch_size)
