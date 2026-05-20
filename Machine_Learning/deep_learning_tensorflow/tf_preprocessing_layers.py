import tensorflow as tf
from copy import deepcopy

class Discretization(tf.keras.layers.Layer):
    def __init__(self, bin_boundaries=None, num_bins=None, **kwargs):
        super().__init__(**kwargs)

        if bin_boundaries is None and num_bins is None:
            raise ValueError('must provide bin_boundaries or num_bins')
        
        self.bin_boundaries = bin_boundaries
        self.num_bins = num_bins

    def adapt(self, data):
        min_val = tf.reduce_min(data)
        max_val = tf.reduce_max(data)
        
        step = (max_val - min_val) / self.num_bins
        self.bin_boundaries = [min_val + tf.cast(i, tf.float32) * step for i in tf.range(1, self.num_bins)] 

    def call(self, input):
        if self.bin_boundaries is None:
            raise ValueError('provide bin_boundaries or adapt layer')
        
        self.bin_boundaries = tf.convert_to_tensor(self.bin_boundaries)
        out = tf.zeros_like(input)

        if len(self.bin_boundaries.shape) == 1:
            last_bin = self.bin_boundaries[0]
            for i, bin in enumerate(self.bin_boundaries[1:], start=1):
                mask = tf.logical_and(last_bin <= input, input < bin)
                out  = tf.where(mask, i, out)
                last_bin = bin

            mask = input >= last_bin
            out  = tf.where(mask, i+1, out)
            return out
        
        for col_i, bins in zip(tf.range(input.shape[-1]), self.bin_boundaries):
            col = tf.identity(input[:, col_i])

            last_bin = bins[0]
            for i, bin in enumerate(bins[1:], start=1):
                mask = tf.logical_and(last_bin <= col, col < bin)
                row_inxs = tf.cast(tf.where(mask), tf.int32)
                inxs = tf.concat([row_inxs, tf.fill(row_inxs.shape, col_i)], axis=1)
                out = tf.tensor_scatter_nd_update(out, inxs, tf.cast(tf.fill([inxs.shape[0]], i), tf.float32))
                last_bin = bin

            row_inxs = tf.cast(tf.where(last_bin <= col), tf.int32)
            inxs = tf.concat([row_inxs, tf.fill(row_inxs.shape, col_i)], axis=1)
            out = tf.tensor_scatter_nd_update(out, inxs, tf.cast(tf.fill([inxs.shape[0]], i + 1), tf.float32))

        return out
    

class CategoryEncoding(tf.keras.layers.Layer):
    def __init__(self, num_tokens, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens

    def call(self, input):
        if input.shape[-1] == 1:
            input = tf.reshape(input, [-1])
    
        out = tf.one_hot(tf.cast(input, tf.int32), self.num_tokens)

        if len(out.shape) == 3:
            out = tf.reduce_max(out, axis=1)

        return out
    
class StringLookup(tf.keras.layers.Layer):
    def __init__(self, vocabulary=None, output_mode='int', num_oov_indices=1, **kwargs):
        super().__init__(**kwargs)
        self.vocabulary = vocabulary
        self.output_mode = output_mode
        self.num_oov_indices = num_oov_indices

    def adapt(self, data):
        self.vocabulary, _, c = tf.unique_with_counts(data)
        inxs = tf.argsort(c, direction='DESCENDING')
        self.vocabulary = tf.gather(self.vocabulary, inxs)

    def vocabulary_size(self):
        if self.vocabulary is None:
            raise ValueError("StringLookup is not adapted")
        
        return len(self.vocabulary) + self.num_oov_indices

    def call(self, input):
        if self.vocabulary is None:
            raise ValueError("StringLookup is not adapted")

        input = tf.convert_to_tensor(input)
        matches = tf.equal(input[..., tf.newaxis], self.vocabulary)
        has_match = tf.reduce_any(matches, axis=-1)

        oov_cats = tf.strings.to_hash_bucket_fast(input, self.num_oov_indices)
        cats = tf.argmax(tf.cast(matches, tf.int32), axis=-1)
        cats = tf.where(has_match, cats + self.num_oov_indices, oov_cats)

        match self.output_mode:
            case 'int': return cats
            case 'one_hot': 
                return CategoryEncoding(len(self.vocabulary) + self.num_oov_indices)(cats)



class Hashing(tf.keras.layers.Layer):
    def __init__(self, num_bins, **kwargs):
        super().__init__(**kwargs)
        self.num_bins = num_bins

    def call(self, input):
        return tf.strings.to_hash_bucket_fast(input, self.num_bins)


class Embedding(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)

        self.embedding_mat = self.add_weight(shape=(input_dim, output_dim),
                                 initializer="random_uniform")

    def call(self, x):
        return tf.gather(self.embedding_mat, x)
    

class TextVectorization(tf.keras.layers.Layer):
    def __init__(self, output_mode='int', split='whitespace', **kwargs):
        super().__init__(**kwargs)
        self.output_mode = output_mode
        self.split = split
    
    def _data_prep(self, data):
        data = tf.convert_to_tensor(data)
        lower = tf.strings.lower(data)
        clean = tf.strings.regex_replace(lower, r"[^\w\s]", "")

        match self.split:
            case 'whitespace': return tf.strings.split(clean)
            case 'character' : return tf.strings.unicode_split(clean, 'UTF-8')
        
        raise ValueError(f'split must be whitespace/character not: {self.split}')

    def adapt(self, data):
        data = self._data_prep(data)
        flatten = tf.reshape(data, [-1])
        self.vocabulary, _, c = tf.unique_with_counts(flatten)
        inxs = tf.argsort(c, direction='DESCENDING')
        self.vocabulary = tf.gather(self.vocabulary, inxs)

        if self.output_mode != 'tf_idf':
            return

        docs = data
        if isinstance(docs, tf.RaggedTensor):
            docs = docs.to_tensor('')  

        d = tf.cast(tf.shape(docs)[0], tf.float32)
        matches = tf.equal(docs[..., tf.newaxis], self.vocabulary)
        appears = tf.reduce_any(matches, axis=1)  
        f = tf.reduce_sum(tf.cast(appears, tf.float32), axis=0)  
        self.idf = tf.math.log(1.0 + d / (f + 1.0))

    def call(self, input):
        tokens = self._data_prep(input)

        if isinstance(tokens, tf.RaggedTensor):
            tokens = tokens.to_tensor('')    

        matches = tf.equal(tokens[..., tf.newaxis],self.vocabulary)                                       
        has_match = tf.reduce_any(matches, axis=-1) 
        is_padding = tf.equal(tokens, '')

        match self.output_mode:
            case 'int':
                matches = tf.cast(matches, tf.int32)
                inxs = tf.argmax(matches, axis=-1, output_type=tf.int32) + 2
                inxs = tf.where(has_match, inxs, tf.ones_like(inxs))
                inxs = tf.where(is_padding, tf.zeros_like(inxs), inxs)

                return inxs
            
            case 'tf_idf':
                matches = tf.cast(matches, tf.float32)

                tf_counts = tf.reduce_sum(matches, axis=1)    
                tfidf_vocab = tf_counts * self.idf               
            
                is_oov = tf.logical_and(~has_match, ~is_padding)
                oov_count = tf.reduce_sum(tf.cast(is_oov, tf.float32), axis=1, keepdims=True)

                oov_idf = tf.reduce_mean(self.idf)
                oov_tfidf = oov_count * oov_idf

                tfidf = tf.concat([oov_tfidf, tfidf_vocab], axis=1)
                return tfidf
            
            case 'one_hot':
                is_oov = tf.logical_and(~has_match, ~is_padding)
                oov = tf.cast(is_oov, tf.float32)[..., tf.newaxis]
                vocab = tf.cast(matches, tf.float32)

                return tf.concat([oov, vocab], axis=-1)
            
            case 'multi_hot':
                is_oov = tf.logical_and(~has_match, ~is_padding)
                oov = tf.reduce_any(is_oov, axis=1, keepdims=True) 
                oov = tf.cast(oov, tf.float32)

                vocab = tf.reduce_any(matches, axis=1)  
                vocab = tf.cast(vocab, tf.float32)

                return tf.concat([oov, vocab], axis=1)
            
            case 'count':
                is_oov = tf.logical_and(~has_match, ~is_padding)
                oov = tf.cast(is_oov, tf.float32)
                oov = tf.reduce_sum(oov, axis=1, keepdims=True) 

                vocab = tf.cast(matches, tf.float32)
                vocab = tf.reduce_sum(vocab, axis=1)  

                return tf.concat([oov, vocab], axis=1)
            
        raise NotImplemented
