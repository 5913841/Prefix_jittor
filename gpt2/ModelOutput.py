


from collections import OrderedDict
from dataclasses import dataclass, fields
import numpy as np
from typing import Any, List, Optional, Tuple
import jittor as jt

def is_jittor_available():
    return True


def is_tensor(x):
    """
    Tests if `x` is a `torch.Tensor`, `tf.Tensor`, `jaxlib.xla_extension.DeviceArray` or `np.ndarray`.
    """
    if is_jittor_available():
        import jittor as jt

        if isinstance(x, jt.Var):
            return True
        
    return isinstance(x, np.ndarray)


# class ModelOutput(OrderedDict):
#     """
#     Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
#     tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
#     python dictionary.

#     <Tip warning={true}>

#     You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
#     before.

#     </Tip>
#     """

#     def __post_init__(self):
#         class_fields = fields(self)

#         # Safety and consistency checks
#         if not len(class_fields):
#             raise ValueError(f"{self.__class__.__name__} has no fields.")
#         if not all(field.default is None for field in class_fields[1:]):
#             raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

#         first_field = getattr(self, class_fields[0].name)
#         other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

#         if other_fields_are_none and not is_tensor(first_field):
#             if isinstance(first_field, dict):
#                 iterator = first_field.items()
#                 first_field_iterator = True
#             else:
#                 try:
#                     iterator = iter(first_field)
#                     first_field_iterator = True
#                 except TypeError:
#                     first_field_iterator = False

#             # if we provided an iterator as first field and the iterator is a (key, value) iterator
#             # set the associated fields
#             if first_field_iterator:
#                 for idx, element in enumerate(iterator):
#                     if (
#                         not isinstance(element, (list, tuple))
#                         or not len(element) == 2
#                         or not isinstance(element[0], str)
#                     ):
#                         if idx == 0:
#                             # If we do not have an iterator of key/values, set it as attribute
#                             self[class_fields[0].name] = first_field
#                         else:
#                             # If we have a mixed iterator, raise an error
#                             raise ValueError(
#                                 f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
#                             )
#                         break
#                     setattr(self, element[0], element[1])
#                     if element[1] is not None:
#                         self[element[0]] = element[1]
#             elif first_field is not None:
#                 self[class_fields[0].name] = first_field
#         else:
#             for field in class_fields:
#                 v = getattr(self, field.name)
#                 if v is not None:
#                     self[field.name] = v

#     def __delitem__(self, *args, **kwargs):
#         raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

#     def setdefault(self, *args, **kwargs):
#         raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

#     def pop(self, *args, **kwargs):
#         raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

#     def update(self, *args, **kwargs):
#         raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

#     def __getitem__(self, k):
#         if isinstance(k, str):
#             inner_dict = {k: v for (k, v) in self.items()}
#             return inner_dict[k]
#         else:
#             return self.to_tuple()[k]

#     def __setattr__(self, name, value):
#         if name in self.keys() and value is not None:
#             # Don't call self.__setitem__ to avoid recursion errors
#             super().__setitem__(name, value)
#         super().__setattr__(name, value)

#     def __setitem__(self, key, value):
#         # Will raise a KeyException if needed
#         super().__setitem__(key, value)
#         # Don't call self.__setattr__ to avoid recursion errors
#         super().__setattr__(key, value)

#     def to_tuple(self) -> Tuple[Any]:
#         """
#         Convert self to a tuple containing all the attributes/keys that are not `None`.
#         """
#         return tuple(self[k] for k in self.keys())


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a ``__getitem__`` that allows indexing by integer or slice (like
    a tuple) or strings (like a dictionnary) that will ignore the ``None`` attributes. Otherwise behaves like a
    regular python dictionary.

    .. warning::
        You can't unpack a :obj:`ModelOutput` directly. Use the :meth:`~transformers.file_utils.ModelOutput.to_tuple`
        method to convert it to a tuple before.
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        assert len(class_fields), f"{self.__class__.__name__} has no fields."
        assert all(
            field.default is None for field in class_fields[1:]
        ), f"{self.__class__.__name__} should not have more than one required field."

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            try:
                iterator = iter(first_field)
                first_field_iterator = True
            except TypeError:
                first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`jt.Var` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(jt.Var))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(jt.Var)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(jt.Var)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jt.Var` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(jt.Var)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jt.Var` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(jt.Var)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `jt.Var` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    last_hidden_state: jt.Var = None
    past_key_values: Optional[Tuple[Tuple[jt.Var]]] = None
    hidden_states: Optional[Tuple[jt.Var]] = None
    attentions: Optional[Tuple[jt.Var]] = None
    cross_attentions: Optional[Tuple[jt.Var]] = None



class CausalLMOutputWithCrossAttentions(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`jt.Var` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`jt.Var` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(jt.Var)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jt.Var` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(jt.Var)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jt.Var` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(jt.Var)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jt.Var` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross attentions weights after the attention softmax, used to compute the weighted average in the
            cross-attention heads.
        past_key_values (`tuple(tuple(jt.Var))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `jt.Var` tuples of length `config.n_layers`, with each tuple containing the cached key,
            value states of the self-attention and the cross-attention layers if model is used in encoder-decoder
            setting. Only relevant if `config.is_decoder = True`.

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[jt.Var] = None
    logits: jt.Var = None
    past_key_values: Optional[Tuple[Tuple[jt.Var]]] = None
    hidden_states: Optional[Tuple[jt.Var]] = None
    attentions: Optional[Tuple[jt.Var]] = None
    cross_attentions: Optional[Tuple[jt.Var]] = None

@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (:obj:`jt.Var` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (:obj:`jt.Var` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`List[jt.Var]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`jt.Var` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            ``past_key_values`` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(jt.Var)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`jt.Var` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(jt.Var)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`jt.Var` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[jt.Var] = None
    logits: jt.Var = None
    past_key_values: Optional[List[jt.Var]] = None
    hidden_states: Optional[Tuple[jt.Var]] = None
    attentions: Optional[Tuple[jt.Var]] = None


class BaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (:obj:`jt.Var` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If :obj:`past_key_values` is used only the last hidden-state of the sequences of shape
            :obj:`(batch_size, 1, hidden_size)` is output.
        past_key_values (:obj:`List[jt.Var]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`jt.Var` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            ``past_key_values`` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(jt.Var)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`jt.Var` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(jt.Var)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`jt.Var` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: jt.Var
    past_key_values: Optional[List[jt.Var]] = None
    hidden_states: Optional[Tuple[jt.Var]] = None
    attentions: Optional[Tuple[jt.Var]] = None
