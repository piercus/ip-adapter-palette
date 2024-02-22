from typing import Any, Callable, Generic, Type, TypeVar, cast, Sequence, get_type_hints

from torch import Tensor, cat, device as Device, dtype as DType, empty, stack

Logger = Callable[[Any], None]

CollatableProps = list[Any] | Tensor


T = TypeVar('T', bound='Batch')

class Batch:
    def __init__(self, **kwargs):
        for attr_name, attr_value in kwargs.items():
            setattr(self, attr_name, attr_value)

    @classmethod
    def collate(cls: Type[T], batch_list: list[T]) -> T:
        collated_attrs = {}

        # Retrieve type hints from the class
        type_hints = get_type_hints(cls)

        for key in type_hints.keys():
            attr_list = [getattr(obj, key) for obj in batch_list]

            if issubclass(type_hints[key], Tensor):
                # Use torch.stack to collate tensors
                collated_attrs[key] = stack(attr_list, dim=0)
            elif issubclass(type_hints[key], list):
                # Concatenate lists
                collated_attrs[key] = [item for sublist in attr_list for item in sublist]
            else:
                raise ValueError(f"Unsupported attribute type for collation: {type_hints[key]}")

        # Dynamically create a new instance of the subclass with the collated attributes
        collated_instance = cls(**collated_attrs)
        return collated_instance
    
    @classmethod
    def keys(cls) -> list[str]:
        return list(get_type_hints(cls).keys())