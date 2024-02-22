from typing import Any, Callable, Generic, Type, TypeVar, cast, Sequence

from torch import Tensor, cat, device as Device, dtype as DType, empty

Logger = Callable[[Any], None]

CollatableProps = list[Any] | Tensor

InputType = TypeVar('InputType', bound="AbstractBatchInput")

class AbstractBatchInput:
    _list_keys: list[str] = []
    _tensor_keys: dict[str, tuple[int, ...]] = {}
    
    def to(self, device: Device, dtype: DType) -> "AbstractBatchInput":
        
        opts : dict[str, CollatableProps] = {}
        for key in self.__class__._tensor_keys:
            opts[key] = getattr(self, key).to(device=device, dtype=dtype)
              
        return self.__class__(**opts)
        
    def __init__(
        self,
        **kwargs : CollatableProps
    ) -> None:
        for key in self.__class__._list_keys:
            if key not in kwargs:
                raise ValueError(f"Key {key} is not present in {kwargs}")
            setattr(self, key, kwargs[key])
        for key in self.__class__._tensor_keys:
            if key not in kwargs:
                raise ValueError(f"Key {key} is not present in {kwargs}")
            setattr(self, key, kwargs[key])

    @classmethod
    def collate_fn(cls: Type[InputType], batch: Sequence["InputType"]) -> "InputType":
        opts : dict[str, CollatableProps] = {}
        for key in cls._list_keys:
            out_list = opts[key] = []
            
            for item in batch:
                if not hasattr(item, key):
                    raise ValueError(f"Key {key} is not present in {item}")
                for prop in getattr(item, key):
                    out_list.append(prop)
                    
        for key in cls._tensor_keys:
            
            lst : list[Tensor] = []
            for item in batch:
                if not hasattr(item, key):
                    raise ValueError(f"Key {key} is not present in {item}")
                tensor = getattr(item, key)
                if not isinstance(tensor, Tensor):
                    raise ValueError(f"Key {key}, {tensor} should be a tensor")
                lst.append(tensor)
            
            opts[key] = cat(lst)

        return cls(**opts)
    
    @classmethod
    def empty(cls: "Type[InputType]") -> "InputType":
        opts : dict[str, CollatableProps] = {}
        
        for key in cls._list_keys:
            opts[key] = []
        for key in cls._tensor_keys:
            size = cls._tensor_keys[key]
            tensor = empty((0,)+ size)
            opts[key] = tensor
            
        return cls(**opts)

    def get_indices(self, indices: list[int]) -> "AbstractBatchInput":
        opts : dict[str, CollatableProps] = {}
        
        for key in self.__class__._list_keys:
            opts[key] = [getattr(self, key)[i] for i in indices]
        for key in self._tensor_keys:
            opts[key] = getattr(self, key)[indices]
            
        return self.__class__(**opts)
    
    def get_prompt(self, prompt: str) -> "AbstractBatchInput":
        prompts = cast(list[str], getattr(self, "source_prompts"))
        indices = [i for i, p in enumerate(prompts) if p == prompt]
        return self.get_indices(indices)
    
    def __getitem__(self, index: str) -> Tensor | list[Any]:
        if index in self.__class__._list_keys:
            return getattr(self, index)
        if index in self.__class__._tensor_keys:
            return getattr(self, index)
        raise ValueError(f"Index {index} not found in {self.__class__}")

class AbstractBatchOutput(Generic[InputType], AbstractBatchInput):
    __input_type: Type[AbstractBatchInput]
    
    def to_input(self) -> "AbstractBatchInput":
        opts : dict[str, CollatableProps] = {}
        
        for key in self.__input_type._list_keys:
            opts[key] = getattr(self, key)
        for key in self.__input_type._tensor_keys:
            opts[key] = getattr(self, key)
        
        return self.__input_type(**opts)
