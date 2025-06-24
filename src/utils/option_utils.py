import logging
from collections.abc import Callable, Iterable
from dataclasses import MISSING, asdict, fields, _MISSING_TYPE
from pathlib import Path
from typing import Any, Literal


def _construct_literal(value: Any, choices: list) -> Any:
    # Handle Literal types by checking if the value is in the allowed options
    if isinstance(value, str) and value in choices: # type: ignore
        return value
    else:
        raise ValueError(f"Value '{value}' is not a valid option for {choices}")
    
def _construct_list(value: Any, elm_constructor: Callable) -> list:
    # Handle list types by converting the value to a list if it is not already
    if isinstance(value, Iterable) and not isinstance(value, str):
        return list(value)
    elif isinstance(value, str):
        return list(map(elm_constructor, value.split(',')))  # Split by comma if it's a string
    else:
        raise ValueError(f"Value '{value}' cannot be converted to a list")
    
    
def get_constructor_from_type_hint(type_hint: Any) -> Callable:
    """
    Get the constructor and default value for a given type hint.
    """
    if type_hint is bool:
        return bool
    elif type_hint is int:
        return int
    elif type_hint is float:
        return float
    elif type_hint is str:
        return str
    elif type_hint is Path:
        return Path
    elif hasattr(type_hint, '__origin__') and type_hint.__origin__ is Literal:
        # Handle Literal types
        return lambda v:_construct_literal(v, type_hint.__args__)
    elif hasattr(type_hint, '__origin__') and type_hint.__origin__ is list:
        return lambda v: _construct_list(v, get_constructor_from_type_hint(type_hint.__args__[0]))
    elif hasattr(type_hint, '__origin__') and type_hint.__origin__ is dict:
        return dict
    elif hasattr(type_hint, '__origin__') and type_hint.__origin__ is tuple:
        return tuple
    else:
        raise ValueError(f"Unsupported type hint: {type_hint}")


# def make_dataclass_from_args_or_instance(cls, instance_or_args: Any = None, **args):
#     """
#     Convert a dataclass or a dictionary of arguments into a dataclass instance.
#     If `arg_or_instance` is an instance of `cls`, it returns it directly.
#     Otherwise, it treats `arg_or_instance` as a dictionary of arguments.
#     """
#     if isinstance(instance_or_args, cls):
#         return instance_or_args
#     elif isinstance(instance_or_args, dict):
#         return make_dataclass_from_args(cls, instance_or_args)
#     return make_dataclass_from_args(cls, args)


def make_dataclass_from_args(cls, args):
    return make_dataclass_from_cp_args(cls, None, args)


def make_dataclass_from_cp_args(cls, cp, args):
    """
    Convert a checkpoint and arguments into a dataclass instance.
    """
    init_args = {}
    error_fields = set()

    cpvals = {} if cp is None else cp if isinstance(cp, dict) else asdict(cp)
    
    logging.info(f"  {cls.__name__}")
    # Convert types if necessary
    for field in fields(cls):

        field_name = field.name
        constructor = get_constructor_from_type_hint(field.type)
        default_value = MISSING if isinstance(field.default, _MISSING_TYPE) else field.default

        cp_value = cpvals.get(field_name)
        arg_value = args.get(field_name)
        if arg_value is not None:
            init_args[field_name] = constructor(arg_value)
            logging.info(f"    {field_name}: {init_args[field_name]} (from args)")
        elif cp_value is not None:
            init_args[field_name] = constructor(cp_value)
            logging.info(f"    {field_name}: {init_args[field_name]} (from cp)")
        elif default_value is not MISSING:
            init_args[field_name] = default_value
            logging.info(f"    {field_name}: {init_args[field_name]} (from defaults)")
        else:
            error_fields.add(field_name)

    if error_fields:
        raise ValueError(f"Missing required fields in {cls.__name__}: {', '.join(error_fields)}")
    
    return cls(**init_args)
