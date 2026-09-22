import inspect
from typing import Callable, Any, Dict, Type, List, Optional, Union, get_origin, get_args
from pydantic import BaseModel, Field, create_model

class Tool:
    """
    A generic, provider-agnostic representation of a tool.
    This serves as the canonical definition of a tool within an agent system.
    """
    def __init__(self, name: str, description: str, func: Callable, schema: Optional[Type[BaseModel]] = None):
        self.name = name
        self.description = description
        self.func = func
        self.schema = schema if schema else self._infer_schema_from_func()

    def _infer_schema_from_func(self) -> Type[BaseModel]:
        """Infers a Pydantic schema from the function's signature."""
        sig = inspect.signature(self.func)
        fields = {}
        for name, param in sig.parameters.items():
            if name == 'self': # Skip 'self' for methods
                continue
            if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD or \
               param.kind == inspect.Parameter.KEYWORD_ONLY:
                field_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
                field_default = param.default if param.default != inspect.Parameter.empty else Field(...)
                fields[name] = (field_type, field_default)
        return create_model(f"{self.name.capitalize()}Input", **fields)

    def __call__(self, **kwargs: Any) -> Any:
        """Executes the tool function with validated arguments."""
        if self.schema:
            validated_args = self.schema(**kwargs).model_dump()
            return self.func(**validated_args)
        return self.func(**kwargs)

class ToolSchemaAdapter:
    """
    Adapts a generic Tool object into a provider-specific format.
    This class handles the conversion of tool schemas to match the API requirements
    of different LLM providers (e.g., Gemini's FunctionDeclaration, OpenAI's tool format).
    """
    def to_gemini_function_declaration(self, tool: Tool) -> Dict[str, Any]:
        """Converts a Tool into Gemini's FunctionDeclaration format."""
        if not tool.schema:
            raise ValueError(f"Tool '{tool.name}' must have a Pydantic schema for Gemini conversion.")

        properties = {}
        required_params = []

        for field_name, field in tool.schema.model_fields.items():
            json_schema_type = self._pydantic_type_to_json_schema_type(field.annotation)
            field_info = {
                "type": json_schema_type,
                "description": field.description or ""
            }
            if json_schema_type == "array":
                # For arrays, we need to specify the items type
                inner_type = get_args(field.annotation)[0] if get_args(field.annotation) else Any
                field_info["items"] = {"type": self._pydantic_type_to_json_schema_type(inner_type)}

            if field.is_required():
                required_params.append(field_name)
            
            properties[field_name] = field_info

        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required_params
            }
        }

    def _pydantic_type_to_json_schema_type(self, pydantic_type: Any) -> str:
        """Maps Pydantic types to JSON schema types, handling Optional and generic types."""
        # Handle Optional types by unwrapping them
        if get_origin(pydantic_type) is Union and type(None) in get_args(pydantic_type):
            # It's an Optional type, get the actual type
            actual_type = [arg for arg in get_args(pydantic_type) if arg is not type(None)][0]
            return self._pydantic_type_to_json_schema_type(actual_type)

        if pydantic_type in (str,):
            return "string"
        elif pydantic_type in (int,):
            return "integer"
        elif pydantic_type in (float,):
            return "number"
        elif pydantic_type in (bool,):
            return "boolean"
        elif get_origin(pydantic_type) is list or pydantic_type is list:
            return "array"
        elif get_origin(pydantic_type) is dict or pydantic_type is dict:
            return "object"
        # If it's a Pydantic BaseModel, treat it as an object
        elif inspect.isclass(pydantic_type) and issubclass(pydantic_type, BaseModel):
            return "object"
        
        return "string" # Default to string for unknown or complex types for simplicity
