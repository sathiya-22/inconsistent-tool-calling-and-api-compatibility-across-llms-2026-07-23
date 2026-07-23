import inspect
from typing import Callable, Any, Dict, Type, List, Optional, Union
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
            field_info = {
                "type": self._pydantic_type_to_json_schema_type(field.annotation),
                "description": field.description or ""
            }
            if field.is_required():
                required_params.append(field_name)
            if field.default is not None and not field.is_required():
                # Gemini doesn't explicitly support 'default' in FunctionDeclaration,
                # but we can include it in description or omit if not strictly needed.
                # For this prototype, we'll just ensure the type is correct.
                pass # Default values are handled by Pydantic validation, not LLM schema.

            properties[field_name] = field_info

        # Handle enums if necessary (Pydantic Enum -> JSON schema enum)
        # Handle nested models if necessary (Pydantic Model -> JSON schema object)
        # This prototype keeps it simple for direct types.

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
        """Maps Pydantic types to JSON schema types."""
        if pydantic_type in (str, Optional[str]):
            return "string"
        elif pydantic_type in (int, Optional[int]):
            return "integer"
        elif pydantic_type in (float, Optional[float]):
            return "number"
        elif pydantic_type in (bool, Optional[bool]):
            return "boolean"
        elif pydantic_type in (list, List[Any], Optional[List[Any]]):
            return "array"
        elif pydantic_type in (dict, Dict[Any, Any], Optional[Dict[Any, Any]]):
            return "object"
        # Handle more complex types or Pydantic models recursively if needed
        return "string" # Default to string for unknown or complex types for simplicity
