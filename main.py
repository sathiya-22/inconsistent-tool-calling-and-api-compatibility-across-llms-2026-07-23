from typing import List, Optional
from pydantic import BaseModel, Field
from tool_adapter import Tool, ToolSchemaAdapter

# --- Define some sample functions and their schemas ---

def get_current_weather(location: str, unit: Optional[str] = "celsius") -> str:
    """
    Get the current weather in a given location.
    Args:
        location: The city and state, e.g. San Francisco, CA
        unit: The unit of temperature, either 'celsius' or 'fahrenheit'. Defaults to 'celsius'.
    """
    # In a real scenario, this would call an external API
    print(f"--- Calling get_current_weather for {location} in {unit} ---")
    if location.lower() == "london, uk":
        return f"It's 15 degrees {unit} and cloudy in London."
    return f"It's 22 degrees {unit} and sunny in {location}."

class SearchFlightInput(BaseModel):
    departure: str = Field(..., description="Departure airport code (e.g., SFO)")
    destination: str = Field(..., description="Destination airport code (e.g., LAX)")
    date: str = Field(..., description="Departure date in YYYY-MM-DD format")
    passengers: int = Field(1, description="Number of passengers")

def search_flights(departure: str, destination: str, date: str, passengers: int = 1) -> str:
    """
    Searches for available flights.
    Args:
        departure: The departure airport code.
        destination: The destination airport code.
        date: The departure date.
        passengers: The number of passengers.
    """
    print(f"--- Calling search_flights from {departure} to {destination} on {date} for {passengers} passengers ---")
    return f"Found flights from {departure} to {destination} on {date} for {passengers} passengers."

# --- Create generic Tool instances ---

# Tool with inferred schema
weather_tool = Tool(
    name="get_current_weather",
    description="Get the current weather in a given location.",
    func=get_current_weather
)

# Tool with explicit Pydantic schema
flight_tool = Tool(
    name="search_flights",
    description="Searches for available flights.",
    func=search_flights,
    schema=SearchFlightInput
)

# --- Use the ToolSchemaAdapter to convert to Gemini's format ---
adapter = ToolSchemaAdapter()

print("--- Converting 'get_current_weather' to Gemini FunctionDeclaration ---")
gemini_weather_schema = adapter.to_gemini_function_declaration(weather_tool)
import json
print(json.dumps(gemini_weather_schema, indent=2))
print("\n" + "="*80 + "\n")

print("--- Converting 'search_flights' to Gemini FunctionDeclaration ---")
gemini_flight_schema = adapter.to_gemini_function_declaration(flight_tool)
print(json.dumps(gemini_flight_schema, indent=2))
print("\n" + "="*80 + "\n")

# --- Simulate LLM output and tool calling ---

print("--- Simulating LLM calling 'get_current_weather' ---")
# An LLM would output something like this based on the generated schema
llm_call_args_weather = {"location": "London, UK", "unit": "fahrenheit"}
try:
    weather_result = weather_tool(**llm_call_args_weather)
    print(f"Weather Tool Result: {weather_result}")
except Exception as e:
    print(f"Error calling weather tool: {e}")

print("\n--- Simulating LLM calling 'search_flights' ---")
llm_call_args_flight = {
    "departure": "JFK",
    "destination": "LAX",
    "date": "2023-12-25",
    "passengers": 2
}
try:
    flight_result = flight_tool(**llm_call_args_flight)
    print(f"Flight Tool Result: {flight_result}")
except Exception as e:
    print(f"Error calling flight tool: {e}")

print("\n--- Simulating LLM calling 'search_flights' with missing required arg (will fail Pydantic validation) ---")
llm_call_args_flight_invalid = {
    "departure": "JFK",
    "destination": "LAX",
    # "date" is missing
    "passengers": 1
}
try:
    flight_result_invalid = flight_tool(**llm_call_args_flight_invalid)
    print(f"Flight Tool Result (invalid): {flight_result_invalid}")
except Exception as e:
    print(f"Error calling flight tool with invalid args: {e}")

print("\n--- End of demonstration ---")
