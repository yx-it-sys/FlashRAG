import json
import re
from json_repair import repair_json

def extract_json(response: str) -> dict | None:
    pattern = r"JSON:\s*(\{[\s\S]+\})"
    match = re.search(pattern, response)
    if match:
        json_string = match.group(1)
        try:
            parsed_json = json.loads(json_string)
        except json.JSONDecodeError as e:
            print("⚠️ LLM output is not valid JSON. Attempting to repair...")
            print(json_string)        
            try:
                repaired_json_string = repair_json(response)
                parsed_json = json.loads(repaired_json_string)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"❌ Failed to parse JSON even after repair. Error: {e}")
                return None
        return parsed_json
    else:
        print(f"❌ Failed to parse JSON: {response}")
        return None
                    
def extract_json_for_assessment(llm_output: str):
    start_index = llm_output.find('{')
    end_index = llm_output.rfind('}')

    if start_index == -1 or end_index == -1 or end_index < start_index:
        print("Error: Could not find a JSON block enclosed in {} braces.")
        return None

    json_str_candidate = llm_output[start_index : end_index + 1]
    print(f"Found potential JSON block:\n{json_str_candidate}")

    try:
        parsed_json = json.loads(json_str_candidate)
        print("Successfully parsed with standard `json` library.")
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"Standard `json.loads` failed: {e}")
        print("Attempting to repair with `json_repair` library...")

        try:
            repaired_json_string = repair_json(json_str_candidate)
            parsed_json = json.loads(repaired_json_string)
            print("Successfully repaired and parsed JSON.")
            return parsed_json
        except Exception as e:
            print(f"Error: JSON repair also failed. The string might be too malformed. Details: {e}")
            return None

def parse_response(response: str):
    if "JSON:" not in response:
        return []
    list_part = response.split("JSON:")[1].strip()
    lines = list_part.splitlines()
    parsed_items = [re.sub(r"^\d+\.\s*", "", line).strip() for line in lines if line.strip()]

    return parsed_items

