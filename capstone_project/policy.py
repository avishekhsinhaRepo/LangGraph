# import os
# import json


# def read_json_file(file_path: str) -> dict:
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     file_path = os.path.join(current_dir, "data", file_path)
#     with open(file_path, "r") as file:
#         return json.load(file)


# policy_id = "POL1012"
# insurance_policies_data = read_json_file("insurance_policies.json")
# policy_by_id = {p["policy_id"]: p for p in insurance_policies_data}
# print(policy_by_id)


file_path = "insurance_policies.json"
full_path = f"./Data/{file_path}"
print(full_path)
