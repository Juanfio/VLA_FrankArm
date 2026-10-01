import json

def get_settings(settings: dict):
    with open(f"{settings}", "r") as file:
        return json.load(file)