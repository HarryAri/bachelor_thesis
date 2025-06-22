import json

def json_to_txt(input_path, output_path):
    '''
    :param input_path: The JSON file directory
    :param output_path: Where the .txt file will be stored
    :return: .txt file
    '''
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)


    def get_line(entry):
        name = entry.get("name")
        desc = entry.get("description")
        return f"{name}: {desc}".strip()

    lines = [get_line(item) for item in data]


    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

json_to_txt("data/local.json", "data/pt_local.txt")
json_to_txt("data/global.json", "data/pt_global.txt")