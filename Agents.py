import autogen
import yaml



def load_yaml(file_path) -> dict:
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config



def load_ZhiPu(config: dict) -> list:
    zhipu_config = config["client"]["zhipu"]
    config_list_gpt = [{**zhipu_config}]
    return config_list_gpt

def load_Gemini(config: dict) -> list:
    zhipu_config = config["client"]["gemini"]
    config_list_gpt = [{**zhipu_config}]
    return config_list_gpt


def Create_Agents(config_list_gpt: list, name: str, prompt: str, seed=None):
    pass

if __name__ == "__main__":
    config = load_yaml("config.yaml")
    config_list_gpt = load_ZhiPu(config)
    config_list_gemini = load_Gemini(config)