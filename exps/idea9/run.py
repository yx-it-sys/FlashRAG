from flashrag.prompt import MMPromptTemplate
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline.mm_pipeline import MMCluePipeline
import tomllib

def main():
    config = Config("my_config.yaml")
    all_split = get_dataset(config)
    test_data = all_split["validation"]

    with open(config['visual_clue_prompt_file'], 'rb') as f:
        prompt_dict = tomllib.load(f)
        sys_prompt = prompt_dict['system_prompt']
        usr_prompt = prompt_dict['user_prompt']
    
    visual_clue_prompt_template = MMPromptTemplate(config, system_prompt=sys_prompt, user_prompt=usr_prompt)
    pipeline = MMCluePipeline(config=config, visual_clue_prompt_template=visual_clue_prompt_template)
    clue_list = pipeline.get_clue(test_data)

if __name__ == "__main__":
    main()