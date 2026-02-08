from abc import ABC, abstractmethod
from tqdm import tqdm
import pandas as pd

class LLMRunner(ABC):
    def __init__(self, save_every, model_id):
        """
        Initialize the LLMRunner with model configuration parameters.
        
        Args:
            save_every (int): Number of iterations between checkpoint saves to CSV.
            model_id (str): Identifier for the LLM model being used.
        
        Returns:
            None
        """
        self.save_every = save_every
        self.model_id = model_id

    def read_csv(self, path):
        """
        Read data from a csv file.
        
        Args:
            path (str): File path.
        
        Returns:
            pd.DataFrame: DataFrame containing the data from the specified sheet.
        """
        return pd.read_csv(path)
    
    def run_llm(self, client, df, output_path):
        """
        Execute LLM prompts on each row of the DataFrame and save results.
        
        Args:
            client: API client for the LLM service.
            df (pd.DataFrame): DataFrame containing rows with prompt data.
            output_path (str): File path where results are saved as CSV (checkpoints every save_every iterations).
        
        Returns:
            None. Modifies df in place by adding 'result' and 'tokens' columns, and saves to output_path.
        """
        df['result'] = None
        df['references'] = None
        df['tokens'] = None

        for i, row in tqdm(df.iterrows(), total=len(df), disable=False):
            try:
                model_answer, references, _ = self.run_one_prompt(client, row.prompt)
                df.at[i, 'result'] = model_answer
                df.at[i, 'references'] = references

            except Exception as e:
                print(f"Error on line {i}: {e}")

            if (i + 1) % self.save_every == 0:
                df.to_csv(output_path, index=False)

        df.to_csv(output_path, index=False)

    @abstractmethod
    def connect(self):
        """
        Establish connection to the LLM API service.
        
        Must be implemented by subclasses.
        
        Args:
            None
        
        Returns:
            Client object for the specific LLM service.
        """
        pass

    @abstractmethod
    def run_one_prompt(self, prompt: str):
        """
        Execute a single LLM prompt on a DataFrame row.
        
        Must be implemented by subclasses.
        
        Args:
            prompt (str): Prompt to be excecuted
        
        Returns:
            str or dict: The model's response to the prompt.
        """
        pass
