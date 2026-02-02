from abc import ABC, abstractmethod
from tqdm import tqdm
import pandas as pd

class LLMRunner(ABC):
    def __init__(self, temperature, save_every, model_id):
        """
        Initialize the LLMRunner with model configuration parameters.
        
        Args:
            temperature (float): Temperature parameter for model sampling (controls randomness).
            save_every (int): Number of iterations between checkpoint saves to CSV.
            model_id (str): Identifier for the LLM model being used.
        
        Returns:
            None
        """
        self.temperature = temperature
        self.save_every = save_every
        self.model_id = model_id

    def read_excel(self, path, sheet_name):
        """
        Read data from an Excel file.
        
        Args:
            path (str): File path to the Excel file.
            sheet_name (str): Name of the sheet to read.
        
        Returns:
            pd.DataFrame: DataFrame containing the data from the specified sheet.
        """
        return pd.read_excel(path, sheet_name=sheet_name, engine='openpyxl')

    def create_user_message(self, main_statement, detailed_statement = None):
        """
        Combine main and detailed statements into a single user message.
        
        Args:
            main_statement (str): The main statement/topic.
            detailed_statement (str, optional): Additional detailed statement. Defaults to None.
        
        Returns:
            str: Combined message (main + detailed), or just main if detailed is None.
        """
        if detailed_statement is not None:
            return main_statement + ' ' + detailed_statement
        return main_statement

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
        df['tokens'] = None

        for i, row in tqdm(df.iterrows(), total=len(df), disable=False):
            try:
                model_answer = self.run_one_prompt(client, row)
                df.at[i, 'result'] = model_answer
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
    def run_one_prompt(self, row):
        """
        Execute a single LLM prompt on a DataFrame row.
        
        Must be implemented by subclasses.
        
        Args:
            row (pd.Series): A row from the DataFrame containing prompt data.
        
        Returns:
            str or dict: The model's response to the prompt.
        """
        pass
