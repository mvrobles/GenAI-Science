
import pandas as pd 
import sys

def create_user_message(main_statement, detailed_statement = None):
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

def create_normal_prompt(row_statement, prompt):
    """
    Replace [TOPIC] placeholder in prompt with user message from row data.
    
    Args:
        row_statement (dict or pd.Series): Row data containing 'MAIN STATMENT' and 'DETAILED STATMENT'.
        prompt (str): Template prompt with [TOPIC] placeholder.
    
    Returns:
        str: Prompt with [TOPIC] replaced by the combined statement message.
    """
    main_statement = row_statement['MAIN STATEMENT']
    detailed_statement = row_statement['DETAILED STATEMENT']
    if detailed_statement == '' or detailed_statement is None or pd.isna(detailed_statement):
        detailed_statement = None

    message = create_user_message(main_statement, detailed_statement)
    
    return prompt.replace("[TOPIC]", message)

def create_link_prompt(row_statement, prompt):
    """
    Replace [TOPIC] and [LINK] placeholders in prompt with statement and reference data.
    
    Args:
        row_statement (dict or pd.Series): Row data containing 'MAIN STATMENT', 'DETAILED STATMENT', and 'REFERENCE'.
        prompt (str): Template prompt with [TOPIC] and [LINK] placeholders.
    
    Returns:
        str: Prompt with both [TOPIC] and [LINK] placeholders replaced.
    """
    prompt = create_normal_prompt(row_statement, prompt)
    link = row_statement['REFERENCE']

    return prompt.replace('[LINK]', link)

def create_term_prompt(row_term, prompt):
    """
    Replace [TERM] placeholder in prompt with term data from row.
    
    Args:
        row_term (dict or pd.Series): Row data containing 'TERM'.
        prompt (str): Template prompt with [TERM] placeholder.
    
    Returns:
        str: Prompt with [TERM] replaced by the term value.
    """
    term = row_term['TERM']
    return prompt.replace('[TERM]', term)

def clean_cols(df, col):
    df[col] = df[col].apply(
        lambda x: None if isinstance(x, str) and x.strip() == "" else x
        )
    df[col] = df[col].apply(
        lambda x: x.replace('\n', ' ') if isinstance(x, str) else x
        )


def main(excel_path, save_path):
    templates = pd.read_excel(excel_path, sheet_name = 'Templates')
    statements = pd.read_excel(excel_path, sheet_name='Statements')
    terms = pd.read_excel(excel_path, sheet_name='Terms')

    clean_cols(statements, 'MAIN STATEMENT')
    clean_cols(statements, 'DETAILED STATEMENT')

    normal_templates = templates.iloc[:-2]
    link_template = templates.iloc[-2]
    term_template = templates.iloc[-1]

    prompts = []

    print('Creating prompts...')
    for _, row_template in normal_templates.iterrows():
        for _, row_statement in statements.iterrows():
            prompts.append(create_normal_prompt(row_statement, row_template['prompt']))
    
    for _, row_statement in statements.iterrows():
        if not pd.isna(row_statement.REFERENCE):    
            prompts.append(create_link_prompt(row_statement, link_template['prompt']))

    for _, row_term in terms.iterrows():
        prompts.append(create_term_prompt(row_term, term_template['prompt']))

    df = pd.DataFrame(prompts, columns = ['prompt'])
    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python templates_to_prompts.py <excel_path> <save_path>")
        sys.exit(1)

    excel_path = sys.argv[1]
    save_path = sys.argv[2]
    main(excel_path, save_path)
    print(f'Prompts saved to {save_path}')

    

