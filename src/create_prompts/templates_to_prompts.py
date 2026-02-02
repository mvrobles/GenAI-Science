
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

def create_normal_prompt(row_statement, prompt):
    """
    Replace [TOPIC] placeholder in prompt with user message from row data.
    
    Args:
        row_statement (dict or pd.Series): Row data containing 'MAIN STATMENT' and 'DETAILED STATMENT'.
        prompt (str): Template prompt with [TOPIC] placeholder.
    
    Returns:
        str: Prompt with [TOPIC] replaced by the combined statement message.
    """
    main_statement = row_statement['MAIN STATMENT']
    detailed_statement = row_statement['DETAILED STATMENT']
    if detailed_statement == '' or detailed_statement is None:
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