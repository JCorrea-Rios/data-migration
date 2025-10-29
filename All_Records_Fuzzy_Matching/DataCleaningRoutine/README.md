Based on the requirements, we will need the following core classes, functions, and methods:

1. `sanitize.py`: This will be the main script that will read the company names from the input file, sanitize them according to the rules, and write the sanitized names to the output file.

2. `sanitize_company_name(name: str, rules: List[Dict[str, str]]) -> str`: This function will take a company name and a list of sanitization rules, apply the rules to the name, and return the sanitized name.

3. `load_sanitization_rules(file_path: str) -> List[Dict[str, str]]`: This function will load the sanitization rules from a JSON file and return them as a list of dictionaries.

4. `write_sanitized_names(names: List[str], file_path: str) -> None`: This function will write a list of sanitized company names to a file.

5. `sanitize_all(names: List[str], rules: List[Dict[str, str]]) -> List[str]`: This function will sanitize all company names in a list using a list of sanitization rules.

6. `remove_duplicates(names: List[str]) -> List[str]`: This function will remove duplicate company names from a list.

7. `proper_case(name: str) -> str`: This function will convert a company name to proper case.

Now, let's write the code for each of these components.

sanitize.py
