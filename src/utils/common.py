#!/usr/bin/env python3
"""
Common utility functions shared across the project.
"""

import os
import sys
import hashlib
import json
import re
from typing import Dict, Any, Optional

def generate_config_hash(config_dict: Dict[str, Any]) -> str:
    """Generate a hash of the configuration for tracking changes."""
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

def get_project_root() -> str:
    """Get the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def ensure_dir_exists(directory: str) -> None:
    """Ensure that the specified directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)


# ============================================================================
# TEXT FORMATTING UTILITIES
# ============================================================================

# Legal suffix mappings for proper title casing
LEGAL_SUFFIX_MAP = {
    # Australian/New Zealand
    r'\bPTY\s+LTD\b': 'Pty Ltd',
    r'\bPTY\s+LIMITED\b': 'Pty Limited',
    r'\bPTE\s+LTD\b': 'Pte Ltd',
    
    # General English
    r'\bLTD\b': 'Ltd',
    r'\bLIMITED\b': 'Limited',
    r'\bINC\b': 'Inc',
    r'\bCORP\b': 'Corp',
    r'\bCORPORATION\b': 'Corporation',
    r'\bCO\b': 'Co',
    
    # European
    r'\bGMBH\b': 'GmbH',
    r'\bAG\b': 'AG',  # Aktiengesellschaft (keep uppercase)
    r'\bSA\b': 'SA',  # Société Anonyme (keep uppercase)
    r'\bNV\b': 'N.V.',  # Naamloze Vennootschap
    r'\bBV\b': 'B.V.',  # Besloten Vennootschap
    r'\bAB\b': 'AB',  # Aktiebolag (Swedish)
    
    # Asian
    r'\bSDN\s+BHD\b': 'Sdn Bhd',  # Malaysian
    r'\bPVT\b': 'Pvt',  # Indian Private Limited
    r'\bPVT\s+LTD\b': 'Pvt Ltd',
    
    # US specific
    r'\bLLC\b': 'LLC',  # Keep uppercase (common convention)
    r'\bLLP\b': 'LLP',  # Keep uppercase
    r'\bPLC\b': 'PLC',  # Keep uppercase (UK)
    r'\bL\.P\.\b': 'L.P.',  # Limited Partnership
    r'\bL\.L\.C\.\b': 'LLC',  # LLC with periods -> standardize
    
    # Other
    r'\bS\.A\.R\.L\.\b': 'S.à r.l.',  # French
    r'\bS\.A\.R\.L\b': 'S.à r.l.',
}

# Common acronyms that should remain uppercase
PRESERVE_ACRONYMS = [
    'IBM', 'HP', 'AT&T', 'IT', 'AI', 'UK', 'US', 'NSW', 'QLD', 'VIC',
    'ACT', 'NT', 'SA', 'WA', 'TAS', 'ABN', 'ACN', 'NZ', 'USA', 'EU',
    'UAE', 'CEO', 'CFO', 'CTO', 'HR', 'IT', 'PR', 'RD', 'IP', 'TV',
    'BMW', 'VW', 'GM', 'GE', 'BP', 'CVS', 'UPS', 'FedEx', 'DHL'
]


def title_case_legal_suffix(name: Optional[str]) -> Optional[str]:
    """
    Convert legal suffixes to proper title case.
    
    Examples:
        PTY LTD → Pty Ltd
        INC → Inc
        LLC → LLC (stays uppercase)
        GMBH → GmbH
    
    Args:
        name: The company name to process
        
    Returns:
        The name with properly title-cased legal suffixes, or None if input is None
    """
    if not name:
        return name
    
    result = name
    for pattern, replacement in LEGAL_SUFFIX_MAP.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result


def smart_title_case_suffix(name: Optional[str], preserve_acronyms: bool = True) -> Optional[str]:
    """
    Intelligently title case legal suffixes while preserving acronyms and special cases.
    
    This function applies proper title casing to legal entity suffixes (e.g., Pty Ltd, Inc)
    while optionally preserving common acronyms in their original uppercase form.
    
    Examples:
        IBM AUSTRALIA PTY LTD → IBM Australia Pty Ltd
        hp inc → HP Inc
        AT&T CORPORATION → AT&T Corporation
        TECH SOLUTIONS LLC → Tech Solutions LLC
    
    Args:
        name: The company name to process
        preserve_acronyms: If True, keep known acronyms in uppercase (default: True)
        
    Returns:
        The name with properly formatted suffixes and optionally preserved acronyms,
        or None if input is None
    """
    if not name:
        return name
    
    # First apply basic suffix title casing
    result = title_case_legal_suffix(name)
    
    # Optionally preserve common acronyms
    if preserve_acronyms:
        for acronym in PRESERVE_ACRONYMS:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(acronym) + r'\b'
            result = re.sub(pattern, acronym, result, flags=re.IGNORECASE)
    
    return result


def batch_title_case_suffixes(names: list, smart: bool = True, preserve_acronyms: bool = True) -> list:
    """
    Apply title casing to a batch of company names.
    
    Args:
        names: List of company names to process
        smart: If True, use smart_title_case_suffix; if False, use basic title_case_legal_suffix
        preserve_acronyms: If True and smart=True, preserve known acronyms
        
    Returns:
        List of processed names with the same length as input
    """
    if smart:
        return [smart_title_case_suffix(name, preserve_acronyms) for name in names]
    else:
        return [title_case_legal_suffix(name) for name in names]
