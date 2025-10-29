"""
Preprocessing Module for Company Name Normalization

This module provides functions to normalize and standardize company names
for better matching and clustering.
"""

import re
import pandas as pd
from typing import List, Set, Dict, Optional

# Define company legal suffixes for normalization
LEGAL_SUFFIXES = {
    'pty ltd': 'pty ltd',
    'pty limited': 'pty ltd',
    'proprietary limited': 'pty ltd',
    'ltd': 'ltd',
    'limited': 'ltd',
    'inc': 'inc',
    'incorporated': 'inc',
    'llc': 'llc',
    'corp': 'corp',
    'corporation': 'corp',
    'group': 'group',
    'holdings': 'holdings',
}

# Define common abbreviations
COMMON_ABBR = {
    'aus': 'australia',
    'aust': 'australia',
    'intl': 'international',
    'int': 'international',
    'svcs': 'services',
    'svc': 'service',
    'tech': 'technology',
    'mfg': 'manufacturing',
    'mgmt': 'management',
    'dev': 'development',
    'assoc': 'associates',
}

def load_common_terms(filepath: str) -> Set[str]:
    """Load common terms from file"""
    try:
        with open(filepath, 'r') as f:
            return {line.strip().lower() for line in f if line.strip()}
    except Exception as e:
        print(f"Warning: Could not load common terms from {filepath}: {e}")
        return set()

def normalize_company_name(name: str, 
                          remove_legal_suffix: bool = False,
                          remove_common_terms: bool = False,
                          common_terms: Set[str] = None) -> str:
    """
    Enhanced company name normalization
    
    Args:
        name: Company name to normalize
        remove_legal_suffix: Whether to remove legal suffixes
        remove_common_terms: Whether to remove common industry terms
        common_terms: Set of common terms to remove
        
    Returns:
        Normalized company name
    """
    if not name or pd.isna(name):
        return ""
    
    # Convert to lowercase and strip whitespace
    name = str(name).lower().strip()
    
    # Remove special characters and replace with space
    name = re.sub(r'[^\w\s]', ' ', name)
    
    # Remove extra whitespace
    name = ' '.join(name.split())
    
    # Expand common abbreviations
    tokens = name.split()
    for i, token in enumerate(tokens):
        if token in COMMON_ABBR:
            tokens[i] = COMMON_ABBR[token]
    name = ' '.join(tokens)
    
    # Handle legal suffixes
    for suffix, replacement in LEGAL_SUFFIXES.items():
        pattern = r'\b' + re.escape(suffix) + r'\b'
        if remove_legal_suffix:
            name = re.sub(pattern, '', name)
        else:
            name = re.sub(pattern, replacement, name)
    
    # Remove common terms if requested
    if remove_common_terms and common_terms:
        tokens = name.split()
        tokens = [t for t in tokens if t not in common_terms]
        name = ' '.join(tokens)
    
    # Clean up again
    return ' '.join(name.split())

def extract_company_core_name(name: str) -> str:
    """Extract core company name without legal suffixes"""
    return normalize_company_name(name, remove_legal_suffix=True)

def tokenize_company_name(name: str) -> List[str]:
    """Split company name into tokens"""
    normalized = normalize_company_name(name)
    return normalized.split()

def get_company_first_letter(name: str) -> str:
    """Get first letter of company name for blocking"""
    normalized = normalize_company_name(name)
    return normalized[0] if normalized else ""
