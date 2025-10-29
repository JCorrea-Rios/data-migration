"""
Blocking Module for Company Entity Resolution

This module provides functions to reduce the number of pairwise comparisons
needed for clustering by grouping similar companies into blocks.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple
from src.preprocessing import normalize_company_name, get_company_first_letter

def create_letter_blocks(companies: List[str]) -> Dict[str, List[int]]:
    """
    Create blocks based on first letter of company name.
    
    Args:
        companies: List of company names
        
    Returns:
        Dictionary mapping first letter to list of company indices
    """
    blocks = {}
    
    for idx, company in enumerate(companies):
        first_letter = get_company_first_letter(company)
        if first_letter:
            if first_letter not in blocks:
                blocks[first_letter] = []
            blocks[first_letter].append(idx)
    
    return blocks

def generate_candidate_pairs(blocks: Dict[str, List[int]]) -> List[Tuple[int, int]]:
    """
    Generate candidate pairs from blocks.
    
    Args:
        blocks: Dictionary mapping block key to list of company indices
        
    Returns:
        List of (idx1, idx2) tuples representing candidate pairs
    """
    pairs = set()
    
    for block_key, indices in blocks.items():
        # Generate all pairs within this block
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx1 = indices[i]
                idx2 = indices[j]
                if idx1 != idx2:
                    # Ensure consistent ordering
                    pair = (min(idx1, idx2), max(idx1, idx2))
                    pairs.add(pair)
    
    return list(pairs)

def apply_blocking(companies: List[str], method: str = 'letter') -> List[Tuple[int, int]]:
    """
    Apply blocking to generate candidate pairs.
    
    Args:
        companies: List of company names
        method: Blocking method ('letter' is the only implemented method)
        
    Returns:
        List of (idx1, idx2) tuples representing candidate pairs
    """
    if method == 'letter':
        blocks = create_letter_blocks(companies)
    else:
        raise ValueError(f"Unknown blocking method: {method}")
    
    return generate_candidate_pairs(blocks)
