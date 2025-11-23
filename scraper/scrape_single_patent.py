#!/usr/bin/env python3
"""
Script to scrape a single patent and apply cleaning function for testing
"""
import requests
from bs4 import BeautifulSoup
import json
import re

def remove_junk(text):
    """Remove FIG references, bracketed numbers, and parenthetical references."""
    if not text:
        return text
    
    # Remove FIG. X, Figure X, etc.
    text = re.sub(r'\bFIG\.?\s*\d+[A-Za-z]?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bFigure\s+\d+[A-Za-z]?\b', '', text, flags=re.IGNORECASE)
    
    # Remove bracketed numbers like [1], [23], etc.
    text = re.sub(r'\[\d+\]', '', text)
    
    # Remove parenthetical references like (see Fig. 3)
    text = re.sub(r'\([^)]*[Ff]ig[^)]*\)', '', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def scrape_patent(url):
    """Scrape a single patent from Google Patents."""
    print(f"Scraping patent from: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        result = {
            'url': url,
            'title': None,
            'publication_date': None,
            'abstract': None,
            'description': None,
            'claims': None
        }
        
        # Extract title
        title_elem = soup.find('meta', {'name': 'DC.title'})
        if title_elem:
            result['title'] = title_elem.get('content', '').strip()
        
        # Extract publication date
        date_elem = soup.find('meta', {'name': 'DC.date'})
        if date_elem:
            result['publication_date'] = date_elem.get('content', '').strip()
        
        # Extract Abstract
        abstract_elem = soup.find('div', {'class': 'abstract'})
        if not abstract_elem:
            abstract_elem = soup.find('section', {'itemprop': 'abstract'})
        if abstract_elem:
            result['abstract'] = abstract_elem.get_text(strip=True)
        
        # Extract Description
        desc_elem = soup.find('section', {'itemprop': 'description'})
        if not desc_elem:
            desc_elem = soup.find('div', {'class': 'description'})
        if desc_elem:
            result['description'] = desc_elem.get_text(strip=True)
        
        # Extract Claims
        claims_elem = soup.find('section', {'itemprop': 'claims'})
        if not claims_elem:
            claims_elem = soup.find('div', {'class': 'claims'})
        if claims_elem:
            result['claims'] = claims_elem.get_text(strip=True)
        
        print(f"\n✓ Successfully scraped patent: {result['title']}")
        print(f"  Publication Date: {result['publication_date']}")
        print(f"  Abstract length: {len(result['abstract']) if result['abstract'] else 0} chars")
        print(f"  Description length: {len(result['description']) if result['description'] else 0} chars")
        print(f"  Claims length: {len(result['claims']) if result['claims'] else 0} chars")
        
        return result
        
    except Exception as e:
        print(f"✗ Error scraping patent: {e}")
        return None

def clean_patent_data(patent_data):
    """Apply junk removal to all patent sections."""
    if not patent_data:
        return None
    
    cleaned = patent_data.copy()
    
    print("\nApplying cleaning function to remove junk...")
    
    if cleaned['abstract']:
        original_len = len(cleaned['abstract'])
        cleaned['abstract'] = remove_junk(cleaned['abstract'])
        print(f"  Abstract: {original_len} → {len(cleaned['abstract'])} chars")
    
    if cleaned['description']:
        original_len = len(cleaned['description'])
        cleaned['description'] = remove_junk(cleaned['description'])
        print(f"  Description: {original_len} → {len(cleaned['description'])} chars")
    
    if cleaned['claims']:
        original_len = len(cleaned['claims'])
        cleaned['claims'] = remove_junk(cleaned['claims'])
        print(f"  Claims: {original_len} → {len(cleaned['claims'])} chars")
    
    return cleaned

def save_to_json(data, filename):
    """Save patent data to JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved to: {filename}")

def main():
    patent_url = "https://patents.google.com/patent/US20020185086A1/en"
    
    # Scrape the patent
    patent_data = scrape_patent(patent_url)
    
    if patent_data:
        # Save raw data
        save_to_json(patent_data, 'test_patent_raw.json')
        
        # Clean the data
        cleaned_data = clean_patent_data(patent_data)
        
        # Save cleaned data
        save_to_json(cleaned_data, 'test_patent_cleaned.json')
        
        # Create a combined text for easy testing
        combined_text = ""
        if cleaned_data['abstract']:
            combined_text += "ABSTRACT:\n" + cleaned_data['abstract'] + "\n\n"
        if cleaned_data['description']:
            combined_text += "DESCRIPTION:\n" + cleaned_data['description'] + "\n\n"
        if cleaned_data['claims']:
            combined_text += "CLAIMS:\n" + cleaned_data['claims']
        
        with open('test_patent_text.txt', 'w', encoding='utf-8') as f:
            f.write(combined_text)
        print(f"✓ Saved combined text to: test_patent_text.txt")
        
        print("\n" + "="*60)
        print("READY FOR TESTING!")
        print("="*60)
        print("\nYou can now:")
        print("1. Use test_patent_text.txt - Copy/paste into the search box")
        print("2. Use test_patent_cleaned.json - For programmatic testing")
        print("3. Compare with test_patent_raw.json - See what was cleaned")
        
    else:
        print("\n✗ Failed to scrape patent")

if __name__ == "__main__":
    main()
