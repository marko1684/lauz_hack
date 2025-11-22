import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict
import logging
import json
import sqlite3
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PatentScraper:
    def __init__(self, csv_path: str):
        """
        Initialize the patent scraper with a CSV file path.
        
        Args:
            csv_path: Path to the CSV file containing patent links
        """
        self.csv_path = csv_path
        self.df = None
        self.results = []
        self.patent_metadata = []
        
    def load_csv(self):
        """Load the CSV file into a pandas DataFrame."""
        try:
            # Skip the first row which contains search URL
            self.df = pd.read_csv(self.csv_path, skiprows=1)
            logger.info(f"Successfully loaded CSV with {len(self.df)} rows")
            logger.info(f"Columns: {self.df.columns.tolist()}")
            return True
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return False
    
    def get_patent_data(self) -> List[Dict]:
        """
        Extract patent links, titles, and publication dates from the CSV.
        
        Returns:
            List of dictionaries containing url, title, and publication_date
        """
        if self.df is None:
            logger.error("CSV not loaded. Call load_csv() first.")
            return []
        
        try:
            patent_data = []
            
            # Get column names or indices
            if 'result link' in self.df.columns:
                links = self.df['result link']
            else:
                links = self.df.iloc[:, 8]  # Column I (index 8)
            
            if 'title' in self.df.columns:
                titles = self.df['title']
            else:
                titles = self.df.iloc[:, 1]  # Column B (index 1)
            
            if 'publication date' in self.df.columns:
                pub_dates = self.df['publication date']
            else:
                pub_dates = self.df.iloc[:, 6]  # Column G (index 6)
            
            # Combine the data
            for idx in range(len(self.df)):
                link = links.iloc[idx]
                if pd.notna(link):
                    patent_data.append({
                        'url': link,
                        'title': titles.iloc[idx] if pd.notna(titles.iloc[idx]) else '',
                        'publication_date': pub_dates.iloc[idx] if pd.notna(pub_dates.iloc[idx]) else ''
                    })
            
            logger.info(f"Found {len(patent_data)} patents to scrape")
            return patent_data
        except Exception as e:
            logger.error(f"Error extracting patent data: {e}")
            return []
    
    def scrape_patent_details(self, patent_data: Dict) -> Dict[str, str]:
        """
        Scrape the abstract, description, and claims from a patent page.
        
        Args:
            patent_data: Dictionary containing url, title, and publication_date
            
        Returns:
            Dictionary containing URL, title, publication_date, abstract, description, claims, and error
        """
        result = {
            'url': patent_data['url'],
            'title': patent_data['title'],
            'publication_date': patent_data['publication_date'],
            'abstract': None,
            'description': None,
            'claims': None,
            'error': None
        }
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(patent_data['url'], headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Google Patents structure
            if 'patents.google.com' in patent_data['url']:
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
            
            # USPTO
            elif 'uspto.gov' in patent_data['url']:
                # Abstract
                abstract_elem = soup.find('p', {'id': 'abstract'})
                if not abstract_elem:
                    abstract_elem = soup.find('div', {'id': 'abstract'})
                if abstract_elem:
                    result['abstract'] = abstract_elem.get_text(strip=True)
                
                # Description
                desc_elem = soup.find('div', {'id': 'description'})
                if desc_elem:
                    result['description'] = desc_elem.get_text(strip=True)
                
                # Claims
                claims_elem = soup.find('div', {'id': 'claims'})
                if claims_elem:
                    result['claims'] = claims_elem.get_text(strip=True)
            
            # EPO (European Patent Office)
            elif 'epo.org' in patent_data['url']:
                abstract_elem = soup.find('div', {'class': 'abstract'})
                if abstract_elem:
                    result['abstract'] = abstract_elem.get_text(strip=True)
                
                desc_elem = soup.find('div', {'class': 'description'})
                if desc_elem:
                    result['description'] = desc_elem.get_text(strip=True)
                
                claims_elem = soup.find('div', {'class': 'claims'})
                if claims_elem:
                    result['claims'] = claims_elem.get_text(strip=True)
            
            # Log success/warnings
            scraped_fields = []
            if result['abstract']:
                scraped_fields.append('abstract')
            if result['description']:
                scraped_fields.append('description')
            if result['claims']:
                scraped_fields.append('claims')
            
            if scraped_fields:
                logger.info(f"Successfully scraped {', '.join(scraped_fields)} from: {patent_data['url']}")
            else:
                result['error'] = "No content found on page"
                logger.warning(f"No content found for: {patent_data['url']}")
                
        except requests.exceptions.RequestException as e:
            result['error'] = f"Request error: {str(e)}"
            logger.error(f"Error scraping {patent_data['url']}: {e}")
        except Exception as e:
            result['error'] = f"Parsing error: {str(e)}"
            logger.error(f"Error parsing {patent_data['url']}: {e}")
        
        return result
    
    def scrape_all(self, delay: float = 1.0) -> pd.DataFrame:
        """
        Scrape abstract, description, and claims from all patents.
        
        Args:
            delay: Delay in seconds between requests (to be respectful)
            
        Returns:
            DataFrame with URLs, titles, publication dates, abstract, description, and claims
        """
        patent_data_list = self.get_patent_data()
        
        if not patent_data_list:
            logger.error("No patents to scrape")
            return pd.DataFrame()
        
        logger.info(f"Starting to scrape {len(patent_data_list)} patents...")
        
        for i, patent_data in enumerate(patent_data_list, 1):
            logger.info(f"Processing {i}/{len(patent_data_list)}: {patent_data['url']}")
            result = self.scrape_patent_details(patent_data)
            self.results.append(result)
            
            # Be respectful and add delay between requests
            if i < len(patent_data_list):
                time.sleep(delay)
        
        results_df = pd.DataFrame(self.results)
        successful = sum(1 for r in self.results if r['abstract'] or r['description'] or r['claims'])
        logger.info(f"Scraping complete. Successfully scraped content from {successful} patents")
        
        return results_df
    
    def save_results(self, output_path: str = 'patent_descriptions.json', format: str = 'json'):
        """
        Save the scraped results to a file in the specified format.
        
        Args:
            output_path: Path for the output file
            format: Output format - 'json', 'csv', 'sqlite', or 'parquet'
        """
        if not self.results:
            logger.warning("No results to save")
            return
        
        df = pd.DataFrame(self.results)
        
        if format == 'json':
            # Add newlines to long text fields for better readability
            formatted_results = []
            for result in self.results:
                formatted_result = result.copy()
                # Add newlines every 100 characters in long text fields
                for field in ['abstract', 'description', 'claims']:
                    if formatted_result.get(field):
                        text = formatted_result[field]
                        # Split into sentences and add newlines periodically
                        sentences = text.replace('. ', '.\n').replace('.\n', '. ').split('. ')
                        formatted_text = []
                        current_line = []
                        current_length = 0
                        
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if not sentence:
                                continue
                            sentence_with_period = sentence if sentence.endswith('.') else sentence + '.'
                            
                            if current_length + len(sentence_with_period) > 100 and current_line:
                                formatted_text.append(' '.join(current_line))
                                current_line = [sentence_with_period]
                                current_length = len(sentence_with_period)
                            else:
                                current_line.append(sentence_with_period)
                                current_length += len(sentence_with_period) + 1
                        
                        if current_line:
                            formatted_text.append(' '.join(current_line))
                        
                        formatted_result[field] = '\n'.join(formatted_text)
                
                formatted_results.append(formatted_result)
            
            # Save as JSON with proper formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_path} (JSON format)")
            
        elif format == 'sqlite':
            # Save to SQLite database
            conn = sqlite3.connect(output_path)
            df.to_sql('patents', conn, if_exists='replace', index=False)
            conn.close()
            logger.info(f"Results saved to {output_path} (SQLite database)")
            
        elif format == 'parquet':
            # Save as Parquet
            df.to_parquet(output_path, index=False, engine='pyarrow')
            logger.info(f"Results saved to {output_path} (Parquet format)")
            
        elif format == 'csv':
            # Save as CSV (with limitations)
            df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path} (CSV format - may truncate long text)")
            
        else:
            logger.error(f"Unknown format: {format}. Using JSON.")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_path} (JSON format)")


def main():
    """Example usage of the PatentScraper."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python scraper.py <csv_file_path> [output_file_path] [format]")
        print("Example: python scraper.py patents.csv output.json json")
        print("Formats: json (default), csv, sqlite, parquet")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'patent_descriptions.json'
    
    # Determine format from file extension or argument
    if len(sys.argv) > 3:
        format = sys.argv[3].lower()
    else:
        # Auto-detect from file extension
        ext = Path(output_path).suffix.lower()
        if ext == '.json':
            format = 'json'
        elif ext == '.db' or ext == '.sqlite':
            format = 'sqlite'
        elif ext == '.parquet':
            format = 'parquet'
        elif ext == '.csv':
            format = 'csv'
        else:
            format = 'json'
            logger.info(f"Unknown extension '{ext}', defaulting to JSON format")
    
    # Create scraper instance
    scraper = PatentScraper(csv_path)
    
    # Load CSV
    if not scraper.load_csv():
        sys.exit(1)
    
    # Scrape all patents
    results_df = scraper.scrape_all(delay=1.0)
    
    # Save results
    scraper.save_results(output_path, format=format)
    
    # Print summary
    print("\n" + "="*50)
    print("SCRAPING SUMMARY")
    print("="*50)
    print(f"Total patents processed: {len(results_df)}")
    if len(results_df) > 0:
        print(f"Patents with abstract: {results_df['abstract'].notna().sum()}")
        print(f"Patents with description: {results_df['description'].notna().sum()}")
        print(f"Patents with claims: {results_df['claims'].notna().sum()}")
        successful = sum(1 for _, row in results_df.iterrows() 
                        if pd.notna(row['abstract']) or pd.notna(row['description']) or pd.notna(row['claims']))
        print(f"Patents with at least one field: {successful}")
        print(f"Results saved to: {output_path}")
    else:
        print("No results to process")


if __name__ == "__main__":
    main()
