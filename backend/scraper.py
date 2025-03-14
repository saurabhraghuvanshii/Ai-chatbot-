import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import os
import json
import time
from urllib.parse import urljoin

class JenkinsDocScraper:
    def __init__(self):
        self.base_urls = {
            "docs": "https://www.jenkins.io/doc/",
            "plugins": "https://plugins.jenkins.io/",
            "blog": "https://www.jenkins.io/node/",
        }
        self.output_dir = "backend/data/raw"
        self.visited_urls = set()
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_page(self, url: str) -> str:
        """Fetch content from a URL with rate limiting"""
        try:
            time.sleep(1)  # Rate limiting
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return ""

    def parse_documentation(self, html: str, url: str) -> Dict:
        """Parse documentation HTML and extract relevant content"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Extract links for further crawling
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('/') or href.startswith(self.base_urls['docs']):
                full_url = urljoin(url, href)
                if full_url not in self.visited_urls:
                    links.append(full_url)
        
        return {
            "content": text,
            "title": soup.title.string if soup.title else "",
            "headers": [h.get_text() for h in soup.find_all(['h1', 'h2', 'h3'])],
            "links": links
        }

    def crawl_documentation(self, url: str, depth: int = 2) -> None:
        """Recursively crawl documentation pages"""
        if depth == 0 or url in self.visited_urls:
            return

        self.visited_urls.add(url)
        print(f"Crawling: {url}")
        
        html = self.fetch_page(url)
        if not html:
            return

        parsed = self.parse_documentation(html, url)
        
        # Save the current page
        filename = url.replace('https://', '').replace('/', '_')
        output_file = os.path.join(self.output_dir, f"{filename}.json")
        with open(output_file, 'w') as f:
            json.dump(parsed, f, indent=2)
        
        # Crawl linked pages
        for link in parsed['links']:
            self.crawl_documentation(link, depth - 1)

    def scrape_documentation(self):
        """Scrape Jenkins documentation"""
        for source, base_url in self.base_urls.items():
            print(f"Scraping {source} documentation...")
            self.crawl_documentation(base_url)
            print(f"Completed scraping {source}")

def main():
    scraper = JenkinsDocScraper()
    scraper.scrape_documentation()

if __name__ == "__main__":
    main()
