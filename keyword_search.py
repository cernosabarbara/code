import requests
from collections import Counter
from tqdm import tqdm
from datetime import datetime

BASE_URL = "https://api.openalex.org/works"
KEYWORDS = [
    "artificial intelligence", "workplace surveillance", "algorithmic management",
    "employee monitoring", "bossware", "digital monitoring", "productivity tracking",
    "workplace privacy", "surveillance ethics", "algorithmic bias", "transparency in AI",
    "AI and human rights", "legislation on AI surveillance", "data protection at work"
]
CURRENT_YEAR = datetime.now().year
YEAR_RANGE = f"{CURRENT_YEAR-9}-{CURRENT_YEAR}"

def fetch_papers(keyword, year_range=YEAR_RANGE, per_page=200, max_pages=2):
    """Fetch papers from OpenAlex with a page limit for testing."""
    papers = []
    page = 1
    while page <= max_pages:
        url = f"{BASE_URL}?filter=title.search:{keyword},publication_year:{year_range}&per-page={per_page}&page={page}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = data.get('results', [])
            if not results:
                break
            
            papers.extend(results)
            total_count = data['meta']['count']
            total_pages = (total_count + per_page - 1) // per_page
            print(f"Keyword '{keyword}': Page {page}/{min(total_pages, max_pages)}, {len(results)} papers (expected ≤ {per_page})")
            page += 1
        except requests.RequestException as e:
            print(f"⚠️ Error fetching '{keyword}' on page {page}: {e}")
            break
    return papers



def extract_title_keywords(title):
    """Simple keyword extraction from titles (split and lowercase)."""
    # Basic splitting on spaces, remove short words and common stop words
    stop_words = {"in", "of", "and", "the", "to", "is", "a", "on"}
    words = title.lower().split()
    # Filter to multi-word phrases or significant terms (adjust as needed)
    keywords = [word for word in words if len(word) > 3 and word not in stop_words]
    return keywords



def main():
    all_papers = []
    keyword_occurrences = Counter()

    for keyword in tqdm(KEYWORDS, desc="Searching OpenAlex"):
        papers = fetch_papers(keyword)
        if papers:
            paper_ids = {paper['id'] for paper in all_papers}
            new_papers = [paper for paper in papers if paper['id'] not in paper_ids]
            all_papers.extend(new_papers)
            for paper in new_papers:
                title = paper.get('title', '')
                if title:
                    title_keywords = extract_title_keywords(title)
                    keyword_occurrences.update(title_keywords)

    if all_papers:
        print(f"\n✅ Retrieved {len(all_papers)} unique papers.")
        print("\nRelated Keywords from Titles (Top 10):")
        for keyword, count in keyword_occurrences.most_common(30):
            print(f"{keyword}: {count}")
    else:
        print("\n⚠️ No papers retrieved.")



if __name__ == "__main__":
    main()

    
