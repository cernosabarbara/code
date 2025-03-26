import requests
import pandas as pd
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx

BASE_URL = "https://api.openalex.org/works"
SEARCH_QUERIES = [
     "artificial intelligence workplace surveillance",
    "AI workplace surveillance",
    "artificial intelligence employee monitoring",
    "AI employee monitoring",
    "algorithmic workplace surveillance",
    "digital workplace monitoring",
    "AI workplace monitoring",
    "artificial intelligence employee surveillance",
    "AI worker surveillance",
    "artificial intelligence employee tracking", 
    "bossware", "hubstaff",
    "technology workplace surveillance"
]
YEAR_RANGE = "2015-2025"

def fetch_papers(query, search_field="title.search"):
    papers = []
    for page in tqdm(range(1, 21), desc=f"'{query}' ({search_field})", leave=False):
        url = f"{BASE_URL}?filter={search_field}:{query},publication_year:{YEAR_RANGE}&per-page=200&page={page}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            results = response.json().get('results', [])
            if not results:
                break
            papers.extend(results)
        except (requests.RequestException, ValueError) as e:
            print(f"Error fetching '{query}' ({search_field}) page {page}: {e}")
            break
    return papers

def get_keywords(title):
    stop_words = {"in", "of", "and", "the", "to", "is", "a", "on", "for", "with"}
    if title is None:
        return []
    words = title.lower().split()
    return [w for w in words if len(w) > 3 and w not in stop_words]

all_papers = []
seen_ids = set()
query_counts = {}

for query in tqdm(SEARCH_QUERIES, desc="Searching OpenAlex"):
    papers = fetch_papers(query, "title.search") + fetch_papers(query, "abstract.search")
    new_papers = [p for p in papers if p['id'] not in seen_ids]
    query_counts[query] = len(new_papers)
    all_papers.extend(new_papers)
    seen_ids.update(p['id'] for p in new_papers)


for query, count in sorted(query_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{query}: {count}")

data = [{"year": int(p.get('publication_year', 0)),
         "authors": [a['author']['display_name'] for a in p.get('authorships', []) if 'author' in a],
         "institutions": [i['display_name'] for a in p.get('authorships', []) for i in a.get('institutions', [])],
         "keywords": get_keywords(p.get('title'))} for p in all_papers]

df = pd.DataFrame(data)
df.to_csv("ai_surveillance_openalex.csv", index=False)

pd.DataFrame(Counter(kw for keywords in df['keywords'] for kw in keywords).items(), columns=['Keyword', 'Count']).sort_values('Count', ascending=False).to_csv("keywords.csv", index=False)
pd.DataFrame(Counter(a for authors in df['authors'] for a in authors).items(), columns=['Author', 'Papers']).sort_values('Papers', ascending=False).to_csv("authors.csv", index=False)
pd.DataFrame(Counter(i for insts in df['institutions'] for i in insts).items(), columns=['Institution', 'Papers']).sort_values('Papers', ascending=False).to_csv("institutions.csv", index=False)

for file in ["keywords.csv", "authors.csv", "institutions.csv"]:
    print(pd.read_csv(file).head(5).to_string(index=False))
    
plt.figure(figsize=(4, 3))
plt.bar(year_counts.index, year_counts.values, color='skyblue')
plt.xlabel('Year')
plt.ylabel('Number of Publications')
plt.title('Publications on AI Workplace Surveillance (2015-2025)')
plt.xticks(year_counts.index, rotation=45)
plt.tight_layout()
plt.savefig("publications_by_year.png")
plt.show()

