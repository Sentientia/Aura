import requests
from bs4 import BeautifulSoup
# import typesense
# from sentence_transformers import SentenceTransformer
from agent.actions.utils import parse_payload
import requests
from agent.actions.action import Action
from agent.controller.state import State
import json
from googlesearch import search
from bs4 import BeautifulSoup
import requests
import wikipediaapi
import re




class WebSearchAdvancedAction(Action):
    def __init__(self, thought: str, payload: str):
        super().__init__(thought, payload)
        self.JUNK_PATTERNS = re.compile(r"(nav|header|footer|sidebar|menu|social|ads)", re.I)
        # self.client = typesense.Client({ #TODO: Change to TypeSense Cloud if needed
        #     'api_key': 'xyz',
        #     'nodes': [{'host': 'localhost', 'port': '8108', 'protocol': 'http'}],
        #     'connection_timeout_seconds': 2
        # })
        # self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # setup_collection()

    def clean_text(self, text: str) -> str:
        return text.encode('ascii', 'ignore').decode()
    
    def get_top_google_links(self, query, num_results=3):
        results = []
        for url in search(query, num_results=10):  # get more in case of filtering
            if "wikipedia.org" not in url:
                results.append(url)
            if len(results) >= num_results:
                break
        return results
    
    # --- WIKIPEDIA SEARCH ---
    def get_wikipedia_summary(self, query):
        wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='my-agent/1.0 (leander.maben@gmail.com)'#TODO: Change to a proper id
        )
        page = wiki.page(query)
        if page.exists():
            return [["Wikipedia", self.clean_text(page.summary)]]
        return [["Wikipedia", f"No Wikipedia article found for '{query}'."]]

    
    def clean_page(self, url):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            html = requests.get(url, timeout=10, headers=headers).text
            soup = BeautifulSoup(html, "html.parser")

            # Remove known junk sections by tag and class/id
            for tag in soup.find_all(['script', 'style', 'noscript', 'footer', 'header', 'nav']):
                tag.decompose()
            for div in soup.find_all(True, {'class': self.JUNK_PATTERNS, 'id': self.JUNK_PATTERNS}):
                div.decompose()

            text = soup.get_text(separator=' ', strip=True)
            return self.clean_text(re.sub(r"\s+", " ", text)[:10000])
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return ""
        
    def run_query_pipeline(self, query):
        urls = self.get_top_google_links(query)
        results = []
        for url in urls:
            print(f"Fetching: {url}")
            text = self.clean_page(url)
            results.append((url, text[:500]))  # preview first 500 chars
        return results
    
    # def crawl_and_clean(url):
    #     try:
    #         html = requests.get(url, timeout=10).text
    #         soup = BeautifulSoup(html, "html.parser")
    #         for s in soup(["script", "style"]): s.decompose()
    #         text = soup.get_text(separator=" ", strip=True)
    #         return text[:10000]  # Limit size
    #     except Exception as e:
    #         print(f"Failed to crawl {url}: {e}")
    #         return ""


    # def setup_collection(client, model):
    #     schema = {
    #         "name": "documents",
    #         "fields": [
    #             {"name": "id", "type": "string"},
    #             {"name": "url", "type": "string"},
    #             {"name": "text", "type": "string"},
    #             {"name": "embedding", "type": "float[]", "num_dim": 384}
    #         ],
    #         "default_sorting_field": "id"
    #     }
    #     try:
    #         client.collections.create(schema)
    #     except:
    #         pass  # Already exists

    # def index_url(url, doc_id):
    #     text = crawl_and_clean(url)
    #     if not text: return
    #     embedding = model.encode(text).tolist()
    #     client.collections['documents'].documents.upsert({
    #         "id": doc_id,
    #         "url": url,
    #         "text": text[:500],  # Preview
    #         "embedding": embedding
    #     })

        
    def execute(self, state: State) -> str:
        payload = parse_payload(self.payload)
        
        google_results = self.run_query_pipeline(payload["google_search_query"])
        wikipedia_results = self.get_wikipedia_summary(payload["wikipedia_search_query"])
        results = google_results + wikipedia_results
        state.history.append({
            'action': {'type': 'web_search', 'payload': self.payload},
            'observation': {"type": "web_search", "payload": self.clean_text(json.dumps(results))}
        })

        return json.dumps(results)
    
if __name__ == "__main__":
    action = WebSearchAdvancedAction(thought="", payload=json.dumps({"google_search_query":"weather in Pittsburgh today", "wikipedia_search_query":"weather in Pittsburgh today"}))
    print(action.execute(State()))
        


