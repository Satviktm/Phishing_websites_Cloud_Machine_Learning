import re
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup

COMMON_LEGIT_DOMAINS = ['google.com', 'facebook.com', 'amazon.com', 'wikipedia.org',
                          'youtube.com', 'twitter.com', 'linkedin.com', 'microsoft.com']

def estimate_url_similarity(domain: str) -> float:
    domain_clean = domain.lower().replace('www.', '')
    for legit in COMMON_LEGIT_DOMAINS:
        if domain_clean == legit:
            return 100.0
    best_score = 0
    for legit in COMMON_LEGIT_DOMAINS:
        common = sum(1 for a, b in zip(domain_clean, legit) if a == b)
        score = (common / max(len(domain_clean), len(legit))) * 100
        best_score = max(best_score, score)
    return best_score


def extract_features(url: str) -> dict:
    """Extracts the 30 features needed by the models, live from a URL."""
    parsed = urlparse(url if url.startswith('http') else 'http://' + url)
    domain = parsed.netloc

    features = {}

    # --- URL-based features (no page fetch needed) ---
    features['URLLength'] = len(url)
    features['DomainLength'] = len(domain)
    features['IsHTTPS'] = 1 if parsed.scheme == 'https' else 0
    features['NoOfSubDomain'] = max(domain.count('.') - 1, 0)
    features['NoOfDegitsInURL'] = sum(c.isdigit() for c in url)
    features['NoOfLettersInURL'] = sum(c.isalpha() for c in url)
    features['DegitRatioInURL'] = features['NoOfDegitsInURL'] / max(len(url), 1)
    features['LetterRatioInURL'] = features['NoOfLettersInURL'] / max(len(url), 1)
    special_chars = len(re.findall(r'[^a-zA-Z0-9.]', url))
    features['NoOfOtherSpecialCharsInURL'] = special_chars
    features['SpacialCharRatioInURL'] = special_chars / max(len(url), 1)
    features['CharContinuationRate'] = 1.0
    features['URLSimilarityIndex'] = estimate_url_similarity(domain)
    features['URLCharProb'] = 0.5

    # --- Content-based features (require fetching the page) ---
    try:
        resp = requests.get(url if url.startswith('http') else 'http://' + url, timeout=6,
                             headers={'User-Agent': 'Mozilla/5.0'})
        html = resp.text
        soup = BeautifulSoup(html, 'html.parser')

        features['LineOfCode'] = len(html.splitlines())
        features['LargestLineLength'] = max((len(l) for l in html.splitlines()), default=0)
        features['NoOfImage'] = len(soup.find_all('img'))
        features['NoOfCSS'] = len(soup.find_all('link', rel='stylesheet')) + len(soup.find_all('style'))
        features['NoOfJS'] = len(soup.find_all('script'))
        features['NoOfiFrame'] = len(soup.find_all('iframe'))
        features['HasDescription'] = 1 if soup.find('meta', attrs={'name': 'description'}) else 0
        features['HasFavicon'] = 1 if soup.find('link', rel=lambda x: x and 'icon' in x.lower()) else 0
        features['HasSubmitButton'] = 1 if soup.find('button', type='submit') or soup.find('input', type='submit') else 0
        features['HasSocialNet'] = 1 if any(s in html.lower() for s in ['facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com']) else 0
        features['HasCopyrightInfo'] = 1 if '©' in html or 'copyright' in html.lower() else 0
        features['IsResponsive'] = 1 if soup.find('meta', attrs={'name': 'viewport'}) else 0

        title_tag = soup.find('title')
        title_text = title_tag.text if title_tag else ''
        domain_root = domain.split('.')[0].lower() if domain else ''
        features['DomainTitleMatchScore'] = 100.0 if domain_root and domain_root in title_text.lower() else 30.0
        features['URLTitleMatchScore'] = features['DomainTitleMatchScore']

        links = soup.find_all('a', href=True)
        self_ref = sum(1 for l in links if domain in l['href'] or l['href'].startswith('/'))
        external_ref = sum(1 for l in links if l['href'].startswith('http') and domain not in l['href'])
        empty_ref = sum(1 for l in links if l['href'] in ['#', ''])
        features['NoOfSelfRef'] = self_ref
        features['NoOfExternalRef'] = external_ref
        features['NoOfEmptyRef'] = empty_ref

    except Exception as e:
        defaults = {'LineOfCode': 0, 'LargestLineLength': 0, 'NoOfImage': 0, 'NoOfCSS': 0,
                     'NoOfJS': 0, 'NoOfiFrame': 0, 'HasDescription': 0, 'HasFavicon': 0,
                     'HasSubmitButton': 0, 'HasSocialNet': 0, 'HasCopyrightInfo': 0,
                     'IsResponsive': 0, 'DomainTitleMatchScore': 0, 'URLTitleMatchScore': 0,
                     'NoOfSelfRef': 0, 'NoOfExternalRef': 0, 'NoOfEmptyRef': 0}
        features.update(defaults)
        features['_fetch_error'] = str(e)

    return features