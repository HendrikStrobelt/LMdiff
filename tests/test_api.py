from typing import *
from fastapi.testclient import TestClient
from urllib.parse import urlencode
from server import app
from server.api import GoodbyePayload

client = TestClient(app)

def make_url(baseurl, to_send:Optional[Dict[str, Any]]=None):
    if to_send is None:
        return baseurl
    return baseurl + "?" + urlencode(to_send)

def test_get_projects():
    response = client.get(make_url("/api/all_projects"))
    r = response.json()
    assert isinstance(r, list)
    assert len(r) > 0
    assert 'model' in set(r[0].keys())

def test_get_suggestions():
    request = {
        "m1": "gpt2",
        "m2": "distilgpt2",
        "corpus": "wiki_split"
    }
    response = client.get(make_url("/api/suggestions", request))
    r = response.json()
    assert r['request']['m1'] == request['m1']
    assert r['request']['m2'] == request['m2']
    assert r['request']['corpus'] == request['corpus']

def test_analyze():
    request = {
        "m1": "gpt2",
        "m2": "distilgpt2",
        "text": "This is a simple bit of text"
    }
    response = client.post("/api/analyze", json=request)
    r = response.json()
    assert r['request']['text'] == request['text']
    # More assertions needed here