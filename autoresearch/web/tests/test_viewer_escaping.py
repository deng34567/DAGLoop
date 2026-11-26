"""Run the shipped viewer JavaScript with an in-memory DOM and no network."""
from __future__ import annotations

import json
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest


class _Markup(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.tags = []
        self.handlers = []
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.handlers.extend(value for key, value in attrs if key == "onclick")


def _node(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to execute viewer JavaScript")
    result = subprocess.run([node, "-"], input=script, text=True, capture_output=True,
                            timeout=10, check=True)
    return json.loads(result.stdout)


def _render(scenario):
    html = (Path(__file__).parents[1] / "static" / "index.html").read_text()
    source = html.split("<script>", 1)[1].split("</script>", 1)[0]
    source = source.split("\napplyLang();", 1)[0]  # skip page startup and its fetches
    harness = """
const elements = new Map();
function element(){ return {style:{}, value:"", children:[], innerHTML:"",
  addEventListener(){}, appendChild(child){this.children.push(child);}}; }
const document = {createElement:element, getElementById(id){
  if(!elements.has(id)) elements.set(id,element()); return elements.get(id); }};
const localStorage = {getItem(){return null;}};
let response = {}, requests = [];
async function fetch(url, options){ requests.push({url,options});
  return {ok:true, async json(){return response;}}; }
"""
    return _node(harness + source + "\n(async()=>{\n" + scenario + "\n})();")


def _assert_safe_handler(markup, callback, argument):
    parsed = _Markup(markup)
    assert len(parsed.handlers) == 1
    result = _node("const vm=require('node:vm'); let calls=[], alerts=[];"
                   + "vm.runInNewContext(" + json.dumps(parsed.handlers[0]) + ",{"
                   + callback + ":(...args)=>calls.push(args),alert:x=>alerts.push(x)});"
                   + "console.log(JSON.stringify({calls,alerts}));")
    assert result == {"calls": [[argument]], "alerts": []}


def test_failed_event_renders_error_and_emitter_as_text():
    markup = _render('''
_mode="live";
addEv({event_type:"agent.failed",t:1,emitter:'<img src=x onerror=alert(1)>',
  payload:{underlying_reason:'<img src=x onerror=alert(2)>'}});
console.log(JSON.stringify(document.getElementById("evs").children[0].innerHTML));
''')
    assert _Markup(markup).tags == ["span", "span", "span"]
    assert "&lt;img" in markup


def test_repo_candidate_click_preserves_url_without_executing_it():
    url = "https://example.org/repo');alert(1);//"
    markup = _render('''
_mode="live"; sid="test-session";
response={session_id:sid,current_state:"HumanIdeaCheckpoint",repo_candidates:[
  {url:''' + json.dumps(url) + ''',name:"candidate"}]};
await refreshState();
console.log(JSON.stringify(document.getElementById("chat").children[0].innerHTML));
''')
    # The prompt also contains a static Continue button.
    candidate = markup.split('<button onclick=', 1)[1].split('</button>', 1)[0]
    _assert_safe_handler('<button onclick=' + candidate + '</button>', "sendCheckpointRepo", url)


def test_artifact_reference_click_preserves_value_without_executing_it():
    ref = "artifact');alert(1);//"
    markup = _render("renderArts({label:" + json.dumps(ref) + "});"
                     'console.log(JSON.stringify(document.getElementById("arts").innerHTML));')
    _assert_safe_handler(markup, "viewArt", ref)


def test_composer_renders_text_and_posts_original_answer():
    answer = '<img src=x onerror=alert(1)>'
    result = _render('''
curQ={question_id:"q1"};
document.getElementById("composer").value=''' + json.dumps(answer) + ''';
send();
console.log(JSON.stringify({html:document.getElementById("chat").children[0].innerHTML,
  answer:JSON.parse(requests[0].options.body).answer_text}));
''')
    assert _Markup(result["html"]).tags == []
    assert result["answer"] == answer
