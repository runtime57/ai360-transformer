from pathlib import Path
import json
import statistics

OUT = Path(__file__).resolve().parent
baseline = json.loads((OUT / "stories.json").read_text())
gated_path = OUT / "rope_elementwise_stories.json"
gated = json.loads(gated_path.read_text()) if gated_path.exists() else None
selected = [
    ("The data center sneezes", "Студенты ИИ360 и чихающий датацентр Яндекса"),
    ("The shoes that did not want to walk", "Ботинки больше не хотят работать"),
    ("The tiny king in a teapot", "Король в чайнике"),
    ("The secret of the lost sock", "Дверь в носке"),
    ("The cloud in the kitchen", "Котлетка и облако"),
]
def section(sample, title=None):
    seed = sample.get("seed", baseline["effective_seed"])
    fence = chr(96) * 3
    return "## " + (title or sample["title"]) + "\n\nSeed: " + str(seed) + "\n\n**Prompt:**\n\n" + fence + "text\n" + sample["prompt"] + "\n" + fence + "\n\n**Полный текст:**\n\n" + sample["text"] + "\n"
def write_markdown(data, filename, title, subset=None):
    samples = data["samples"]
    if subset:
        sections = [section(next(s for s in samples if s["title"] == key), name) for key, name in subset if any(s["title"] == key for s in samples)]
    else:
        sections = [section(s) for s in samples]
    settings = data["settings"]
    header = "# " + title + "\n\nBeam width " + str(settings["beam_width"]) + ", temperature " + str(settings["beam_temperature"]) + ", top_p " + str(settings["top_p"]) + ", контекст " + str(settings["context_len"]) + ", лимит " + str(settings["max_new_tokens"]) + " новых токенов. Тексты без исправлений.\n\n"
    (OUT / filename).write_text(header + "\n".join(sections))
write_markdown(baseline, "stories.md", "TinyStories — baseline")
write_markdown(baseline, "selected.md", "TinyStories — подборка baseline", selected)
if gated:
    write_markdown(gated, "rope_elementwise_stories.md", "TinyStories — RoPE + elementwise gate")
    write_markdown(gated, "rope_elementwise_selected.md", "TinyStories — подборка RoPE + elementwise gate", selected)
if gated and len(gated["samples"]) == len(baseline["samples"]):
    assert gated["settings"] == baseline["settings"]
    assert gated["device"] == baseline["device"]
    assert gated["gpt_source_sha256"] == baseline["gpt_source_sha256"]
    assert gated["tokenizer_sha256"] == baseline["tokenizer_sha256"]
    pairs = []
    for a, b in zip(baseline["samples"], gated["samples"]):
        seed = a.get("seed", baseline["effective_seed"])
        assert (a["title"], a["prompt"], seed) == (b["title"], b["prompt"], b["seed"])
        assert a["decoded_prompt"] == b["decoded_prompt"]
        assert a["text"].startswith(a["decoded_prompt"])
        assert b["text"].startswith(b["decoded_prompt"])
        pairs.append({
            "title": a["title"], "prompt": a["prompt"], "seed": seed,
            "baseline": {**a, "continuation": a["text"][len(a["decoded_prompt"]):]},
            "gated": {**b, "continuation": b["text"][len(b["decoded_prompt"]):]},
        })
    summary = {}
    for key, data in (("baseline", baseline), ("gated", gated)):
        samples = data["samples"]
        summary[key] = {
            "samples": len(samples),
            "eos_count": sum(s["ended_with_eos"] for s in samples),
            "cutoff_count": sum(not s["ended_with_eos"] for s in samples),
            "mean_new_tokens": statistics.mean(s["new_tokens"] for s in samples),
            "median_new_tokens": statistics.median(s["new_tokens"] for s in samples),
        }
    comparison = {
        "settings": baseline["settings"], "device": baseline["device"],
        "inference_dtype": gated["inference_dtype"],
        "checkpoint_epochs": {"baseline": baseline["checkpoint_epoch"], "gated": gated["checkpoint_epoch"]},
        "checkpoint_sha256": {"baseline": gated["baseline_checkpoint_sha256"], "gated": gated["checkpoint_sha256"]},
        "gpt_source_sha256": gated["gpt_source_sha256"], "tokenizer_sha256": gated["tokenizer_sha256"],
        "summary": summary, "pairs": pairs,
    }
    (OUT / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2))
    payload = json.dumps(comparison, ensure_ascii=False).replace("<", "\\u003c")
    document = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TinyStories — baseline и RoPE + elementwise gate</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#243047;font:16px/1.6 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:32px}h1{font-size:30px;line-height:1.25;margin:0 0 12px}p{margin:8px 0}.muted{color:#627087}.toolbar{display:flex;gap:16px;align-items:end;flex-wrap:wrap;margin:24px 0}label{display:flex;flex-direction:column;font-size:13px;font-weight:600;gap:6px}select,button{font:inherit;color:#243047;background:white;border:1px solid #cbd4e1;border-radius:8px;padding:10px 14px}button{cursor:pointer}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}.card{background:white;border:1px solid #dbe2ec;border-top:4px solid;border-radius:12px;padding:24px}.card h2{font-size:20px;margin:0 0 10px}.meta{font-size:13px;color:#627087;margin-bottom:18px}.prompt{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf2f8;border-radius:8px;padding:16px;font:14px/1.6 ui-monospace,monospace;margin:8px 0 20px}.story{white-space:pre-wrap;overflow-wrap:anywhere;font:18px/1.7 Georgia,serif;margin-top:10px}.footer{margin-top:26px;font-size:13px;color:#627087}.hint{background:#eaf0f8;border-radius:10px;padding:14px 18px;font-size:14px}@media(max-width:800px){main{padding:18px}.cards{grid-template-columns:1fr}h1{font-size:25px}select{max-width:100%}}
</style>
</head>
<body>
<main>
<h1>TinyStories: baseline и RoPE + elementwise gate</h1>
<p class="muted">18 пар с одинаковыми prompt, seed и настройками генерации. Перед каждым продолжением показан заданный prompt. Тексты без исправлений.</p>
<div class="toolbar">
<label>История<select id="sampleSelect"></select></label>
<button id="blindButton" type="button">Скрыть названия моделей</button>
</div>
<div class="cards" id="cards"></div>
<p class="hint">Сравни сохранение персонажей и условий затравки, развитие сюжета, противоречия, повторы и завершение. В скрытом режиме выбери лучший вариант до раскрытия названий.</p>
<div class="footer" id="settings"></div>
<div class="footer" id="summary"></div>
</main>
<script>
const data = __DATA__;
const names = {baseline:'Baseline',gated:'RoPE + elementwise gate'};
const colors = {baseline:'#3366CC',gated:'#9467BD'};
const select = document.getElementById('sampleSelect');
let order=['baseline','gated'];
let blind=false;
data.pairs.forEach((pair,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+'. '+pair.title+' · seed '+pair.seed;select.append(o)});
select.value='12';
function render(){
 const pair=data.pairs[Number(select.value)];
 const cards=document.getElementById('cards');cards.replaceChildren();
 order.forEach((key,i)=>{
  const sample=pair[key];
  const card=document.createElement('article');card.className='card';card.style.borderTopColor=blind?'#778399':colors[key];
  const title=document.createElement('h2');title.textContent=blind?'Модель '+String.fromCharCode(65+i):names[key];
  const meta=document.createElement('div');meta.className='meta';meta.textContent='seed '+pair.seed+' · '+sample.new_tokens+' новых токенов · '+(sample.ended_with_eos?'EOS: модель завершила историю':'Достигнут лимит');
  const promptLabel=document.createElement('strong');promptLabel.textContent='Prompt';
  const prompt=document.createElement('div');prompt.className='prompt';prompt.textContent=pair.prompt;
  const storyLabel=document.createElement('strong');storyLabel.textContent='Продолжение модели';
  const story=document.createElement('div');story.className='story';story.textContent=sample.continuation;
  card.append(title,meta,promptLabel,prompt,storyLabel,story);cards.append(card);
 });
 document.getElementById('blindButton').textContent=blind?'Раскрыть названия моделей':'Скрыть названия моделей';
 const s=data.settings;
 document.getElementById('settings').textContent='GPT.predict, beam sampling: width '+s.beam_width+', temperature '+s.beam_temperature+', top_p '+s.top_p+'. Контекст '+s.context_len+', лимит '+s.max_new_tokens+' новых токенов. '+data.device.toUpperCase()+', '+data.inference_dtype.toUpperCase()+', dropout выключен. Оба чекпоинта: эпоха 10.';
 document.getElementById('summary').textContent=blind?'':Object.entries(data.summary).map(([key,x])=>names[key]+': EOS '+x.eos_count+'/'+x.samples+', по лимиту '+x.cutoff_count).join(' · ');
}
select.addEventListener('change',render);
document.getElementById('blindButton').addEventListener('click',()=>{
 blind=!blind;
 order=blind&&Math.random()<0.5?['gated','baseline']:['baseline','gated'];
 render();
});
render();
</script>
</body>
</html>
"""
    (OUT / "comparison.html").write_text(document.replace("__DATA__", payload))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
print("Markdown reports updated with prompts.", flush=True)
