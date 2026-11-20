import json
from pathlib import Path


def write_report(payloads, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payloads, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    path.write_text(TEMPLATE.replace("__DATA__", data), encoding="utf-8")


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kernel Lab · Benchmark</title>
<style>
:root{color-scheme:dark;font:15px system-ui;background:#10161e;color:#e5edf5}body{max-width:1200px;margin:36px auto;padding:0 24px}h1{font-size:30px;margin-bottom:8px}p{color:#a9bacd;line-height:1.6}.controls{display:flex;gap:14px;flex-wrap:wrap;margin:25px 0}label{display:grid;gap:7px;font-size:12px;color:#a9bacd}select{padding:9px;background:#1a2735;color:#e5edf5;border:1px solid #3a4c60;border-radius:6px}.card{background:#17212d;padding:20px;border-radius:12px;margin:20px 0;overflow:auto}.barrow{display:grid;grid-template-columns:255px 1fr 130px;gap:14px;align-items:center;margin:12px 0}.bar{height:20px;border-radius:4px;min-width:2px;background:#58c9b9}.bar.baseline{background:#6c91dc}.barlabel{font:12px ui-monospace,monospace;overflow-wrap:anywhere}.value{text-align:right;font:13px ui-monospace,monospace}table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:13px}td,th{text-align:right;padding:10px;border-bottom:1px solid #304052}td:first-child,th:first-child{text-align:left}.pass{color:#58c9b9}.failed,.unimplemented{color:#ffb078}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}details{margin:18px 0}#empty{color:#ffb078}@media(max-width:700px){.barrow{grid-template-columns:150px 1fr 90px}body{padding:0 12px}.card{padding:12px}}
</style>
<h1>Kernel Lab <span style="color:#58c9b9">/ Benchmark</span></h1>
<p>先验证正确性，再看性能。蓝色为 PyTorch baseline，绿色为通过校验的练习实现。未实现和失败项不会显示性能柱。</p>
<div class="controls">
<label>实验 / operation<select id="op"></select></label>
<label>数据类型<select id="dtype"></select></label>
<label>测试形状<select id="shape"></select></label>
<label>运行记录<select id="run"></select></label>
<label>指标<select id="metric"><option value="median_us">延迟 μs（越低越好）</option><option value="speedup">加速比（越高越好）</option><option value="gb_s">逻辑吞吐 GB/s（越高越好）</option></select></label>
</div>
<div class="card"><div id="chart"></div><div id="empty"></div></div>
<p>加速比 = 同次运行、同形状、同 dtype 的 PyTorch 中位延迟 / 当前中位延迟。GB/s 由逻辑输入输出字节估算，受缓存影响，不是实测显存带宽。不同设备或计时配置的结果应分开比较。</p>
<div class="card"><table><thead><tr><th>run / 实现</th><th>形状</th><th>dtype</th><th>状态</th><th>μs 中位数</th><th>P25–P75 μs</th><th>加速比</th><th>GB/s</th></tr></thead><tbody id="table"></tbody></table></div>
<details><summary>环境、计时配置与源文件指纹</summary><pre id="metadata"></pre></details>
<details><summary>未实现 / 校验失败详情</summary><pre id="errors"></pre></details>
<script type="application/json" id="data">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent);
const $=id=>document.getElementById(id);
const rows=data.flatMap((d,i)=>d.rows.map(r=>({...r,run:i,tag:d.tag,device:d.environment.gpu||d.environment.device})));
function fill(id,values){const el=$(id);el.replaceChildren();for(const [value,label] of values){const o=document.createElement('option');o.value=value;o.textContent=label;el.append(o)}}
const unique=key=>[...new Set(rows.map(r=>r[key]))];
fill('op',unique('op').map(x=>[x,x]));fill('dtype',unique('dtype').map(x=>[x,x]));
fill('run',[['all','全部运行'],...data.map((d,i)=>[i,`${d.tag} · ${d.created_at} · ${d.environment.gpu||d.environment.device}`])]);
function shapes(){fill('shape',[['all','全部形状'],...[...new Set(rows.filter(r=>r.op===$('op').value).map(r=>r.case))].map(x=>[x,x])])}
function format(n,d=3){return Number.isFinite(n)?n.toFixed(d):'—'}
function draw(){
 const selected=rows.filter(r=>r.op===$('op').value&&r.dtype===$('dtype').value&&($('shape').value==='all'||r.case===$('shape').value)&&($('run').value==='all'||r.run===Number($('run').value)));
 const metric=$('metric').value, timed=selected.filter(r=>Number.isFinite(r[metric])&&['pass','baseline'].includes(r.status));
 const max=Math.max(...timed.map(r=>r[metric]),0.0001);$('chart').replaceChildren();$('table').replaceChildren();
 $('empty').textContent=timed.length?'':'没有已测量结果。先实现 kernel 并通过校验，或用 --baseline-only 查看 PyTorch。';
 for(const r of timed){const row=document.createElement('div');row.className='barrow';const label=document.createElement('div');label.className='barlabel';label.textContent=`${r.case} / ${r.implementation} / ${r.tag} #${r.run+1}`;const track=document.createElement('div');const bar=document.createElement('div');bar.className='bar '+(r.status==='baseline'?'baseline':'');bar.style.width=`${100*r[metric]/max}%`;track.append(bar);const value=document.createElement('div');value.className='value';value.textContent=format(r[metric])+(metric==='speedup'?'x':'');row.append(label,track,value);$('chart').append(row)}
 for(const r of selected){const tr=document.createElement('tr');for(const value of [`${r.tag} #${r.run+1} / ${r.implementation}`,r.case,r.dtype,r.status,format(r.median_us),`${format(r.p25_us)}–${format(r.p75_us)}`,format(r.speedup,2)+'x',format(r.gb_s,2)]){const td=document.createElement('td');td.textContent=value;tr.append(td)}tr.children[3].className=r.status;$('table').append(tr)}
 $('metadata').textContent=JSON.stringify(data.map(d=>({tag:d.tag,created_at:d.created_at,environment:d.environment,config:d.config,sources:d.rows.filter(r=>r.source_hash).map(r=>({implementation:r.implementation,source_hash:r.source_hash}))})),null,2);
 $('errors').textContent=selected.filter(r=>r.error).map(r=>`${r.case} / ${r.dtype} / ${r.implementation}: ${r.error}`).join('\n\n')||'无';
}
$('op').onchange=()=>{shapes();draw()};for(const id of ['dtype','shape','run','metric'])$(id).onchange=draw;
shapes();draw();
</script></html>'''
