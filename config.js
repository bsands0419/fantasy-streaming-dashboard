// Deployed Cloudflare Worker endpoint for on-demand v4.2 refreshes.
window.V42_REFRESH_ENDPOINT = "https://fantasy-streaming-dashboard.bsands0419.workers.dev";

// Inject a simple, position-aware explanation of model accuracy and the case for streaming.
window.addEventListener('DOMContentLoaded',()=>{
  const controls=document.querySelector('.controls');
  const pos=document.getElementById('position');
  const updatedBadge=document.getElementById('updatedBadge');
  const header=document.querySelector('.header');
  if(!controls||!pos||!updatedBadge||!header)return;

  const style=document.createElement('style');
  style.textContent=`
    .header-tools{display:flex;align-items:center;gap:7px;margin-left:auto}
    #streamWhy{padding:5px 9px;border-radius:999px;font-size:11px;line-height:1.2;white-space:nowrap;background:#15213a;color:var(--text);border:1px solid var(--border)}
    #streamWhy:hover{filter:brightness(1.08)}
    .stream-explainer{display:none;margin-top:14px;padding:16px;border:1px solid var(--border);border-radius:13px;background:#0e172a;line-height:1.55}
    .stream-explainer.on{display:block}
    .stream-explainer h3{margin:0 0 8px;font-size:17px}
    .stream-explainer p{margin:8px 0;color:var(--muted);font-size:13px}
    .stream-explainer strong{color:var(--text)}
    .stream-stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;margin:12px 0}
    .stream-stat{padding:10px;border:1px solid var(--border);border-radius:10px;background:#111a2e}
    .stream-stat .k{display:block;color:var(--muted);font-size:11px;margin-bottom:3px}
    .stream-stat .v{font-size:17px;font-weight:750}
    .stream-caveat{font-size:11px!important;color:var(--muted)!important}
    @media(max-width:700px){.stream-stats{grid-template-columns:1fr}.header-tools{width:100%;justify-content:flex-end;margin-left:0}}
    @media(max-width:520px){#streamWhy{font-size:10px;padding:5px 8px}.header-tools{gap:5px}}
  `;
  document.head.appendChild(style);

  const btn=document.createElement('button');
  btn.id='streamWhy';
  btn.type='button';
  btn.textContent='Accuracy & Why Stream?';
  btn.setAttribute('aria-expanded','false');

  const tools=document.createElement('div');
  tools.className='header-tools';
  header.insertBefore(tools,updatedBadge);
  tools.appendChild(btn);
  tools.appendChild(updatedBadge);

  const panel=document.createElement('div');
  panel.id='streamExplainer';
  panel.className='stream-explainer';
  controls.parentNode.insertBefore(panel,document.getElementById('custom'));

  function content(){
    if(pos.value==='dst'){
      return `<h3>D/ST accuracy & why streaming works</h3>
        <p><strong>How accurate is the model?</strong> In leakage-safe walk-forward testing, v4.2 was typically off by about <strong>4.1 fantasy points</strong> per D/ST game. More important for streaming, its weekly ranking signal held up out of sample: about <strong>0.28</strong> in 2021–24 and <strong>0.32</strong> in 2025. The model's No. 1 weekly D/ST averaged about <strong>9.4 points</strong> in development testing and <strong>10.4 points</strong> in 2025.</p>
        <div class="stream-stats"><div class="stream-stat"><span class="k">Typical error</span><span class="v">~4.1 pts</span></div><div class="stream-stat"><span class="k">Weekly rank signal</span><span class="v">0.28–0.32</span></div><div class="stream-stat"><span class="k">No. 1 pick average</span><span class="v">~9.4–10.4</span></div></div>
        <p><strong>Why stream D/ST?</strong> Defense scoring is highly matchup-driven. Opponent sacks and turnovers, quarterback quality, betting spread and total, expected game script, and scoring environment can move a defense's weekly outlook substantially. That creates waiver-wire defenses with real one-week upside, even if they are not strong season-long units.</p>
        <p><strong>Best way to use this page:</strong> focus on projection plus P(Top 5), P(Top 10), and P(10+) rather than treating a tiny gap between two adjacent ranks as meaningful.</p>
        <p class="stream-caveat">Streaming is a strategy, not a guarantee. These tests support using weekly matchup information, but they do not prove that streaming will beat holding every elite season-long D/ST.</p>`;
    }
    return `<h3>Kicker accuracy & why streaming works</h3>
      <p><strong>How accurate is the model?</strong> Kicker scoring is noisy, but v4.2's bucket-scoring forecasts were typically off by about <strong>3.5–3.7 fantasy points</strong> per game. Its weekly ranking signal was about <strong>0.17</strong> in 2021–24 and improved to about <strong>0.22</strong> in 2025. The model's No. 1 bucket-scoring kicker averaged about <strong>9.4 points</strong> in development testing and <strong>10.4 points</strong> in 2025.</p>
      <div class="stream-stats"><div class="stream-stat"><span class="k">Typical error</span><span class="v">~3.5–3.7 pts</span></div><div class="stream-stat"><span class="k">Weekly rank signal</span><span class="v">0.17–0.22</span></div><div class="stream-stat"><span class="k">No. 1 pick average</span><span class="v">~9.4–10.4</span></div></div>
      <p><strong>Why stream kicker?</strong> Weekly opportunity depends heavily on team scoring environment, spread and total, expected field-goal attempts, weather, and job security. The position is also relatively flat, so there is often less reason to stay attached to a mediocre kicker when a better weekly setup is available.</p>
      <p><strong>Best way to use this page:</strong> treat the top group as a tier. Use projection and P(Top 5)/P(Top 10) to find strong opportunities, but do not overreact to very small differences between nearby kickers.</p>
      <p class="stream-caveat">Kicker is less predictable than D/ST. These results support matchup-based streaming, but they do not prove that streaming will beat holding every elite season-long kicker.</p>`;
  }

  function update(){panel.innerHTML=content();}
  btn.addEventListener('click',()=>{
    const open=!panel.classList.contains('on');
    panel.classList.toggle('on',open);
    btn.setAttribute('aria-expanded',String(open));
    btn.textContent=open?'Hide Guide':'Accuracy & Why Stream?';
    if(open)update();
  });
  pos.addEventListener('change',()=>{if(panel.classList.contains('on'))update();});
});
