// app.js — improved UI behavior, typing animation, colorful loader, welcome screen
const $ = id => document.getElementById(id);
const messagesEl = $('messages');
const form = $('composer');
const input = $('input');
const sendBtn = $('send');
const topkEl = $('topk');
const clearBtn = $('clear');
const enterBtn = $('enterBtn');
const welcome = $('welcome');

const API_URL = "http://127.0.0.1:8000/ask"; // change if needed

// Welcome flow
if (enterBtn) {
  enterBtn.addEventListener('click', () => {
    welcome.style.transition = 'opacity 300ms ease, transform 300ms ease';
    welcome.style.opacity = '0';
    welcome.style.transform = 'translateY(-10px)';
    setTimeout(() => welcome.remove(), 320);
    input.focus();
  });
}

// Helpers
function scrollBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function createMessageElement(role) {
  const row = document.createElement('div');
  row.className = 'msg-row';
  const bubble = document.createElement('div');
  bubble.className = 'msg ' + (role === 'user' ? 'user' : 'assistant');
  const body = document.createElement('div');
  body.className = 'body';
  bubble.appendChild(body);
  row.appendChild(bubble);
  return { row, bubble, body };
}

function appendUserMessage(text) {
  const { row, bubble, body } = createMessageElement('user');
  // user messages stay plain text
  body.textContent = text;
  messagesEl.appendChild(row);
  scrollBottom();
}

function appendAssistantPlaceholder() {
  const { row, bubble, body } = createMessageElement('assistant');
  const loaderWrap = document.createElement('div');
  loaderWrap.style.display = 'flex';
  loaderWrap.style.alignItems = 'center';
  loaderWrap.style.gap = '10px';

  const loader = document.createElement('span');
  loader.className = 'loader';
  const hint = document.createElement('div');
  hint.className = 'muted';
  hint.style.color = 'var(--muted)';
  hint.style.fontSize = '13px';
  hint.textContent = 'RoboChat is thinking…';

  loaderWrap.appendChild(loader);
  loaderWrap.appendChild(hint);

  body.appendChild(loaderWrap);
  messagesEl.appendChild(row);
  scrollBottom();
  return { row, bubble, body, loaderWrap };
}

// Typing effect with Markdown rendering
async function typeText(targetEl, text, speed = 18) {
  targetEl.innerHTML = ''; // clear placeholder
  const plainText = text;
  let rendered = "";

  for (let i = 0; i < plainText.length; i++) {
    rendered += plainText[i];
    if (i % 5 === 0 || i === plainText.length - 1) {
      // progressively render markdown
      targetEl.innerHTML = marked.parse(rendered);
      await new Promise(r => setTimeout(r, speed));
      scrollBottom();
    }
  }

  targetEl.classList.add('typing');
  setTimeout(() => targetEl.classList.remove('typing'), 800);
  scrollBottom();
}

// Send request to backend
async function askServer(question, top_k) {
  const payload = { question, top_k: Number(top_k) };
  const resp = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(txt || 'Server error');
  }
  return resp.json();
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  appendUserMessage(text);
  input.value = '';
  setSending(true);

  // add assistant placeholder with loader
  const { row, bubble, body, loaderWrap } = appendAssistantPlaceholder();

  try {
    const topk = topkEl.value;
    const data = await askServer(text, topk);
    // remove loader
    body.removeChild(loaderWrap);

    const answer = data.answer || data.response || 'No answer';
    const context = data.context || data.retrieved_context || null;

    // typing animation with Markdown support
    await typeText(body, answer, 14);

    // Retrieved context toggle with Markdown rendering
    if (context && Array.isArray(context) && context.length) {
      const ctxWrapper = document.createElement("div");
      ctxWrapper.className = "context-wrapper";

      ctxWrapper.innerHTML = `
        <div class="toggle-header" style="cursor:pointer; display:flex; align-items:center; gap:6px; font-size:13px; color:var(--muted); margin-top:8px; padding-top:8px; border-top:1px dashed rgba(255,255,255,0.08);">
          <span>Retrieved context</span>
          <span class="arrow">▼</span>
        </div>
        <div class="context-list" style="display:none; font-size:13px; color:var(--muted); white-space:pre-wrap;">
          ${context.map(c => marked.parse(c)).join("<hr/>")}
        </div>
      `;

      const header = ctxWrapper.querySelector(".toggle-header");
      const list = ctxWrapper.querySelector(".context-list");
      const arrow = ctxWrapper.querySelector(".arrow");

      header.addEventListener("click", () => {
        if (list.style.display === "none") {
          list.style.display = "block";
          arrow.textContent = "▲";
        } else {
          list.style.display = "none";
          arrow.textContent = "▼";
        }
      });

      body.appendChild(ctxWrapper);
    }

  } catch (err) {
    body.textContent = `Error: ${err.message}`;
  } finally {
    setSending(false);
    scrollBottom();
  }
});

clearBtn.addEventListener('click', () => {
  messagesEl.innerHTML = '';
  input.focus();
});

function setSending(isSending) {
  sendBtn.disabled = isSending;
  if (isSending) {
    sendBtn.innerHTML = '<span class="loader" style="width:18px;height:18px"></span>';
  } else {
    sendBtn.innerHTML = 'Send';
  }
}

// auto-resize textarea
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(180, input.scrollHeight) + 'px';
});

input.focus();
