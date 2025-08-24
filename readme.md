# 🤖 RoboChat  

RoboChat is an AI-powered assistant designed for **electronics, robotics, and IoT learners**.  
It combines **local knowledge retrieval**, **web search integration**, and **Hugging Face inference models** to provide:  

- Technical explanations for microcontrollers, sensors, and robotics modules.  
- Real-time store lookup and pricing information for components.  
- Interactive tutor mode for step-by-step project guidance.  
- Full deployment with **backend + frontend**, making it usable as a standalone web app.  

---

## 🚀 Features
- **LLM Integration**: Powered by Hugging Face Inference (`openai/gpt-oss-120b:fireworks-ai`).  
- **Graph-based Tool Use**: Runs reasoning pipelines with retrieval + web tools.  
- **Frontend**: Simple, lightweight interface (HTML, CSS, JS only, no frameworks).  
- **Backend**: Python FastAPI backend to handle inference calls.  
- **Deployment Ready**: Works on local setup or can be hosted on cloud platforms.  

---

## 📂 Project Structure
```txt
RoboChat/
    │── backend/ # Agent, Tools, FastAPI backend, connects to Hugging Face inference
    │── frontend/ # Simple HTML/CSS/JS interface
    │── notebook.ipynb # Jupyter notebooks for experiments & demos
    │── README.md # Project documentation
```

---

## 🛠️ Setup & Installation  

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Seif-Sameh/RoboChat.git
cd robochat
```

### 2️⃣ Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

### 3️⃣ Frontend Setup
Frontend is a simple static app (no frameworks).
Open frontend/index.html in a browser, or serve via a local web server:
```bash
cd frontend
python -m http.server 8000
```
Now visit:
http://localhost:8000

---

## 📒 Notebooks
Inside notebooks/ you’ll find:
- robochat_demo.ipynb → Example of querying RoboChat with LangChain graph.
- Cells showing local Q&A and web-assisted queries.

---

## 🧪 Example Usage
Inside Jupyter notebooks, you can test RoboChat’s reasoning:
```python
result = bot.graph.invoke({"messages": [HumanMessage(content="What is Arduino UNO?")]})
print(result['messages'][-1].content)
```

---

## 📦 Deployment
This project is designed for full-stack deployment:
- Backend → FastAPI with Hugging Face inference.
- Frontend → Simple HTML/CSS/JS UI for interaction.
- Optional Hosting → Deploy on platforms like Vercel (frontend) + Render/Heroku (backend).

---

## 🔮 Future Plans
- Expand knowledge base for advanced robotics topics.
- Multi-modal support (images + circuit diagrams).

---

## 💡 Credits
- Hugging Face Inference API
- LangChain for orchestration
- OpenAI OSS Models (gpt-oss-20b, gpt-oss-120b)
