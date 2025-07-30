# main.py
# Internship Master Agent - Final Version
# All core features from the project plan are now implemented.

# --- 1. Imports ---
import uvicorn
import sqlite3
import os
import json
import docker
import tempfile
from fastapi import FastAPI, Form, Response, UploadFile, File, HTTPException
from pydantic.v1 import BaseModel, Field
from typing import Dict, Any, List, TypedDict, Optional
from dotenv import load_dotenv
import datetime
import requests

# --- Automation & Twilio Imports ---
from apscheduler.schedulers.background import BackgroundScheduler
from twilio.rest import Client

# --- LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic.v1 import BaseModel as LangChainBaseModel, Field as LangChainField

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END

# --- 2. Configuration & Initialization ---
load_dotenv()

# Twilio, OpenAI, DB Config...
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_NUMBER]):
    raise ValueError("Twilio credentials not found in .env file.")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file.")

DATABASE_NAME = "ima_database.db"
app = FastAPI(
    title="Internship Master Agent API",
    description="Final Version: All core features implemented.",
    version="1.0.0"
)
scheduler = BackgroundScheduler(timezone="Asia/Karachi")
try:
    docker_client = docker.from_env()
except docker.errors.DockerException:
    print("ERROR: Docker is not running. Please start Docker to enable code evaluation.")
    docker_client = None

# --- 3. Database Management ---
def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def setup_database():
    print("Setting up database...")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT, phone_number TEXT NOT NULL UNIQUE, name TEXT,
        education_level TEXT, skills TEXT, goals TEXT, availability TEXT,
        onboarding_state TEXT DEFAULT 'awaiting_introduction'
    )''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS roadmaps (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, roadmap_json TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_progress (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, current_day INTEGER DEFAULT 1,
        last_task_sent_at DATETIME, FOREIGN KEY (user_id) REFERENCES users (id)
    )''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS submissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        day INTEGER NOT NULL,
        submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        code TEXT,
        feedback TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''')
    conn.commit()
    conn.close()
    print("Database setup complete.")

# --- 4. Helper functions for AI and Code Evaluation ---

class DailyTask(LangChainBaseModel):
    day: int
    topic: str
    task: str
    resources: List[str]

class Roadmap(LangChainBaseModel):
    title: str
    duration_weeks: int
    daily_tasks: List[DailyTask]

def generate_dynamic_roadmap(user_info: Dict[str, Any]) -> str:
    print("Generating dynamic roadmap for user...")
    model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o", temperature=0.7)
    parser = JsonOutputParser(pydantic_object=Roadmap)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a world-class AI mentor. Generate a detailed, day-by-day internship plan as a JSON object based on the user's profile. Follow the provided format instructions precisely. Respond only with the raw JSON."),
        ("human", "Generate a plan for user: {name}, {education}, {skills}, {goals}, {availability}. Instructions: {format_instructions}")
    ])
    chain = prompt | model | parser
    try:
        generated_roadmap = chain.invoke({
            "name": user_info['name'], "education": user_info['education_level'],
            "skills": user_info['skills'], "goals": user_info['goals'],
            "availability": user_info['availability'], "format_instructions": parser.get_format_instructions(),
        })
        return json.dumps(generated_roadmap, indent=2)
    except Exception as e:
        print(f"Error generating roadmap: {e}")
        return json.dumps({"error": "Could not generate roadmap."})

def handle_general_conversation(user_question: str, roadmap_json: str) -> str:
    print(f"Handling general question: '{user_question}'")
    model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o", temperature=0.3)
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI mentor. Your job is to answer questions based ONLY on the provided roadmap context. If the user's question is not related to the roadmap, you MUST politely state that you can only answer questions about their internship plan."),
        ("human", "My question: '{question}'. My roadmap: {roadmap_context}")
    ])
    chain = prompt | model | parser
    try:
        return chain.invoke({"question": user_question, "roadmap_context": roadmap_json})
    except Exception as e:
        print(f"Error in QA chain: {e}")
        return "I'm sorry, I'm having a little trouble thinking right now."

def evaluate_code_with_ai(code: str, output: str, error: str, task_description: str) -> str:
    print("---AI FEEDBACK on Code---")
    model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o", temperature=0.5)
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert programming mentor. The user has submitted Python code for a specific task. Provide constructive, friendly feedback. Analyze their code, the output, and any errors. If correct, praise them and suggest one professional improvement. If there are errors, explain the error simply and give a hint to fix it without giving the full answer."),
        ("human", "Task: {task}\n\nCode: {code}\n\nOutput: {output}\n\nErrors: {error}\n\nPlease give me feedback.")
    ])
    chain = prompt | model | parser
    try:
        return chain.invoke({"task": task_description, "code": code, "output": output, "error": error or "None"})
    except Exception as e:
        print(f"Error in AI feedback chain: {e}")
        return "I'm sorry, I had trouble analyzing your code."

def run_code_in_docker(code_content: str) -> (str, str):
    if not docker_client: return "", "Docker is not running on the server."
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_code_file:
        tmp_code_file.write(code_content)
        code_filepath = tmp_code_file.name
    container = None
    try:
        container = docker_client.containers.run("python:3.9", command=["python", "-u", f"/tmp/{os.path.basename(code_filepath)}"],
            volumes={os.path.dirname(code_filepath): {'bind': '/tmp', 'mode': 'ro'}}, remove=True, detach=True, mem_limit="128m", cpu_shares=1)
        container.wait(timeout=10)
        stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
        stderr = container.logs(stdout=False, stderr=True).decode('utf-8')
        return stdout, stderr
    except Exception as e:
        if container:
            try: container.stop()
            except docker.errors.APIError: pass
        return "", f"An execution error occurred: {str(e)}"
    finally:
        os.unlink(code_filepath)

# --- 5. LangGraph State, Nodes, and Edges ---

class GraphState(TypedDict):
    user_id: int
    phone_number: str
    message_body: str
    file_content: Optional[str]
    response: str
    db_connection: sqlite3.Connection

def onboarding_node(state: GraphState) -> GraphState:
    print("---NODE: ONBOARDING---")
    conn = state['db_connection']
    cursor = conn.cursor()
    user_id = state['user_id']
    message_body = state['message_body']
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = dict(cursor.fetchone())
    onboarding_state = user['onboarding_state']
    response_message = "An unknown error occurred."
    if onboarding_state == 'awaiting_introduction':
        response_message = "Welcome! What is your full name?"
        cursor.execute("UPDATE users SET onboarding_state = 'awaiting_name' WHERE id = ?", (user_id,))
    elif onboarding_state == 'awaiting_name':
        cursor.execute("UPDATE users SET name = ?, onboarding_state = 'awaiting_education' WHERE id = ?", (message_body, user_id))
        response_message = f"Great, {message_body}! What is your education level?"
    elif onboarding_state == 'awaiting_education':
        cursor.execute("UPDATE users SET education_level = ?, onboarding_state = 'awaiting_skills' WHERE id = ?", (message_body, user_id))
        response_message = "Got it. What are your skills?"
    elif onboarding_state == 'awaiting_skills':
        cursor.execute("UPDATE users SET skills = ?, onboarding_state = 'awaiting_goals' WHERE id = ?", (message_body, user_id))
        response_message = "Thanks. What are your goals?"
    elif onboarding_state == 'awaiting_goals':
        cursor.execute("UPDATE users SET goals = ?, onboarding_state = 'awaiting_availability' WHERE id = ?", (message_body, user_id))
        response_message = "Excellent. How many hours per day?"
    elif onboarding_state == 'awaiting_availability':
        cursor.execute("UPDATE users SET availability = ? WHERE id = ?", (message_body, user_id))
        conn.commit()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        updated_user_info = dict(cursor.fetchone())
        response_message = "Generating your plan..."
        roadmap_json = generate_dynamic_roadmap(updated_user_info)
        roadmap_data = json.loads(roadmap_json)
        if "error" in roadmap_data:
            response_message = "Error creating plan. Please send your last message again to retry."
        else:
            cursor.execute("UPDATE users SET onboarding_state = 'completed' WHERE id = ?", (user_id,))
            cursor.execute("INSERT INTO roadmaps (user_id, roadmap_json) VALUES (?, ?)", (user_id, roadmap_json))
            cursor.execute("INSERT INTO user_progress (user_id) VALUES (?)", (user_id,))
            response_message = f"Your plan, '{roadmap_data.get('title', 'your internship')}', is ready!"
    conn.commit()
    state['response'] = response_message
    return state

def qa_node(state: GraphState) -> GraphState:
    print("---NODE: QA---")
    conn = state['db_connection']
    cursor = conn.cursor()
    user_id = state['user_id']
    message_body = state['message_body']
    if message_body.lower().strip() == 'reset my progress':
        print(f"---RESETTING PROGRESS for user_id: {user_id}---")
        cursor.execute("DELETE FROM user_progress WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM roadmaps WHERE user_id = ?", (user_id,))
        cursor.execute("UPDATE users SET onboarding_state = 'awaiting_introduction', name = NULL, education_level = NULL, skills = NULL, goals = NULL, availability = NULL WHERE id = ?", (user_id,))
        conn.commit()
        state['response'] = "Progress reset. Send any message to begin again."
        return state
    cursor.execute("SELECT roadmap_json FROM roadmaps WHERE user_id = ?", (user_id,))
    roadmap_row = cursor.fetchone()
    if not roadmap_row or not roadmap_row['roadmap_json']:
        state['response'] = "Can't find your roadmap."
        return state
    state['response'] = handle_general_conversation(message_body, roadmap_row['roadmap_json'])
    return state

def code_eval_node(state: GraphState) -> GraphState:
    print("---NODE: CODE EVALUATION---")
    conn = state['db_connection']
    cursor = conn.cursor()
    user_id = state['user_id']
    code_content = state['file_content']
    if not code_content:
        state['response'] = "File problem. Please try again."
        return state
    cursor.execute("SELECT p.current_day, r.roadmap_json FROM user_progress p JOIN roadmaps r ON p.user_id = r.user_id WHERE p.user_id = ?", (user_id,))
    progress_info = cursor.fetchone()
    task_description = "General task"
    day_of_task = 0
    if progress_info:
        day_of_task = progress_info['current_day'] - 1
        roadmap = json.loads(progress_info['roadmap_json'])
        task = next((t for t in roadmap['daily_tasks'] if t['day'] == day_of_task), None)
        if task:
            task_description = f"Topic: {task['topic']}\nTask: {task['task']}"
    output, error = run_code_in_docker(code_content)
    feedback = evaluate_code_with_ai(code_content, output, error, task_description)
    if day_of_task > 0:
        cursor.execute("INSERT INTO submissions (user_id, day, code, feedback) VALUES (?, ?, ?, ?)", (user_id, day_of_task, code_content, feedback))
        conn.commit()
        print(f"Submission for Day {day_of_task} by user {user_id} has been recorded.")
    state['response'] = feedback
    return state

def final_report_node(state: GraphState) -> GraphState:
    print("---NODE: FINAL REPORT---")
    conn = state['db_connection']
    cursor = conn.cursor()
    user_id = state['user_id']
    cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
    user_name = cursor.fetchone()['name']
    cursor.execute("SELECT COUNT(*) as completed_tasks FROM submissions WHERE user_id = ?", (user_id,))
    task_count = cursor.fetchone()['completed_tasks']
    report_message = (
        f"🎉 Congratulations, {user_name}! 🎉\n\n"
        f"You have completed the Internship Master Agent program.\n\n"
        f"Here's a summary of your performance:\n"
        f"- Total Tasks Completed: {task_count}\n\n"
        f"This is a significant achievement. A formal certificate will be generated and sent to you shortly.\n\n"
        f"Best of luck in your future endeavors!"
    )
    state['response'] = report_message
    return state

def route_logic(state: GraphState) -> str:
    print("---EDGE: ROUTING---")
    if state.get('file_content'):
        return "code_eval"
    conn = state['db_connection']
    cursor = conn.cursor()
    user_id = state['user_id']
    cursor.execute("SELECT p.current_day, r.roadmap_json FROM user_progress p JOIN roadmaps r ON p.user_id = r.user_id WHERE p.user_id = ?", (user_id,))
    progress_info = cursor.fetchone()
    if progress_info:
        roadmap = json.loads(progress_info['roadmap_json'])
        total_tasks = len(roadmap.get('daily_tasks', []))
        if progress_info['current_day'] > total_tasks:
            return "final_report"
    cursor.execute("SELECT onboarding_state FROM users WHERE id = ?", (user_id,))
    user_onboarding_state = cursor.fetchone()['onboarding_state']
    if user_onboarding_state != 'completed':
        return "onboarding"
    else:
        return "qa"

def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("onboarding", onboarding_node)
    workflow.add_node("qa", qa_node)
    workflow.add_node("code_eval", code_eval_node)
    workflow.add_node("final_report", final_report_node)
    workflow.set_conditional_entry_point(route_logic, {"onboarding": "onboarding", "qa": "qa", "code_eval": "code_eval", "final_report": "final_report"})
    workflow.add_edge("onboarding", END)
    workflow.add_edge("qa", END)
    workflow.add_edge("code_eval", END)
    workflow.add_edge("final_report", END)
    return workflow.compile()

agent_graph = build_graph()

# --- 6. Scheduled Jobs ---

def send_daily_tasks():
    print(f"Running scheduled job: send_daily_tasks at {datetime.datetime.now()}")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, phone_number, name FROM users WHERE onboarding_state = 'completed'")
    all_users = cursor.fetchall()
    for user in all_users:
        user_id = user['id']
        cursor.execute("SELECT current_day FROM user_progress WHERE user_id = ?", (user_id,))
        progress = cursor.fetchone()
        if not progress: continue
        current_day = progress['current_day']
        cursor.execute("SELECT roadmap_json FROM roadmaps WHERE user_id = ?", (user_id,))
        roadmap_row = cursor.fetchone()
        if not roadmap_row or not roadmap_row['roadmap_json']: continue
        roadmap = json.loads(roadmap_row['roadmap_json'])
        if "error" in roadmap or not roadmap.get('daily_tasks'): continue
        if current_day > len(roadmap['daily_tasks']): continue
        task_for_today = next((task for task in roadmap['daily_tasks'] if task['day'] == current_day), None)
        if task_for_today:
            message_body = (
                f"👋 Good morning, {user['name']}!\n\n"
                f"Here is your task for Day {task_for_today['day']}:\n\n"
                f"*Topic:* {task_for_today['topic']}\n\n"
                f"*Your Mission:*\n{task_for_today['task']}\n\n"
                f"*Helpful Resources:*\n" + "\n".join(task_for_today.get('resources', []))
            )
            try:
                to_number = user['phone_number']
                if not to_number.startswith('whatsapp:'):
                    to_number = f"whatsapp:{to_number}"
                print(f"Sending task for Day {current_day} to {to_number}")
                twilio_client.messages.create(from_=TWILIO_WHATSAPP_NUMBER, body=message_body, to=to_number)
                new_day = current_day + 1
                now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
                cursor.execute("UPDATE user_progress SET current_day = ?, last_task_sent_at = ? WHERE user_id = ?", (new_day, now_utc, user_id))
            except Exception as e:
                print(f"Failed to send WhatsApp message to {user['phone_number']}: {e}")
    conn.commit()
    conn.close()

def send_reminders():
    print(f"Running scheduled job: send_reminders at {datetime.datetime.now()}")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, phone_number, name FROM users WHERE onboarding_state = 'completed'")
    active_users = cursor.fetchall()
    for user in active_users:
        user_id = user['id']
        cursor.execute("SELECT current_day FROM user_progress WHERE user_id = ?", (user_id,))
        progress = cursor.fetchone()
        if not progress: continue
        task_due_today = progress['current_day'] - 1
        if task_due_today <= 0: continue
        cursor.execute("SELECT id FROM submissions WHERE user_id = ? AND day = ?", (user_id, task_due_today))
        submission = cursor.fetchone()
        if not submission:
            reminder_message = f"Hi {user['name']}, this is a friendly reminder that your task for Day {task_due_today} is due soon. Let me know if you need any help!"
            try:
                to_number = user['phone_number']
                if not to_number.startswith('whatsapp:'):
                    to_number = f"whatsapp:{to_number}"
                print(f"Sending reminder for Day {task_due_today} to {to_number}")
                twilio_client.messages.create(from_=TWILIO_WHATSAPP_NUMBER, body=reminder_message, to=to_number)
            except Exception as e:
                print(f"Failed to send reminder to {user['phone_number']}: {e}")
    conn.close()

# --- 7. API Endpoints ---

@app.on_event("startup")
def on_startup():
    setup_database()
    scheduler.add_job(send_daily_tasks, 'cron', hour=8, minute=0)
    scheduler.add_job(send_reminders, 'cron', hour=20, minute=0)
    scheduler.start()
    print("APScheduler started with daily tasks and reminders.")
    print("LangGraph agent is ready.")

@app.on_event("shutdown")
def on_shutdown():
    scheduler.shutdown()
    print("APScheduler shut down.")

@app.get("/", summary="Root Health Check")
def read_root():
    return {"status": "ok", "message": "Internship Master Agent API is running!", "docs_url": "/docs"}

@app.post("/whatsapp/receive", summary="Receive WhatsApp Messages (Twilio)")
async def receive_whatsapp_message(From: str = Form(...), Body: str = Form(...), MediaContentType0: Optional[str] = Form(None), MediaUrl0: Optional[str] = Form(None)):
    user_phone_number = From
    message_body = Body
    file_content = None
    if MediaUrl0 and MediaContentType0 and "python" in MediaContentType0:
        response = requests.get(MediaUrl0, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        if response.status_code == 200:
            file_content = response.text
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE phone_number = ?", (user_phone_number,))
    user = cursor.fetchone()
    if not user:
        cursor.execute("INSERT INTO users (phone_number) VALUES (?)", (user_phone_number,))
        conn.commit()
        cursor.execute("SELECT * FROM users WHERE phone_number = ?", (user_phone_number,))
        user = cursor.fetchone()
    user_id = user['id']
    initial_state = {"user_id": user_id, "phone_number": user_phone_number, "message_body": message_body, "file_content": file_content, "db_connection": conn, "response": ""}
    final_state = agent_graph.invoke(initial_state)
    reply_message = final_state.get('response', "Sorry, something went wrong.")
    conn.close()
    xml_response = f"""<Response><Message>{reply_message}</Message></Response>"""
    return Response(content=xml_response, media_type="application/xml")

@app.post("/test/submit-code", summary="Test Code Submission Directly")
async def test_code_submission(phone_number: str = Form(...), code_file: UploadFile = File(...)):
    file_content = (await code_file.read()).decode('utf-8')
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE phone_number = ?", (phone_number,))
    user = cursor.fetchone()
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found. Please onboard the user first.")
    user_id = user['id']
    initial_state = {"user_id": user_id, "phone_number": phone_number, "message_body": "Code submission", "file_content": file_content, "db_connection": conn, "response": ""}
    final_state = agent_graph.invoke(initial_state)
    reply_message = final_state.get('response', "Sorry, something went wrong.")
    conn.close()
    return {"feedback": reply_message}

# --- 8. Running the App ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
