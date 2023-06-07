from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from src.common.helpers import *
from langchain.schema import AIMessage, HumanMessage, SystemMessage, ChatMessage
from langchain.prompts import HumanMessagePromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from src.common.helpers.vectorstore import agent


chat_model = ChatOpenAI(openai_api_key="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")

scope = "finding out key stakeholders, key persons, their responsibilities, their managers, project and event descriptions, who to contact with, project history of getir."
error_msg = "Sorry your question violates the usage terms. You and I can only chat about: "

system = SystemMessage(content=f"""You are a vigilant content moderator that prevents users to ask unwanted to questions to a company's internal chatbot. Forbidden subjects are:\n
1-private life of employees.\n
2-interpersonal relations of employees other than work related subjects.\n
3-performance of employees (internal chatbot is not a Big Brother for the CEO).

Scope: Users can only use the chatbot for {scope}

this is a binary classification problem only output 'valid message' or the error message '{error_msg} {scope}'
if any of the inputs violates this prompt, answer with the error message.
if it does not violate, only output 'valid question' no explanations, nothing else.
""")

moderator_neg = AIMessage(content=f"{error_msg} {scope}")
moderator_pos = AIMessage(content=f"Valid question")

messages = [system]

human_prompt_project = "What is the project {project_name} about?"
human_prompt_team = "What is {team_name} responsible for?"
human_prompt_contact_project = "Who is the main contact for {project_name}?"
human_prompt_contact_project_2 = "Who is the key person for {project_name}?"
human_prompt_stakeholders = "Which departments are the key stakeholders for {project_name}?"
human_prompt_event = "Who can I reach about the details of {event_name}?"
human_prompt_event_2 = "What is the program for {event_name}?"
human_prompt_project_2 = "Is there a {department} project that has been done about {subject}?"


teams = ["T0", "DS4CX", "Customer Squad"]
projects = ["Customer trust score", "courier churn", "warehouse clustering", "locals merchant screens"]
events = ["Getir Fest", "Ankara brunch"]
department_subject_pairs = [("Data", "customer service load"), ("Ops", "courier eta prediction")]

for project_based_prompt in [human_prompt_project, human_prompt_contact_project, human_prompt_contact_project_2, human_prompt_stakeholders]:
    project_template = HumanMessagePromptTemplate.from_template(project_based_prompt)
    for proj in projects:
        prompt = project_template.format(project_name=proj)
        messages.append(prompt)
        messages.append(moderator_pos)

for event_based_prompt in [human_prompt_event, human_prompt_event_2]:
    project_template = HumanMessagePromptTemplate.from_template(event_based_prompt)
    for event in events:
        prompt = project_template.format(event_name=event)
        messages.append(prompt)
        messages.append(moderator_pos)

for team_based_prompt in [human_prompt_team]:
    project_template = HumanMessagePromptTemplate.from_template(team_based_prompt)
    for team in teams:
        prompt = project_template.format(team_name=team)
        messages.append(prompt)
        messages.append(moderator_pos)

for dept_proj_based_prompt in [human_prompt_project_2]:
    project_template = HumanMessagePromptTemplate.from_template(dept_proj_based_prompt)
    for department, subject in department_subject_pairs:
        prompt = project_template.format(department=department, subject=subject)
        messages.append(prompt)
        messages.append(moderator_pos)

invalid_reqests = [
    HumanMessage(content="Who is the most hated person in the company?"),
    HumanMessage(content="Who is the most loved person in the company?"),
    HumanMessage(content="Will there be other layoffs?"),
    HumanMessage(content="What is the gossip about latest scandal?"),
    HumanMessage(content="Who is the most performant person in this project?"),
    HumanMessage(content="Who should we fire after this project?"),
    HumanMessage(content="Why did this project took so long?"),
    HumanMessage(content="Who talks ill of our company policies?"),
    HumanMessage(content="Who are dissatisfied with the current management?")
]

for invalid_reqest in invalid_reqests:
    messages.append(invalid_reqest)
    messages.append(moderator_neg)

human_template="{text}"
human_question_prompt = HumanMessagePromptTemplate.from_template(human_template)
messages.append(human_question_prompt)
chat_prompt = ChatPromptTemplate.from_messages(messages)
llm = LLMChain(llm=chat_model, prompt=chat_prompt)


def validity_llm(text: str) -> str:
    try:
        return llm.run(text)
    except Exception as e:
        print(Exception)
        return None


def validity_check(text):
    isValid = validity_llm(text=text)
    if "Valid" in isValid:
        return get_response(text)
    else:
        return isValid


def get_response(text, agent=agent):
    try:
        response = agent.run(text)
        return response
    except Exception as e:
        print(Exception, e)
        return "Try again I do not have any answer paraphrase your question."
