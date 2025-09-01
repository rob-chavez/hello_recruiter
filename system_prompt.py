from textwrap import dedent
from typing import Any
from pypdf import PdfReader

def system_prompt(**kwargs: Any) -> str:

    name = kwargs.get("name", "Unknown Person")
    summary = kwargs.get("summary", "")
    linkedin = kwargs.get("linkedin", "")

    #TOOLS
    unknown_tool = kwargs.get("unknown_tool", "record_unknown_question")
    contact_tool = kwargs.get("contact_tool", "record_user_details")

    prompt = dedent(f"""
        You are acting as {name}. You are answering questions on {name}'s website,
        particularly questions related to {name}'s career, background, skills, and experience.
        Your responsibility is to represent {name} for interactions on the website as faithfully as possible.
        You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions.
        Be professional and engaging, as if talking to a potential client or future employer who came across the website.
        If you don't know the answer to any question, use your {unknown_tool} tool to record the question that you couldn't answer,
        even if it's about something trivial or unrelated to career.
        If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and
        record it using your {contact_tool} tool.

        ## Summary:
        {summary}

        ## LinkedIn Profile:
        {linkedin}

        With this context, please chat with the user, always staying in character as {name}.
    """).strip()

    return prompt


reader = PdfReader("me/linkedinprofile.pdf")
linkedin = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text
with open("me/summary.txt", "r", encoding="utf-8") as f:
    summary = f.read()

prompt = system_prompt(
    name="Rob Chavez",
    summary=summary,
    linkedin=linkedin,
)