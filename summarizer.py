from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_llm():
    return ChatMistralAI(
        model = "ministral-8b-latest",
        temperature=0.2,
        api_key=os.getenv("MISTRAL_API_KEY")
    )

def split_text(transcript: str)-> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 4000,
        chunk_overlap = 200
    )

    return splitter.split_text(transcript)

def reduce_summaries(summaries: list[str], chain, group_size: int = 10) -> str:

    while len(summaries) > 1:
        grouped = [
            summaries[i:i + group_size] for i in range(0, len(summaries), group_size)
        ]

        # invoking chain
        summaries = chain.batch(
            [
                { "text": "\n\n".join(group) } for group in grouped
            ]
        )

        print(f"Reduced to {len(summaries)} summaries")

    return summaries[0]

def summarize(transcript: str) -> str:
    llm = load_llm()
    
    chunks = split_text(transcript)

    chunk_prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are an expert meeting summarizer.
Summarize this transcript chunk.
Extract only information explicitly present in the chunk.

Return:

Key Points:
- ...

Decisions:
- ...

Action Items:
- Owner:
  Task:
  Deadline:

Risks:
- ...

Open Questions:
- ...

Rules:
- Omit empty sections.
- Always attribute decisions and action items when possible.
- Include deadlines when mentioned.
- Open Questions: Questions, decisions, concerns, blockers, or topics explicitly left unresolved and requiring follow-up.
- Decisions: Any choice, approval, agreement, prioritization, acceptance, rejection, or conclusion reached during the meeting.
- Action items: Assignments, Commitments, Promises, Follow-up tasks, Agreed next steps
"""),
    ("human", "{text}")
    ])

    chunk_chain = chunk_prompt | llm | StrOutputParser()

    chunk_summaries = chunk_chain.batch(
        [{"text": chunk} for chunk in chunks]
    )

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system","""
You are combining multiple meeting summaries.
Merge the information into a single summary.
         
Rules:
- Remove duplicate information.
- Merge repeated action items.
- Consolidate overlapping decisions.
- Preserve speaker attribution.
- Preserve deadlines.
- Preserve risks.
- Preserve unresolved questions.

Return using exactly this structure:

Key Points:
...

Decisions:
...

Action Items:
...

Risks:
...

Open Questions:
...

- Omit empty sections.
"""
        ),
        ("human", "{text}")
    ])

    summary_chain = (
        summary_prompt | llm | StrOutputParser()
    )

    compressed_summary = reduce_summaries(
        chunk_summaries,
        summary_chain,
        group_size=10
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are an expert meeting summarizer.
Create a final professional meeting report.

Return:
- Title
- Key Discussion Points
- Decisions Made
- Action Items (Owner, Task, Deadline)
- Risks
- Open Questions

Rules:
- Remove duplicates.
- Preserve speaker attribution.
- Include deadlines when available.
- If a section has no information, write "Information not present".
- Open Questions: Questions, decisions, concerns, blockers, or topics explicitly left unresolved and requiring follow-up.
- Decisions: Any choice, approval, agreement, prioritization, acceptance, rejection, or conclusion reached during the meeting.
- Action items: Assignments, Commitments, Promises, Follow-up tasks, Agreed next steps
"""
        ),
        ("human", "{text}")
    ])

    final_chain = (
        final_prompt | llm | StrOutputParser()
    )

    final_summary = final_chain.invoke(
        {"text": compressed_summary}
    )

    return final_summary
