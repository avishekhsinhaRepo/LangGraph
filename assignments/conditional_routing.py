from typing import TypedDict
from langgraph.graph import END, START, StateGraph

# Define the structure of the input state (job application)
class JobApplication(TypedDict):
    applicant_name: str
    years_experience: int

# TODO: Implement the function to categorize candidates based on experience
def categorize_candidate(application: JobApplication):
    pass  # Replace this with actual implementation

# Function for interview scheduling
def schedule_interview(application: JobApplication):
    print(f"Candidate {application['applicant_name']} is shortlisted for an interview.")
    return {"status": "Interview Scheduled"}

# Function for skills test
def assign_skills_test(application: JobApplication):
    print(f"Candidate {application['applicant_name']} is assigned a skills test.")
    return {"status": "Skills Test Assigned"}

# Create the state graph
graph = StateGraph(JobApplication)

# TODO: Add nodes to the graph

# TODO: Define edges (workflow steps)

# Compile the workflow
runnable = graph.compile()

# Simulate job applications
print(runnable.invoke({"applicant_name": "Alice", "years_experience": 6}))
print(runnable.invoke({"applicant_name": "Bob", "years_experience": 3}))
