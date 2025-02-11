from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from .config.in_out_struct import UserRequirements

####################################################################################################
##################################### Crew Definition ##############################################
####################################################################################################
@CrewBase
class CustomerServiceCrew():
	"""CustomerServiceCrew crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	def __init__(self, llm):
		self.llm = llm
	

	@agent
	def customer_service_specialist(self) -> Agent:
		return Agent(
			config=self.agents_config['customer_service_specialist'],
			llm=self.llm,
			verbose=True
		)


	@task
	def customer_service_specialist_task(self) -> Task:
		return Task(
			config=self.tasks_config['customer_service_specialist_task'],
			output_pydantic = UserRequirements,
			verbose=True,
		)


	@crew
	def crew(self) -> Crew:
		"""Creates the CustomerServiceCrew crew"""

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			#memory = True,
			llm=self.llm,
			verbose=True
			)
