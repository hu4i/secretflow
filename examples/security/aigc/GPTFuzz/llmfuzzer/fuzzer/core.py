from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mutator import Mutator, MutatePolicy
    from .selection import SelectPolicy

from llmfuzzer.utils.template import synthesis_message
from llmfuzzer.utils.predict import Predictor


class PromptNode:
    def __init__(self,
                 fuzzer: 'LLMFuzzer',
                 prompt: str,
                 response: str = None,
                 results: 'list[int]' = None,
                 parent: 'PromptNode' = None,
                 mutator: 'Mutator' = None):
        self.fuzzer: 'LLMFuzzer' = fuzzer
        self.prompt: str = prompt
        self.response: str = response
        self.results: 'list[int]' = results
        self.visited_num = 0

        self.parent: 'PromptNode' = parent
        self.mutator: 'Mutator' = mutator
        self.child: 'list[PromptNode]' = []
        self.level: int = 0 if parent is None else parent.level + 1

        self._index: int = None

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: int):
        self._index = index
        if self.parent is not None:
            self.parent.child.append(self)

    @property
    def num_jailbreak(self):
        return sum(self.results)

    @property
    def num_reject(self):
        return len(self.results) - sum(self.results)

    @property
    def num_query(self):
        return len(self.results)


class LLMFuzzer:
    def __init__(self,
                 questions: 'list[str]',
                 target,
                 predictor: 'Predictor',
                 initial_seed: 'list[str]',
                 mutate_policy: 'MutatePolicy',
                 select_policy: 'SelectPolicy',
                 logger,
                 max_query: int = -1,
                 max_jailbreak: int = -1,
                 max_reject: int = -1,
                 max_iteration: int = -1,
                 energy: int = 1,
                 result_file: str = None,
                 ):

        self.questions: 'list[str]' = questions
        self.target = target
        self.predictor = predictor
        self.prompt_nodes: 'list[PromptNode]' = [
            PromptNode(self, prompt) for prompt in initial_seed
        ]
        self.initial_prompts_nodes = self.prompt_nodes.copy()

        for i, prompt_node in enumerate(self.prompt_nodes):
            prompt_node.index = i

        self.mutate_policy = mutate_policy
        self.select_policy = select_policy

        self.current_query: int = 0
        self.current_jailbreak: int = 0
        self.current_reject: int = 0
        self.current_iteration: int = 0

        self.max_query: int = max_query
        self.max_jailbreak: int = max_jailbreak
        self.max_reject: int = max_reject
        self.max_iteration: int = max_iteration

        self.energy: int = energy
        self.logger = logger

        self.setup()

    def setup(self):
        self.mutate_policy.fuzzer = self
        self.select_policy.fuzzer = self

    def is_stop(self):
        checks = [
            ('max_query', 'current_query'),
            ('max_jailbreak', 'current_jailbreak'),
            ('max_reject', 'current_reject'),
            ('max_iteration', 'current_iteration'),
        ]
        return any(getattr(self, max_attr) != -1 and getattr(self, curr_attr) >= getattr(self, max_attr) for max_attr, curr_attr in checks)

    def run(self):
        self.logger.info("Fuzzing started!")
        # try:
        while not self.is_stop():
            seed = self.select_policy.select()
            mutated_results = self.mutate_policy.mutate_single(seed)
            self.evaluate(mutated_results)
            self.update(mutated_results)
            self.log()

        self.logger.info("Fuzzing finished!")

    def evaluate(self, prompt_nodes: 'list[PromptNode]'):
        for prompt_node in prompt_nodes:
            responses = []
            for question in self.questions:
                message = synthesis_message(question, prompt_node.prompt)
                if message is None:  # The prompt is not valid
                    prompt_node.response = []
                    prompt_node.results = []
                    break
                response = self.target.generate(message)
                responses.append(response[0] if isinstance(
                    response, list) else response)
            else:
                prompt_node.response = responses
                prompt_node.results = self.predictor.predict(responses)

    def update(self, prompt_nodes: 'list[PromptNode]'):
        self.current_iteration += 1

        for prompt_node in prompt_nodes:
            if prompt_node.num_jailbreak > 0:
                prompt_node.index = len(self.prompt_nodes)
                self.prompt_nodes.append(prompt_node)
                self.logger.info(f"Question: \n{self.questions[0]}")
                self.logger.info(f"Response: \n{prompt_node.response}")
                print(prompt_node.response)

            self.current_jailbreak += prompt_node.num_jailbreak
            self.current_query += prompt_node.num_query
            self.current_reject += prompt_node.num_reject

        self.select_policy.update(prompt_nodes)

    def log(self):
        self.logger.info(
            f"Iteration {self.current_iteration}: {self.current_jailbreak} jailbreaks, {self.current_reject} rejects, {self.current_query} queries")