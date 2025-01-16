# task_executor.py

from agent import Agent
from data_loader import DataLoader
from config import MBTI_TYPES, EXPERIMENT_CONFIG, TASK_MODEL_MAPPING
from dotenv import load_dotenv, find_dotenv
import random
import os
from typing import List

_ = load_dotenv(find_dotenv())
    
class TaskExecutor:
    def __init__(self, task_type, specific_task, api_keys, model_type, data_path):
        """
        Initialize task executor.
        :param task_type: Main task type (e.g., open_task, complex_task)
        :param specific_task: Specific task type (e.g., poetry, story, nli, math)
        :param api_keys: Dictionary of API keys
        :param model_type: Model type
        :param data_path: Path to the data file
        """
        self.task_type = task_type
        self.specific_task = specific_task
        self.config = EXPERIMENT_CONFIG[task_type]
        self.api_keys = api_keys
        self.model_type = model_type
        self.data = DataLoader.load_data(data_path, specific_task)  # Load data based on task type

    def run_task(self):
        """
        Execute multi-round tasks.
        :return: Results of each round.
        """
        rounds = self.config["rounds"]
        agents_per_round = self.config["agents_per_round"]
        final_round_personalities = self.config["final_round_personalities"]

        # Round 0: Initialize agents
        personalities = MBTI_TYPES["open_personalities"] if self.specific_task in ["poetry", "story"] else MBTI_TYPES["complex_personalities"]
        agents, cached_personalities = self._create_agents(personalities, agents_per_round)

        inputs = self._prepare_initial_prompts()  # Initial list of questions (original question, prompt)
        # print("inputs:",inputs)
        results = []

        for round_num in range(rounds):
            print(f"-------------------------Round {round_num + 1}/{rounds}--------------------------")
            # print("inputs:",inputs)
            if round_num == rounds - 1:
                # Final round: Use one agent to summarize and select the best answer
                # print("results length:",len(results))
                # print("results:",results)
                # print("results[-1]:",results[-1])
                final_result = self._run_final_round(results[-1], final_round_personalities)
                results.append(final_result)
            else:
                # Round 0 to rounds-2
                round_results, new_inputs = self._run_round(inputs, agents, cached_personalities)
                results.append(round_results)

                # print("round_results:",round_results)
                # print("results:",results)

                # Ensure each question generates only one set of inputs
                inputs = [
                    self._generate_round_prompt(result)
                    for result in round_results
                    if result["agent_number"] == 0  # Ensure that each question generates only one set of inputs
                ]

        return results
    
    def _select_personality(self, personalities, agents_per_round):
        """
        Randomly select a specified number of personality types, avoiding duplicates.
        :param personalities: List of selectable personality types
        :param agents_per_round: Number of agents needed per round
        :return: Randomly selected personalities
        """
        if len(personalities) >= agents_per_round:
            return random.sample(personalities, agents_per_round)
        else:
            return random.choices(personalities, k=agents_per_round)
    
    def _create_agents(self, personalities, agents_per_round):
        """
        Create a list of agents and record their corresponding personalities.
        :param personalities: Personality types used in the current round
        :param agents_per_round: Number of agents per round
        :return: List of agents and their personalities
        """
        selected_personalities = self._select_personality(personalities, agents_per_round)
        agents = [Agent(personality, self.model_type, self.api_keys) for personality in selected_personalities]
        return agents, selected_personalities

    def _generate_outputs(self, agents: List[Agent], prompt: str) -> List[str]:
        """
        Let agents generate outputs.
        :param agents: List of agents
        :param prompt: Input task prompt
        :return: List of outputs
        """
        return [agent.generate(prompt) for agent in agents]

    def _run_round(self, inputs, agents, cached_personalities):
        """
        Execute one round of the task.
        :param inputs: Inputs for the current round
        :param agents: List of agents (reused from round 0 to rounds-2)
        :param cached_personalities: List of agent personalities
        :return: Results of the current round
        """
        results = []
        for original_question, prompt in inputs:
            # Each agent generates outputs
            outputs = self._generate_outputs(agents, prompt)  
            # print("original_question type:",type(original_question))
            # print("original_question:",original_question)
            # Record the results of each agent
            question_results = []
            for agent_idx, output in enumerate(outputs):
                other_responses = outputs[:agent_idx] + outputs[agent_idx + 1:]
                if self.specific_task in ["math","nli"]:
                    question_results.append({
                        "question": original_question,  # The first half of the original input
                        "input": prompt,  # The complete prompt for this round
                        "output": output,  # Current agent output
                        "other_responses": other_responses,
                        "agent_number": agent_idx, # Current agent number
                        "personalities": cached_personalities[agent_idx]  # The personality type of the current agent
                    })
                elif self.specific_task in ["poetry","story"]:
                    question_results.append({
                        "question": original_question,  # The first half of the original input
                        "input": prompt,  # The complete prompt for this round
                        "output": output,  # Current agent output
                        "other_responses": other_responses,
                        "agent_number": agent_idx, # Current agent number
                        "personalities": cached_personalities[agent_idx]  # The personality type of the current agent
                    })
            # Ensure that only agents_per_round results are generated per question
            #print("question_results:",question_results)
            results.extend(question_results[:len(agents)])
        
        # Generate new prompts for the next round
        new_inputs = [
            self._generate_round_prompt(result) 
            for result in results if result["agent_number"] == 0
        ]

        return results, new_inputs

    def _run_final_round(self, last_round_results, final_round_personalities):
        """
        Execute the final round of the task, summarizing and deciding the best answer for each question.
        :param last_round_results: Results from the previous round
        :param final_round_personalities: Personalities used in the final round
        :return: Final summarized results for each question
        """
        final_results = []
        # print("last_round_results:",last_round_results)
        grouped_results = self._group_by_question(last_round_results)
        
        for question_data in grouped_results:
            # Generate summary prompts based on task type
            summary_prompt = self._generate_final_prompt(question_data)
            # Summarizing with an Agent
            personality = random.choice(final_round_personalities)
            agent = Agent(personality, self.model_type, self.api_keys)
            final_output = agent.generate(summary_prompt)
            # Save the final output
            final_results.append({
                "question": question_data["question"],  # The first half of the original input
                "input": summary_prompt,  # Complete prompt
                "personality": personality,
                "final_output": final_output
            })
        return final_results
    
    def _group_by_question(self, results):
        """
        Group results by question.
        :param results: Current round's results
        :return: Grouped results by question
        """
        # print("results:",results)
        grouped = {}
        for result in results:
            # print("result", result)
            # question = result["question"]
            # print("question:",question)
            # 按问题分组
            # if question_key not in grouped:
            #     grouped[question_key] = {"question": result["question"], "responses": []}

            if self.specific_task in ["math","nli"]:
                question_key = result["question"]["question"]
                if question_key not in grouped:
                    grouped[question_key] = {"question": result["question"]["question"], "responses": []}
            elif self.specific_task in ["poetry","story"]:
                question_key = result["question"]
                if question_key not in grouped:
                    grouped[question_key] = {"question": result["question"], "responses": []}
            
            grouped[question_key]["responses"].append(result["output"])
        return list(grouped.values())

    def _prepare_initial_prompts(self):
        """
        Generate initial prompt according to task type
        :return: Initial prompt list
        """
        #print("self.data:",self.data)
        self.data = self.data[:3]

        prompts = []
        for data_item in self.data:
            if self.specific_task == "poetry":
                prompts.append((
                    data_item,
                    self._generate_prompt(data_item)
                ))
            elif self.specific_task == "story":
                prompts.append((
                    data_item,
                    self._generate_prompt(data_item)
                ))
            elif self.specific_task == "nli":
                prompts.append((
                    data_item,
                    self._generate_prompt(data_item)
                ))
            elif self.specific_task == "math":
                prompts.append((
                    data_item,
                    self._generate_prompt(data_item)
                ))
            else:
                raise ValueError(f"Unsupported specific task type: {self.specific_task}")
        
        return prompts

    def _generate_round_prompt(self, result):
        """
        Generate prompts for intermediate rounds based on task type
        :param result: Results of the previous round
        :return: A new round of prompts
        """
        if self.task_type == "open_task":
            # Open-ended tasks: poetry generation or story creation
            if self.specific_task == "poetry":  # Poetry Generation
                return (result["question"],  # Original question
                        "Your task is to compose elegant and beautiful Chinese poetry. Below are your previous response and the responses from other agents. Please reflect on them and create an even more exquisite next line of poetry."
                        "\n\n**Please adhere to the following requirements:**\n"
                        "1. **Synthesize and Refine**: Combine the strengths of your response and the responses from other agents. Avoid repetition or simple stitching, and aim for integrated creativity.\n"
                        "2. **Maintain Originality and Continuity**: The new line must be entirely original while maintaining coherence with the artistic conception of the previous line. Use refined language consistent with the style of classical Chinese poetry.\n"
                        "3. **Enhance Quality**: Strive for a more concise expression, deeper artistic meaning, and harmonious rhythm, demonstrating a higher level of literary skill.\n"
                        "4. **Strict Output Format**: Only output a single line of poetry without any additional content.\n\n"
                        f"**Your Response:** {result['output']}\n"
                        f"**Other Agents' Responses:** {', '.join(result['other_responses'])}\n\n"
                        "Based on the above content, please create a higher-quality next line of poetry:")
            elif self.specific_task == "story":  # Story Creation
                return (result["question"],  # Original question
                        "Your task is to craft an engaging story based on the provided keywords. Below are your previous response and the responses from other agents. Please analyze and synthesize these responses to improve and enhance the original story, making it more captivating."
                        "\n\n**Please adhere to the following requirements:**\n"
                        "1. **Synthesize and Refine**: Combine the strengths of your response and the responses from other agents. Avoid repetition or simple stitching, and aim for integrated creativity.\n"
                        "2. **Enhance Engagement**: Optimize the story's plot to make it more vivid, interesting, and captivating for readers.\n"
                        "3. **Use Vivid Language**: Use more expressive and engaging language that matches the tone of storytelling.\n"
                        "4. **Strict Output Format**: Only output the complete story content without any additional explanations.\n\n"
                        f"**Your Response:** {result['output']}\n"
                        f"**Other Agents' Responses:** {', '.join(result['other_responses'])}\n\n"
                        "Based on the above content, please optimize and rewrite the complete story:")

        elif self.task_type == "complex_task":
            # Complexity Tasks
            if self.specific_task == "nli":  # NLI
                return (result["question"],  
                        "Based on the following natural language inference (NLI) task, please analyze and synthesize the responses from multiple agents to provide a final conclusion."
                        "\n\n**Please adhere to the following requirements:**\n"
                        "1. **Analyze and Compare**: Evaluate the strengths and weaknesses of your response and those of other agents.\n"
                        "2. **Ensure Accuracy**: The final conclusion must be precise and logically consistent with the given premise and hypothesis.\n"
                        "3. **Strict Output Format**: Only output 'entailment', 'contradiction', or 'neutral' without any additional content.\n\n"
                        f"**Premise:** {result['question']}\n"
                        f"**Hypothesis:** {result['input']}\n"
                        f"**Your Response:** {result['output']}\n"
                        f"**Other Agents' Responses:** {', '.join(result['other_responses'])}\n\n"
                        "Based on the above, provide the final conclusion:")
            elif self.specific_task == "math":  #Mathematical Reasoning
                return (result["question"],  
                        "The following are multiple solutions to a mathematical problem. Please analyze and synthesize these solutions to provide a comprehensive final answer."
                        "\n\n**Please adhere to the following requirements:**\n"
                        "1. **Analyze and Validate**: Evaluate the correctness of your response and those from other agents.\n"
                        "2. **Ensure Completeness**: The final solution must include detailed, logical steps and provide an accurate answer.\n"
                        "3. **Strict Output Format**: Output the full solution steps, and conclude with the answer in the format '#### Final Answer'.\n\n"
                        f"**Problem:** {result['question']}\n"
                        f"**Your Response:** {result['output']}\n"
                        f"**Other Agents' Responses:** {', '.join(result['other_responses'])}\n\n"
                        "Based on the above, rewrite the complete solution steps:")

        raise ValueError(f"Unsupported task type: {self.task_type}")

    def _generate_prompt(self, data_item):
        """
        Generate prompt according to task type
        :param data_item: Current data item
        :param round_num: Current round
        :return: prompt 
        """
        # print("data_item:",data_item)
        if self.task_type == "open_task":
            if self.specific_task == "poetry":  # Poetry Generation
                return (
                    "You are a highly talented Chinese ancient poet, renowned for composing exquisite Chinese poetry. I will provide the opening line of a poem, and your task is to craft the next line based on its artistic conception, rhythm, and style, creating a harmonious and beautiful continuation."
                    "\n\nPlease adhere to the following requirements:\n"
                    "1. **Originality**: Your line must be entirely original, not borrowed from any existing poetry.\n"
                    "2. **Rhythmic Consistency**: Ensure the tonal pattern and rhyme scheme align perfectly with the previous line.\n"
                    "3. **Artistic Continuity**: Your line should seamlessly extend the meaning, atmosphere, and imagery of the given line, evoking a sense of beauty in nature, emotion, or philosophy.\n"
                    "4. **Elegance in Language**: Employ refined and elegant diction that reflects the style and artistry of classical Chinese poetry.\n"
                    "5. **Strict Output Format**: Only output a single poetic line, without any additional explanations or context.\n\n"
                    f"**Opening Line:** {data_item}\n**Next Line:**"
                )
            elif self.specific_task == "story":  # Story Generation
                return (
                    "You are an imaginative and skilled storyteller, renowned for crafting captivating and engaging narratives. I will provide you with a set of keywords, and your task is to write a short story that seamlessly incorporates these keywords into its plot."
                    "\n\nPlease adhere to the following requirements:\n"
                    "1. **Incorporate All Keywords**: The story must include all the provided keywords, integrating them naturally and meaningfully into the storyline.\n"
                    "2. **Structured Narrative**: Ensure the story has a clear beginning, middle, and end, with a logical and coherent progression.\n"
                    "3. **Engaging and Vivid Language**: Use descriptive and vivid language to bring the story to life and captivate the audience.\n"
                    "4. **Creativity**: Demonstrate originality and creativity, making the story unique and memorable.\n\n"
                    f"**Keywords**: {', '.join(data_item)}\n**Start your story:**"
                )

        elif self.task_type == "complex_task":
            if self.specific_task == "nli":  # NLI
                return (
                    "You are a logical reasoning assistant with exceptional skills in analyzing the relationship between a premise and a hypothesis. I will provide you with a pair of sentences, and your task is to determine their relationship and explain your reasoning."
                    "\n\nPlease adhere to the following requirements:\n"
                    "1. **Relationship Determination**: Identify whether the relationship is 'entailment', 'neutral', or 'contradiction', and map it to the corresponding label value:\n"
                    "   - 'entailment' → label value: 0\n"
                    "   - 'neutral' → label value: 1\n"
                    "   - 'contradiction' → label value: 2\n"
                    "2. **Output Format**: Strictly output only the label value and a concise explanation of your reasoning.\n"
                    "   Example:\n"
                    "   - 0: The hypothesis logically follows from the premise.\n"
                    "   - 1: The hypothesis is unrelated or lacks sufficient information to infer from the premise.\n"
                    "   - 2: The hypothesis contradicts the premise.\n\n"
                    f"**Premise**: {data_item['premise']}\n"
                    f"**Hypothesis**: {data_item['hypothesis']}\n\n"
                    "**Start your analysis and provide your output:**"
                )
            elif self.specific_task == "math":  # Mathematical Reasoning
                # question = data_item["question"]
                return (
                    "You are an expert mathematics assistant, skilled at clearly explaining and solving complex mathematical problems. I will provide you with a problem, and your task is to solve it step by step while adhering to the following requirements."
                    "\n\nPlease adhere to the following requirements:\n"
                    "1. **Detailed Steps**: Provide a step-by-step solution, ensuring each step is logical, detailed, and easy to follow.\n"
                    "2. **Final Answer Format**: Clearly state the final answer at the end in the format 'Answer: XXX'.\n"
                    "3. **Units**: If the problem involves units, include them in your final answer.\n"
                    "4. **Clarity and Precision**: Make sure your explanation is concise, accurate, and free of ambiguity.\n\n"
                    f"**Problem**: {data_item['question']}\n\n"
                    "**Start solving:**"
                )

        raise ValueError(f"Unsupported task type: {self.specific_task}")

    def _generate_final_prompt(self, question_data):
        """
        Generate a summary prompt for the final round based on the task type
        :param question_data: Results of a question grouping
        :return: Final summary prompt
        """
        if self.task_type == "open_task":
            if self.specific_task == "poetry":  # Poetry Generation
                return (
                    f"Opening line: {question_data['question']}\n"
                    "The following are poetic responses from multiple agents to the given opening line. Please analyze and select the best answer:\n"
                    + "\n".join([f"Agent {idx+1}'s response: {response}" for idx, response in enumerate(question_data['responses'])])
                    + "\n\nPlease evaluate the responses based on the following criteria:\n"
                    "1. **Artistic Depth**: The response should evoke profound imagery and meaning.\n"
                    "2. **Language Elegance**: The language should be refined and poetic.\n"
                    "3. **Rhythmic Harmony**: The response should maintain tonal and rhyming consistency.\n"
                    "4. **Creativity and Coherence**: The response should demonstrate originality while seamlessly continuing the opening line.\n\n"
                    "Please strictly adhere to the following output format:\n"
                    "{Best Answer: , Explanation: }"
                )
            elif self.specific_task == "story":  # Story Creation
                return (
                    f"Keywords: {question_data['question']}\n"
                    "The following are story responses from multiple agents based on the given keywords. Please analyze and select the best answer:\n"
                    + "\n".join([f"Agent {idx+1}'s response: {response}" for idx, response in enumerate(question_data['responses'])])
                    + "\n\nPlease evaluate the responses based on the following criteria:\n"
                    "1. **Plot Completeness**: The story should have a clear beginning, middle, and end.\n"
                    "2. **Engagement**: The story should captivate the reader and maintain interest.\n"
                    "3. **Vivid Language**: The language should be descriptive and bring the story to life.\n\n"
                    "Please strictly adhere to the following output format:\n"
                    "{Best Answer: , Explanation: }"
                )

        elif self.task_type == "complex_task":
            if self.specific_task == "nli":  #  NLI
                return (
                    f"Premise: {question_data['question']}\n"
                    "The following are inference results from multiple agents regarding the given hypothesis. Please analyze and select the best answer:\n"
                    + "\n".join([f"Agent {idx+1}'s response: {response}" for idx, response in enumerate(question_data['responses'])])
                    + "\n\nPlease evaluate the responses based on the following criteria:\n"
                    "1. **Accuracy**: The response must correctly reflect the relationship between the premise and the hypothesis.\n"
                    "2. **Logical Consistency**: The reasoning must be logically sound and free of contradictions.\n\n"
                    "Please strictly adhere to the following output format:\n"
                    "{Best Answer: , Explanation: }"
                )
            elif self.specific_task == "math":  # Mathematical Reasoning
                return (
                    f"Problem: {question_data['question']}\n"
                    "The following are solutions from multiple agents to the given problem. Please analyze and select the best answer:\n"
                    + "\n".join([f"Agent {idx+1}'s response: {response}" for idx, response in enumerate(question_data['responses'])])
                    + "\n\nPlease evaluate the responses based on the following criteria:\n"
                    "1. **Accuracy**: The solution must be mathematically correct.\n"
                    "2. **Completeness**: The solution must include all necessary steps and provide a full explanation.\n\n"
                    "Please strictly adhere to the following output format:\n"
                    "{Best Answer: , Explanation: }"
                )

        raise ValueError(f"Unsupported task type: {self.specific_task}")
