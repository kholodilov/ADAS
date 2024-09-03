# Structured Feedback and Ensemble Agent
def forward(self, taskInfo):
    # Step 1: Generate initial candidate solutions using multiple LLM agents
    initial_instruction = 'Please think step by step and then solve the task by writing the code.'
    initial_models = ['gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-0125', 'gpt-4o-2024-08-06']
    initial_agents = [LLMAgentBase(['thinking', 'code'], 'Initial Solution Agent', model=m, temperature=0.8) for m in initial_models]

    solutions = []
    for i, agent in enumerate(initial_agents):
        thoughts = agent([taskInfo], initial_instruction, i)
        thinking, code = thoughts[0], thoughts[1]
        feedback, correct_examples, wrong_examples = self.run_examples_and_get_feedback(code)
        if len(correct_examples) > 0:  # Only consider solutions that passed at least one example
            solutions.append({'thinking': thinking, 'code': code, 'feedback': feedback, 'correct_count': len(correct_examples)})
    # Step 2: Simulate human-like feedback for each candidate solution
    human_like_feedback_agent = LLMAgentBase(['thinking', 'feedback'], 'Human-like Feedback Agent', temperature=0.5)
    human_feedback_instruction = 'Please provide human-like feedback for the code, focusing on common mistakes, heuristic corrections, and best practices.'

    for i, sol in enumerate(solutions):
        thoughts = human_like_feedback_agent([taskInfo, sol['thinking'], sol['code'], sol['feedback']], human_feedback_instruction, i)
        human_thinking, human_feedback = thoughts[0], thoughts[1]
        sol['human_feedback'] = human_feedback
  
    # Step 3: Assign expert advisors to evaluate and provide targeted feedback
    expert_roles = ['Efficiency Expert', 'Readability Expert', 'Simplicity Expert']
    expert_advisors = [LLMAgentBase(['thinking', 'feedback'], role, model='gpt-4o-2024-08-06', temperature=0.6) for role in expert_roles]
    expert_instruction = 'Please evaluate the given code and provide targeted feedback for improvement.'

    for i, sol in enumerate(solutions):
        sol_feedback = {}
        for advisor in expert_advisors:
            thoughts = advisor([taskInfo, sol['thinking'], sol['code']], f'You are {advisor.agent_name}. ' + expert_instruction, i)
            thinking, feedback = thoughts[0], thoughts[1]
            sol_feedback[advisor.agent_name] = feedback
        sol['expert_feedback'] = sol_feedback

    # Step 4: Parse and structure the feedback to avoid redundancy and refine the solutions iteratively
    max_refinement_iterations = 2
    refinement_agent = LLMAgentBase(['thinking', 'code'], 'Refinement Agent', model='gpt-4-turbo-2024-04-09', temperature=0.5)

    for j, sol in enumerate(solutions):
        for i in range(max_refinement_iterations):
            combined_feedback = "Evaluation results:\n" + sol['feedback'].content + '\n\nGeneral feedback: ' + sol['human_feedback'].content + '\n' + '\n'.join([f"Feedback by {name}: " + fb.content for name, fb in sol['expert_feedback'].items()])
            # structured_feedback = ' '.join(set(combined_feedback.split()))  # Avoid redundancy
            refinement_instruction = 'Using the structured feedback, refine the solution to improve its performance.'
            thoughts = refinement_agent([taskInfo, sol['thinking'], sol['code'], Info('feedback', 'Structured Feedback', combined_feedback, j*10 + i)], refinement_instruction, j*10 + i)
            refinement_thinking, refined_code = thoughts[0], thoughts[1]
            feedback, correct_examples, wrong_examples = self.run_examples_and_get_feedback(refined_code)
            if len(correct_examples) >= sol['correct_count']:
                sol.update({'thinking': refinement_thinking, 'code': refined_code, 'feedback': feedback, 'correct_count': len(correct_examples)})

    # Step 5: Select the best-performing solutions and make a final decision using an ensemble approach
    sorted_solutions = sorted(solutions, key=lambda x: x['correct_count'], reverse=True)
    top_solutions = sorted_solutions[:3]  # Select the top 3 solutions

    final_decision_instruction = 'Given all the above solutions, reason over them carefully and provide a final answer by writing the code.'
    final_decision_agent = LLMAgentBase(['thinking', 'code'], 'Final Decision Agent', model='gpt-4-turbo-2024-04-09', temperature=0.1)
    final_inputs = [taskInfo] + [item for solution in top_solutions for item in [solution['thinking'], solution['code'], solution['feedback']]]
    final_thoughts = final_decision_agent(final_inputs, final_decision_instruction)
    final_thinking, final_code = final_thoughts[0], final_thoughts[1]
    answer = self.get_test_output_from_code(final_code)
    return (final_code, answer)
