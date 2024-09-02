# Structured Feedback and Ensemble Agent
def forward(self, taskInfo):
    # Step 1: Generate initial candidate solutions using multiple LLM agents
    initial_instruction = 'Please think step by step and then solve the task by writing the code.'
    num_candidates = 3  # Number of initial candidates
    initial_agents = [LLMAgentBase(['thinking', 'code'], 'Initial Solution Agent', temperature=0.8) for _ in range(num_candidates)]

    initial_solutions = []
    for i in range(num_candidates):
        thoughts = initial_agents[i]([taskInfo], initial_instruction, i)
        thinking, code = thoughts[0], thoughts[1]
        feedback, correct_examples, wrong_examples = self.run_examples_and_get_feedback(code)
        if len(correct_examples) > 0:  # Only consider solutions that passed at least one example
              initial_solutions.append({'thinking': thinking, 'code': code, 'feedback': feedback, 'correct_count': len(correct_examples)})
    # Step 2: Simulate human-like feedback for each candidate solution
    human_like_feedback_agent = LLMAgentBase(['thinking', 'feedback'], 'Human-like Feedback Agent', temperature=0.5)
    human_feedback_instruction = 'Please provide human-like feedback for the code, focusing on common mistakes, heuristic corrections, and best practices.'

    for i, sol in enumerate(initial_solutions):
        thoughts = human_like_feedback_agent([taskInfo, sol['thinking'], sol['code']], human_feedback_instruction, i)
        human_thinking, human_feedback = thoughts[0], thoughts[1]
        sol['human_feedback'] = human_feedback
  
    # Step 3: Assign expert advisors to evaluate and provide targeted feedback
    expert_roles = ['Efficiency Expert', 'Readability Expert', 'Simplicity Expert']
    expert_advisors = [LLMAgentBase(['thinking', 'feedback'], role, temperature=0.6) for role in expert_roles]
    expert_instruction = 'Please evaluate the given code and provide targeted feedback for improvement.'

    for i, sol in enumerate(initial_solutions):
        sol_feedback = {}
        for advisor in expert_advisors:
            thoughts = advisor([taskInfo, sol['thinking'], sol['code']], expert_instruction, i)
            thinking, feedback = thoughts[0], thoughts[1]
            sol_feedback[advisor.role] = feedback
        sol['expert_feedback'] = sol_feedback

    # Step 4: Parse and structure the feedback to avoid redundancy and refine the solutions iteratively
    max_refinement_iterations = 2
    refinement_agent = LLMAgentBase(['thinking', 'code'], 'Refinement Agent', temperature=0.5)
    refined_solutions = []

    for j, sol in enumerate(initial_solutions):
        for i in range(max_refinement_iterations):
            combined_feedback = sol['feedback'].content + sol['human_feedback'].content + ''.join([fb.content for fb in sol['expert_feedback'].values()])
            structured_feedback = ' '.join(set(combined_feedback.split()))  # Avoid redundancy
            refinement_instruction = 'Using the structured feedback, refine the solution to improve its performance.'
            thoughts = refinement_agent([taskInfo, sol['thinking'], sol['code'], Info('feedback', 'Structured Feedback', structured_feedback, j*10 + i)], refinement_instruction, j*10 + i)
            refinement_thinking, refined_code = thoughts[0], thoughts[1]
            feedback, correct_examples, wrong_examples = self.run_examples_and_get_feedback(refined_code)
            if len(correct_examples) > 0:
                sol.update({'thinking': refinement_thinking, 'code': refined_code, 'feedback': feedback, 'correct_count': len(correct_examples)})
                refined_solutions.append(sol)

    # Step 5: Select the best-performing solutions and make a final decision using an ensemble approach
    sorted_solutions = sorted(refined_solutions, key=lambda x: x['correct_count'], reverse=True)
    top_solutions = sorted_solutions[:3]  # Select the top 3 solutions

    final_decision_instruction = 'Given all the above solutions, reason over them carefully and provide a final answer by writing the code.'
    final_decision_agent = LLMAgentBase(['thinking', 'code'], 'Final Decision Agent', temperature=0.1)
    final_inputs = [taskInfo] + [item for solution in top_solutions for item in [solution['thinking'], solution['code'], solution['feedback']]]
    final_thoughts = final_decision_agent(final_inputs, final_decision_instruction)
    final_thinking, final_code = final_thoughts[0], final_thoughts[1]
    answer = self.get_test_output_from_code(final_code)
    return (final_code, answer)
