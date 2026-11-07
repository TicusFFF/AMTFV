import os
import sys
sys.path.append(os.environ["PROJECT_PATH"])
import re
import time
import datetime
import random
from tqdm import tqdm
from copy import deepcopy
from argparse import ArgumentParser
from extract_answer import extract_answer
from utils.call_llm import call_llm
from utils.process_file import read_file,save_file
from mod_C0_compress import maybe_compress_text

# Agent / Prompt numbering for 2-2-1-1_P4-5Py, based on 2-2-1_toolFlowMathReasoning
#
# P1 / Agent1: Reverse Verification Flow Generator
#     gen_verify_code_dict
#
# P2 / Agent2: Main Verification Flow Executor
#     execute_verify_code_dict
#     + PythonToolLLMCodeExecutor.__init__ runtime prompt
#
# P3 / Agent3: Feedback Summarizer
#     feedback_dict
#
# P4 / Agent4: Verification Flow Refiner with ToolFlow loop
#     code_refine_dict
#
# P5 / Agent5: Response Refiner with ToolFlow loop
#     refine_dict
#
# P6 / Agent6: Contrast / Insight Agent
#     contrast_response_dict
#
# P7 / Agent7: ToolFlow Implementer
#     branch_math_rewrite_dict
#
# P8 / Agent8: Answer Extractor
#     extract_prompt

# ------------------------------------------------------------------------------------------------------------------------
# Logger

class Logger:
    def __init__(self, args):
        self.args = args
        self.logs=[]
        self.start_time=self.get_current_time()
        save_model_name=self.args.model
        self.log_path=f"MATH/logs/{save_model_name}/infered/{self.args.experiment_prefix}_{self.start_time}.json"

    def get_current_time(self):
        current_time = datetime.datetime.now()
        return current_time.strftime("%Y-%m-%d-%H:%M")

    def init_sample(self,sample_id,query):
        self.logs.append({"id":sample_id,"query":query,"turns":[]})
        
    def update_turn(self,cur_turn,verify_code,response,result,execute_result,execute_content,feedback,reflection,code_feedback):
        self.logs[-1]['turns'].append({"cur_turn":cur_turn,"verify_code":verify_code,"response":response,"result":result,
         "execute_result":execute_result,"execute_content":execute_content,"feedback":feedback,"reflection":reflection,"code_feedback":code_feedback})
    
    def update_final_response(self,response,result):
        self.logs[-1]["final_response"]=response
        self.logs[-1]["final_result"]=result

    def update_insights(self,insights):
        self.logs[-1]["insights"]=insights

    def save_log(self,):
        log_data={
            "start_time":self.start_time,
            "end_time":self.get_current_time(),
            "args":self.args.__dict__,
            "result":None,
            "logs":self.logs}
        save_file(self.log_path,log_data)

# ------------------------------------------------------------------------------------------------------------------------
# VerifyCodeGenerator

# P1 / Agent1: Reverse Verification Flow Generator
# Generate a mathematical verification flow close to mathematical-tool calls.
gen_verify_code_dict = {
    "system_prompt": r"""You are an expert in reverse reasoning verification. Given a problem, you need to generate a reverse mathematical verification flow for that problem.

[Reverse Reasoning Introduction]
* Reverse reasoning is a method of thinking that starts from the result and verifies the problem backwards. Specifically, it involves:

1. Starting with the given answer, rather than the initial conditions of the problem.
2. Assuming this answer is correct, then conducting reverse deduction based on this assumption and the known conditions in the problem.
3. Through this reverse deduction, checking whether other known conditions or constraints in the problem can be satisfied.
4. If the results of the reverse deduction can satisfy all conditions, then the original answer can be considered correct.

[Requirements]
1. Start the verification flow with the proposed answer and verify it against the original problem.
2. Let the verification flow accept one input, the answer, and derive a True/False verification result.
3. After generating the verification flow name, first write the reverse analysis verification approach as comments, then generate the verification flow.
4. Keep a mathematical reasoning style. Prefer mathematical objects such as sets, predicates, functions, recurrences, equations, inequalities, extrema, counts, sums, and solution sets.
5. Use mathematical-tool-level operations when appropriate: max, min, argmax, argmin, count, sum, solve, simplify, factor, recurrence, DP state, and set construction.
6. Use natural mathematical definitions instead of implementation-level details.
7. Represent full verification targets with complete mathematical objects, especially for all-values, count, maximum, minimum, existence, uniqueness, and infinitely-many problems.
8. Define the mathematical object to be computed before asking for a result.
9. The flow may use mathematical notation and natural-language definitions; Python implementation is handled later by the tool-flow implementer.
10. Use ASCII mathematical notation for robustness:
    - x in S
    - CartesianProduct(A, B)
    - expr mod n = r
    - count(V)
    - max(V), min(V), sum(V)
    - V is nonempty
11. Output only the verification flow.""",

    "input_case_1": r"""Generate a validation flow for an answer to the following question:
The length of one leg of a right triangle is 9 meters. The lengths of the other two sides are consecutive integer numbers of meters. What is the number of meters in the perimeter of the triangle?""",

    "output_case_1": r"""verify_triangle_perimeter(perimeter):
    # Reverse verification approach:
    # 1. Start with the proposed perimeter.
    # 2. Express the other leg and hypotenuse using the perimeter.
    # 3. Define the set of side choices satisfying the original integer, consecutive-side, and Pythagorean conditions.
    # 4. The proposed perimeter is verified iff this set is nonempty.

    ans := perimeter
    leg1 := 9
    S := ans - leg1
    h(l) := S - l

    V(ans) := {l in positive integers :
        h(l) in positive integers
        and abs(h(l) - l) = 1
        and leg1^2 + l^2 = h(l)^2
    }

    Since l + h(l) = S by construction, every l in V(ans) gives a right triangle with perimeter ans.

    Verification passes iff V(ans) is nonempty.""",

    "input_case_2": r"""Generate a validation flow for an answer to the following question(**The verification flow should start with the input answer and use reverse reasoning to validate the correctness of the answer.**):
Which of the following numbers is the smallest? Your answer is the letter in front of the smallest number.

$\bullet$ A. $14.45001$ rounded to the nearest tenth.

$\bullet$ B. $14.54999$ rounded to the nearest tenth.

$\bullet$ C. $14.44999$ rounded to the nearest tenth.""",

    "output_case_2": r"""verify_smallest_number_choice(choice):
    # Reverse verification approach:
    # 1. Start with the proposed choice.
    # 2. Define the rounded value of each option.
    # 3. Define the set of options with the smallest rounded value.
    # 4. The proposed choice is verified iff it belongs to that set.

    ans := choice
    O := {A, B, C}

    r(A) := round_to_nearest_tenth(14.45001)
    r(B) := round_to_nearest_tenth(14.54999)
    r(C) := round_to_nearest_tenth(14.44999)

    m := min {r(o) for o in O}
    V := {o in O : r(o) = m}

    Verification passes iff ans in V.""",

    "input_case_3": r"""Generate a validation flow for an answer to the following question(**The verification flow should start with the input answer and use reverse reasoning to validate the correctness of the answer.**):
In how many ways can the letters of the word ``COPYRIGHT'' be arranged?""",

    "output_case_3": r"""verify_arrangement_count(arrangement_count):
    # Reverse verification approach:
    # 1. Start with the proposed arrangement count.
    # 2. Verify that COPYRIGHT has 9 distinct letters.
    # 3. Define the number of arrangements as the factorial of the number of distinct letters.
    # 4. The proposed count is verified iff it equals this factorial value.

    ans := arrangement_count
    word := COPYRIGHT
    L := set of letters in word

    Check count(L) = 9 and every letter of word occurs exactly once.

    N := 9!

    Verification passes iff ans = N.""",

    "input_template": r"""Generate a validation flow for an answer to the following question(**The verification flow should start with the input answer and use reverse reasoning to validate the correctness of the answer.**):
{query}"""
}

class VerifyCodeGenerator:
    def __init__(self, model,temperature):
        self.init_messages=[
            {"role": "system", "content": gen_verify_code_dict["system_prompt"]},
            {"role": "user", "content": gen_verify_code_dict["input_case_1"]},
            {"role": "assistant", "content": gen_verify_code_dict["output_case_1"]},
            {"role": "user", "content": gen_verify_code_dict["input_case_2"]},
            {"role": "assistant", "content": gen_verify_code_dict["output_case_2"]},
            {"role": "user", "content": gen_verify_code_dict["input_case_3"]},
            {"role": "assistant", "content": gen_verify_code_dict["output_case_3"]},
        ]
        self.input_template=gen_verify_code_dict["input_template"]
        self.model=model
        self.temperature=temperature

    def gen_verify_code(self,query):
        messages=self.init_messages+[{"role":"user","content":self.input_template.format(query=query)}]
        response=call_llm(model=self.model,messages=messages,temperature=self.temperature)
        return response

# ------------------------------------------------------------------------------------------------------------------------
# LLMCodeExecutor

# P2 / Agent2: Main Verification Flow Executor
# Base prompt and few-shot examples for executing mathematical verification flows.
# The PythonTool-enhanced runtime prompt is stored in this dict as
# python_tool_input_template and python_tool_system_suffix.
execute_verify_code_dict = {
    "system_prompt": r"""You are a verification expert proficient in mathematical reasoning and tool-assisted verification.

Your task is to verify a given mathematical problem and its solution. A data annotator has written a mathematical verification flow for this problem.

The verification flow is mathematical reasoning. It may define sets, predicates, functions, recurrences, equations, inequalities, extrema, counts, sums, or solution sets.

Please proceed as follows:

1. Carefully read the provided problem, solution process, answer, and verification flow.
2. Interpret the verification flow step by step.
3. Before executing, confirm that the flow includes the original conditions, the intended domain, the target mathematical object, and any required completeness condition.
4. Evaluate simple explicit checks directly when reliable.
5. When a step requires reliable computation, symbolic manipulation, set evaluation, recurrence evaluation, dynamic programming, solving equations or inequalities, extrema computation, counting, summation, counterexample search, or completeness checking, write the corresponding local tool flow inside <tool_flow>...</tool_flow>.
6. The <tool_flow> should preserve the mathematical object from the verification flow. Use tool-level operations such as set construction, count, sum, max, min, argmax, argmin, solve, simplify, factor, recurrence evaluation, or emptiness checking when appropriate.
7. For set, count, extremum, existence, uniqueness, or completeness tasks, evaluate the full mathematical object.
8. Use only assumptions, constants, variables, and conditions already stated in the verification flow.
9. Use ASCII mathematical notation inside <tool_flow>:
   - x in S
   - CartesianProduct(A, B)
   - expr mod n = r
   - count(V)
   - max(V), min(V), sum(V)
   - V is nonempty
10. After the tool result is returned, continue the verification and derive the final verification result.

Input information:

[Problem]
{query}

[Solution Process]
{response}

[Solution Answer]
{result}

[Verification Flow]
{validate_response_fuc}

Please output exactly in the following format:

[Execution of Verification Code]
{Detailed step-by-step execution process}

[Verification Result]
{True or False}""",

    "input_case_1": r"""[Problem]
The product $ab = 1200$, $a$ is an integer, and $b$ is an odd integer. What is the largest possible value of $b$?

[Solution Process]
To solve for the largest possible value of \( b \) given that \( ab = 1200 \), \( a \) is an integer, and \( b \) is an odd integer, we start by factoring 1200.

First, we find the prime factorization of 1200:
\[
1200 = 12 \times 100 = (2^2 \times 3) \times (2^2 \times 5^2) = 2^4 \times 3 \times 5^2
\]

Since \( b \) is an odd integer, it cannot have any factors of 2. Therefore, \( b \) must be composed only of the odd prime factors of 1200, which are 3 and 5.

The prime factorization of 1200 without the factor of 2 is:
\[
3 \times 5^2 = 3 \times 25 = 75
\]

Thus, the largest possible value of \( b \) is 75. We verify this by checking if \( a \) is an integer when \( b = 75 \).

Calculate \( a \) when \( b = 75 \):
\[
a = \frac{1200}{b} = \frac{1200}{75}
\]

Perform the division:
\[
1200 \div 75 = 16
\]

Since \( a = 16 \) is an integer, the conditions are satisfied. Therefore, the largest possible value of \( b \) is \boxed{75}.

[Solution Answer]
75

[Verification Flow]
verify_largest_odd_b(b):
    # Reverse verification approach:
    # 1. Start with the proposed value of b.
    # 2. Define the set of all positive odd divisors of 1200.
    # 3. The proposed answer is verified iff it equals the maximum of that set.

    ans := b
    N := 1200

    V := {d in positive integers :
        d divides N
        and d is odd
    }

    M := max(V)

    Verification passes iff ans = M.""",

    "output_case_1": r"""[Execution of Verification Code]

1. The verification flow is interpreted with ans := 75 and N := 1200.

2. The flow defines the full valid set:
   V := {d in positive integers : d divides 1200 and d is odd}.

3. It then defines M := max(V) and requires ans = M. This correctly represents the original problem, since b must be an odd integer divisor of 1200 and the question asks for the largest possible b.

4. Computing V and max(V) is a finite divisor computation, so I use PythonInterpreterTool through a local tool flow.

<tool_flow>
Evaluate V = {d in positive integers : d divides 1200 and d is odd}.
Return V and max(V).
</tool_flow>

The execution result of PythonInterpreterTool gives max(V) = 75.

5. Since ans = 75 equals max(V), the verification passes.

[Verification Result]
The final result of the execution is:
<result>True</result>""",

    "input_case_2": r"""[Problem]
At the candy store, a chocolate bar costs $c$ dollars and a vanilla bar costs $2$ dollars more than a chocolate bar. Jamie buys a chocolate bar and three vanilla bars, and Kevin buys five chocolate bars. How much money, in total, do Jamie and Kevin spend at the candy store in terms of $c$?

[Solution Process]
Jamie spends a total of $c + 3(c + 2) = 3c + 6$ dollars.
Kevin spends a total of $5c$ dollars.
The total spent by both Jamie and Kevin is $3c + 6 + 5c = 8c + 6$ dollars.

[Solution Answer]
8c + 6

[Verification Flow]
verify_total_spent(total_spent):
    # Reverse verification approach:
    # 1. Start with the proposed total expression.
    # 2. Define the true total expression from the original purchase conditions.
    # 3. The proposed answer is verified iff the symbolic difference is identically zero.

    ans := total_spent

    v(c) := c + 2
    J(c) := c + 3*v(c)
    K(c) := 5*c
    T(c) := J(c) + K(c)

    D(c) := simplify(T(c) - ans)

    Verification passes iff D(c) = 0 identically.""",

    "output_case_2": r"""[Execution of Verification Code]

The verification flow is interpreted with ans := 8c + 6.

1. The vanilla cost is v(c) := c + 2.

2. Jamie's cost is J(c) := c + 3*v(c) = c + 3*(c + 2) = 4c + 6.

3. Kevin's cost is K(c) := 5*c.

4. The true total is T(c) := J(c) + K(c) = 9c + 6.

5. The flow defines D(c) := simplify(T(c) - ans). This symbolic simplification is a suitable tool-level operation.

<tool_flow>
Let c be a symbolic variable.
ans := 8*c + 6
v(c) := c + 2
J(c) := c + 3*v(c)
K(c) := 5*c
T(c) := J(c) + K(c)
Return simplify(T(c) - ans).
</tool_flow>

The execution result of PythonInterpreterTool is c, which is a nonzero expression.

6. Therefore T(c) is different from ans. The proposed solution answer fails verification.

[Verification Result]
The final result of the execution is:
<result>False</result>""",

    "input_template": r"""[Problem]
{query}

[Solution Process]
{response}

[Solution Answer]
{result}

[Verification Flow]
{validate_response_fuc}""",

    "python_tool_input_template": r"""[Problem]
{query}

[Solution Process]
{response}

[Solution Answer]
{result}

[Verification Flow]
{validate_response_fuc}

Execute the verification flow step by step. The verification flow is mathematical reasoning.
Use PythonInterpreterTool when a mathematical object needs reliable evaluation.
Wrap the mathematical tool flow with <tool_flow> and </tool_flow>.
The <tool_flow> should preserve the mathematical object from the verification flow, such as a set, count, sum, max, min, argmax, argmin, solve target, simplify target, recurrence, or emptiness check.
For set, count, extremum, and completeness verification, evaluate the full mathematical object.
Use ASCII mathematical notation: x in S, CartesianProduct(A, B), expr mod n = r, count(V), max(V), and V is nonempty.
End with <result>True</result> or <result>False</result>.""",

    "python_tool_system_suffix": r"""

You may use PythonInterpreterTool as a mathematical verification workbench.
It can support exact arithmetic, symbolic algebra, finite set construction, counting, summation, max/min or argmax/argmin over defined domains, recurrence or DP evaluation, equation solving, inequality checking, counterexample search, and completeness checks.
When such reliable computation is needed, write the local mathematical tool flow in <tool_flow>...</tool_flow>.
The tool flow will be implemented as executable Python before PythonInterpreterTool execution.
After a tool call succeeds, do not issue an equivalent <tool_flow> again. If it fails, revise the <tool_flow> based on the error before retrying.
A tool flow should preserve the mathematical object from the verification flow and evaluate the full set, count, extremum, recurrence, symbolic target, or completeness condition when required.
Use ASCII mathematical notation in tool flows.
After tool results are returned, continue the verification and end with <result>True</result> or <result>False</result>."""
}

def split_content_result(response):
    if "[Verification Result]" in response:
        content,result=response.split("[Verification Result]")
        content=content.strip()
        result=result.strip()
        if "true" in result.lower():
            result=True
        else:
            result=False
        return content,result
    else:
        return response,True

def call_llm_with_tool_flow_loop(
    model,
    messages,
    temperature,
    tool_runner,
    caller,
    final_instruction,
):
    if tool_runner is None:
        return call_llm(
            model=model,
            messages=messages,
            temperature=temperature,
        )

    return tool_runner(
        model=model,
        messages=messages,
        temperature=temperature,
        caller=caller,
        final_instruction=final_instruction,
    )

class PythonToolRequiredExecutor:
    # 2-1_branchMath must be run through the PythonTool test entrypoint.
    # The old pure-LLM simulated execution path is intentionally disabled to
    # avoid confusing it with the actual PythonInterpreterTool experiment.
    def __init__(self,model,temperature):
        raise RuntimeError(
            "Legacy pure-LLM verification execution is disabled for 2-1_branchMath. "
            "Run 2-1_branchMath through repro/scripts/math_hard_test_parallel_python_tool.py, "
            "which uses PythonToolLLMCodeExecutor and PythonInterpreterTool."
        )

    def execute(self,query,response,result,validate_response_fuc):
        raise RuntimeError("Legacy pure-LLM verification execution is disabled for 2-1_branchMath.")

# ------------------------------------------------------------------------------------------------------------------------
# Gen Feedback

# P3 / Agent3: Feedback Summarizer
# Convert verification flow execution process into natural-language feedback.
feedback_dict = {
    "input_case_1": r"""Extract key information from the execution process of the verification flow and convert it into natural language form.

[Problem]
In right triangle $EAD$ with right angle at $A$, $AE=4$ units, $AB=BC=CD$ and the area of triangle $ABE=6$ sq units. What is the length of segment $CE$? Express your answer as a decimal to the nearest tenth.

[Execution of Verification Code]
The predicted length of segment \( CE \) is 9.4.

The verification flow reconstructs AB from the area condition:
Since area_ABE = 6 and AE = 4, AB = 2*6/4 = 3.

Since AB = BC = CD, AD = 3*AB = 9.

The flow defines the required length from the original right triangle:
CE := sqrt(AE^2 + AD^2).

This computation is evaluated by a local tool flow:
<tool_flow>
Evaluate CE = sqrt(4^2 + 9^2).
Return CE exactly and rounded to the nearest tenth.
</tool_flow>

The execution result of PythonInterpreterTool is sqrt(97), which rounds to 9.8.

Since 9.8 differs from the proposed value 9.4, the verification fails.

[Verification Result]
The final result of the execution is:
<result>False</result>""",

    "output_case_1": r"""The verification flow shows that the given length 9.4 differs from the value derived from the original problem conditions. From the area of triangle ABE and AE = 4, we get AB = 3. Since AB = BC = CD, AD = 9. Therefore CE = sqrt(4^2 + 9^2) = sqrt(97), which rounds to 9.8 to the nearest tenth. Since 9.8 differs from 9.4, the original answer fails verification.""",

    "input_case_2": r"""Extract key information from the execution process of the verification flow and convert it into natural language form.

[Problem]
Evaluate $\sqrt{5\cdot10\cdot14\cdot21\cdot15\cdot20}$.

[Execution of Verification Code]
The predicted answer is \(1050\sqrt{2}\).

The verification flow defines the product under the square root:
P := 5*10*14*21*15*20.

A local tool flow evaluates this product:
<tool_flow>
Evaluate P = 5*10*14*21*15*20.
Return P.
</tool_flow>

The execution result of PythonInterpreterTool is 4410000.

The verification flow compares the square of the proposed answer with P:
<tool_flow>
Let ans = 1050*sqrt(2).
Return simplify(ans^2).
</tool_flow>

The execution result of PythonInterpreterTool is 2205000.

Since 2205000 differs from 4410000, the proposed answer fails the square-root verification.

[Verification Result]
The final result of the execution is:
<result>False</result>""",

    "output_case_2": r"""The verification flow shows that the predicted answer \(1050\sqrt{2}\) is inconsistent with the original square-root expression. The exact product under the square root is 4410000, while squaring the predicted answer gives 2205000. Since these values differ, the predicted answer fails verification.""",

    "input_template": r"""Extract key information from the execution process of the verification flow and convert it into natural language form.

[Problem]
{query}

[Execution of Verification Code]
{execute_content}
"""
}

class Feebacker:
    def __init__(self,model,temperature):
        self.init_messages=[
            {"role":"user","content":feedback_dict["input_case_1"]},
            {"role":"assistant","content":feedback_dict["output_case_1"]},
            {"role":"user","content":feedback_dict["input_case_2"]},
            {"role":"assistant","content":feedback_dict["output_case_2"]},
        ]
        self.input_template=feedback_dict["input_template"]
        self.model=model
        self.temperature=temperature

    def __call__(self, query,response,execute_code,execute_result,execute_content):
        execute_content=execute_content.replace("[Execution of Verification Code]","").strip()
        execute_content=execute_content.replace("[Execution of Verification Flow]","").strip()
        execute_content = maybe_compress_text(
            stage="P3.execute_content",
            text=execute_content,
            fixed_messages=self.init_messages + [{
                "role": "user",
                "content": self.input_template.format(
                    query=query,
                    response=response,
                    execute_content="",
                ),
            }],
        )
        input_content=self.input_template.format(query=query,
                                                response=response,
                                                execute_content=execute_content)
        messages=self.init_messages+[{"role":"user","content":input_content}]
        feedback=call_llm(model=self.model,messages=messages,temperature=self.temperature)
        feedback=feedback.strip()
        return feedback

# ------------------------------------------------------------------------------------------------------------------------
# VerifyCodeRefiner

# P4 / Agent4: Verification Flow Refiner
# Reflect on and revise mathematical verification flows when needed.
code_refine_dict = {
    "system_prompt": r"""Your task is to reflect on whether a verification flow is correct.

[Reverse Reasoning Introduction]
* Reverse reasoning is a method of thinking that starts from the result and verifies the problem backwards. Specifically, it involves:

1. Starting with the given answer, rather than the initial conditions of the problem.
2. Assuming this answer is correct, then conducting reverse deduction based on this assumption and the known conditions in the problem.
3. Through this reverse deduction, checking whether other known conditions or constraints in the problem can be satisfied.
4. If the results of the reverse deduction can satisfy all conditions, then the original answer can be considered correct.

Given a problem [Problem], a reverse reasoning validation expert generated a mathematical verification flow [Initial Verification Flow]. However, the verification flow did not pass when verifying a solution [Solution].

The error may come from the solution or from the verification flow.

You need to carefully analyze and complete the following tasks:

1. Reflect on the initial verification flow:
   - Treat the flow as computational mathematical reasoning.
   - Check that the flow includes every original problem condition.
   - Check that the flow defines the full mathematical object required by the problem: a set, predicate, function, recurrence, count, sum, maximum, minimum, solution set, or infinite family.
   - Check that candidate values are justified by a complete set, predicate, or extremum definition.
   - Check that the intended domain is preserved, including positive/nonnegative, integer/real, finite/infinite, and ordered/unordered distinctions.
   - Check that any tool-flow step preserves a full mathematical object, such as a full set, count, max/min, argmin, recurrence, solve target, or completeness check.

2. Provide a new verification flow:
   - If the initial verification flow is correct, repeat it in the new verification flow section.
   - If errors or inadequacies are found, revise it into a stronger mathematical verification flow.
   - Keep the style close to mathematical-tool calls, using sets, predicates, functions, recurrences, equations, inequalities, extrema, counts, sums, and solution sets.
   - Use mathematical-tool-level expressions such as count(V), max(V), min(V), sum(V), solve(...), simplify(...), and recurrence evaluation.
   - Use ASCII mathematical notation: x in S, CartesianProduct(A, B), expr mod n = r, count(V), max(V), and V is nonempty.

When reliable computation is needed to diagnose the verification flow, write a local <tool_flow>...</tool_flow> in the [Reflection] section.

The <tool_flow> should preserve the mathematical object related to the verification flow, such as a full set, count, maximum, minimum, recurrence, symbolic expression, or completeness condition.

After a tool call succeeds, do not issue an equivalent <tool_flow> again. If it fails, revise the <tool_flow> based on the error before retrying.

Tool-flow usage example for verification-flow diagnosis:
<tool_flow>
Evaluate V = {(A,B) in CartesianProduct({0,1,2,3,4,5,6,7,8,9}, {0,1,2,3,4,5,6,7,8,9}) :
    (2 + 5 + A + B) mod 9 = 0}.
Return V and count(V).
</tool_flow>

After the tool result is returned, continue the reflection and produce the final output in the required format:
[Reflection]
...
[New Verification Flow]
...

Please strictly output your analysis and revision according to the following format:

[Reflection]
{Detailed reflection process}

[New Verification Flow]
{Complete verification flow based on the reflection}""",

    "input_case_1": r"""Reflect on the initial verification flow and provide a new verification flow.
(Note: If the verification flow is correct, you also need to repeat the initial verification flow word for word in the new verification flow section [New Verification Flow].)

[Problem]
The four-digit number $25AB$ is divisible by nine, with $A$ the tens digit and $B$ the units digit. How many different such four-digit numbers could $25AB$ represent?

[Initial Verification Flow]
verify_divisible_by_nine_count(count):
    # Reverse verification approach:
    # 1. Start with the proposed count.
    # 2. Define the full digit-pair domain.
    # 3. Define the valid digit pairs using the divisibility-by-9 rule.
    # 4. The proposed count is verified iff it equals the size of this valid set.

    ans := count
    D := {0,1,2,3,4,5,6,7,8,9}

    V := {(A,B) in CartesianProduct(D, D) :
        (2 + 5 + A + B) mod 9 = 0
    }

    N := count(V)

    Verification passes iff ans = N.

[Solution]
To solve this problem, we need to understand the divisibility rule for 9. A number is divisible by 9 if the sum of its digits is divisible by 9.

For the four-digit number \(25AB\), the sum of the digits is:
\[
2 + 5 + A + B = 7 + A + B
\]

We require that this sum be divisible by 9. Therefore, we need:

\[
7 + A + B \equiv 0 \pmod{9}
\]

This implies:

\[
A + B \equiv -7 \equiv 2 \pmod{9}
\]

Thus, we need:

\[
A + B \equiv 2 \pmod{9}
\]

Each value of \(A\) from 0 to 9 gives a unique corresponding value of \(B\), resulting in 10 valid pairs \((A, B)\).

Thus, there are \(\boxed{10}\) different four-digit numbers \(25AB\) that are divisible by 9.

[Verification Process]
The verification flow is interpreted with ans := 10.

It defines D = {0,1,2,3,4,5,6,7,8,9} and V = {(A,B) in CartesianProduct(D, D) : (2 + 5 + A + B) mod 9 = 0}.

<tool_flow>
Evaluate V = {(A,B) in CartesianProduct({0,1,2,3,4,5,6,7,8,9}, {0,1,2,3,4,5,6,7,8,9}) :
    (2 + 5 + A + B) mod 9 = 0}.
Return V and count(V).
</tool_flow>

The tool result gives count(V) = 11.

Since the proposed answer is 10 but count(V) = 11, the verification fails.

[Verification Result]
The final result of the execution is:
<result>False</result>""",

    "output_case_1": r"""[Reflection]
Upon reviewing the initial verification flow, it correctly represents the original problem.

1. It starts with the proposed count.
2. It defines the full digit-pair domain CartesianProduct(D, D), where A and B are both digits from 0 to 9.
3. It defines the valid set V using the original divisibility-by-9 condition for the number 25AB.
4. It defines N := count(V) and compares the proposed answer with N.

The flow defines the full set of valid digit pairs and therefore checks completeness. The failure comes from the solution, since the complete count is 11 while the solution claims 10.

[New Verification Flow]
verify_divisible_by_nine_count(count):
    # Reverse verification approach:
    # 1. Start with the proposed count.
    # 2. Define the full digit-pair domain.
    # 3. Define the valid digit pairs using the divisibility-by-9 rule.
    # 4. The proposed count is verified iff it equals the size of this valid set.

    ans := count
    D := {0,1,2,3,4,5,6,7,8,9}

    V := {(A,B) in CartesianProduct(D, D) :
        (2 + 5 + A + B) mod 9 = 0
    }

    N := count(V)

    Verification passes iff ans = N.""",

    "input_case_2": r"""Reflect on the initial verification flow and provide a new verification flow.

[Problem]
Amber, Micah, and Ahito ran 52 miles in total. Amber ran 8 miles. Micah ran 3.5 times what Amber ran. How many miles did Ahito run?

[Initial Verification Flow]
verify_ahito_miles(ahito_miles):
    # Reverse verification approach:
    # 1. Start with the proposed miles run by Ahito.
    # 2. Define Micah's miles from Amber's miles.
    # 3. Define the value forced by the total-distance condition.
    # 4. The proposed answer is verified iff it equals this forced value.

    ans := ahito_miles
    total := 52
    amber := 8
    micah := 3.5 * amber

    H := total - amber - micah

    Verification passes iff ans = H.

[Solution]
Micah ran 8 x 3.5 = <<8*3.5=28>>28 miles.
So, Ahito ran 52 - 8 - 28 = <<52-8-28=16>>16 miles. Answer: \boxed{16}.

[Verification Process]
The verification flow is interpreted with ans := 16.

It defines micah := 3.5 * 8 = 28 and H := 52 - 8 - 28 = 16.

Since ans = H, the verification result is True.""",

    "output_case_2": r"""[Reflection]
The initial verification flow accurately checks whether the solution for Ahito's miles is correct.

1. It starts with the proposed answer.
2. It correctly encodes the total distance, Amber's distance, and Micah's relation to Amber's distance.
3. It defines the value H forced by the total-distance condition.
4. It compares the proposed answer with H.

For ans = 16, H = 52 - 8 - 3.5*8 = 16. The verification flow is complete and no revision is needed.

[New Verification Flow]
verify_ahito_miles(ahito_miles):
    # Reverse verification approach:
    # 1. Start with the proposed miles run by Ahito.
    # 2. Define Micah's miles from Amber's miles.
    # 3. Define the value forced by the total-distance condition.
    # 4. The proposed answer is verified iff it equals this forced value.

    ans := ahito_miles
    total := 52
    amber := 8
    micah := 3.5 * amber

    H := total - amber - micah

    Verification passes iff ans = H.""",

    "input_template": r"""Reflect on the initial verification flow and provide a new verification flow.
(Note: If the verification flow is correct, you also need to repeat the initial verification flow word for word in the new verification flow section [New Verification Flow].)

[Problem]
{query}

[Initial Verification Flow]
{verify_code}

[Solution]
{response}

[Verification Process]
{verify_process}
"""
}

class CodeRefiner:
    def __init__(self, model, query, temperatue, tool_runner=None):
        self.init_messages=[
            {"role":"system","content":code_refine_dict["system_prompt"]},
            {"role":"user","content":code_refine_dict["input_case_1"]},
            {"role":"assistant","content":code_refine_dict["output_case_1"]},
            {"role":"user","content":code_refine_dict["input_case_2"]},
            {"role":"assistant","content":code_refine_dict["output_case_2"]},
        ]
        self.template = code_refine_dict["input_template"]
        self.model = model
        self.query = query
        self.memory = []
        self.temperatue = temperatue
        self.tool_runner = tool_runner

    def extract_new_solution(self,new_response):
        if "[New Verification Flow]" in new_response:
            return (
                new_response.split("[New Verification Flow]")[0].strip(),
                new_response.split("[New Verification Flow]")[1].strip(),
            )
        if "[New Verification Code]" in new_response:
            return (
                new_response.split("[New Verification Code]")[0].strip(),
                new_response.split("[New Verification Code]")[1].strip(),
            )
        return "",new_response

    def refine(self, response,verify_code,verify_process):
        verify_process=verify_process.replace("[Execution of Verification Code]","").strip()
        verify_process=verify_process.replace("[Execution of Verification Flow]","").strip()
        verify_process = maybe_compress_text(
            stage="P4.verify_process",
            text=verify_process,
            fixed_messages=self.init_messages + [{
                "role": "user",
                "content": self.template.format(
                    query=self.query,
                    response=response,
                    verify_code=verify_code,
                    verify_process="",
                ),
            }],
        )
        input_context = self.template.format(query=self.query,response=response,verify_code=verify_code,verify_process=verify_process)
        messages=self.init_messages+[{"role":"user","content":input_context}]

        final_instruction = (
            "Please now produce the final output in the required format:\n\n"
            "[Reflection]\n"
            "{Detailed reflection process}\n\n"
            "[New Verification Flow]\n"
            "{Complete verification flow based on the reflection}"
        )

        output = call_llm_with_tool_flow_loop(
            model=self.model,
            messages=messages,
            temperature=self.temperatue,
            tool_runner=self.tool_runner,
            caller="P4",
            final_instruction=final_instruction,
        )

        output = output.strip()
        reflection,new_code=self.extract_new_solution(output)
        if "```python" in new_code:
            new_code=new_code.replace("```python","")
        if "```" in new_code:
            new_code=new_code.replace("```","")
        return reflection,new_code


# ------------------------------------------------------------------------------------------------------------------------
# ResponseRefiner

# P5 / Agent5: Response Refiner
# Reflect on and revise the solution response using feedback.
refine_dict={
    "system_prompt": r"""Your task is to reflect on whether a solution is correct.

Given a problem [Problem], a reverse reasoning validation expert generated a mathematical verification flow [Initial Verification Flow] of this problem. The verification flow produced feedback for a solution [Solution].

Given a problem, a corresponding solution, and feedback from a verification flow, carefully compare the solution and the feedback.

You need to carefully analyze and complete the following tasks:
1. Reflect on the initial solution process:
   - Compare in detail the solution and the feedback.
   - Examine the logic and accuracy of the solution approach step by step.
   - Consider whether the feedback supports a revision to the solution or reveals a weakness in the verification flow.
   - Verification feedback can be incomplete when the verification flow is incomplete, too narrow, or checks the wrong mathematical object.
   - If errors or inadequacies are found, provide detailed feedback and suggestions for improvement.
   - If the solution is correct and the feedback is caused by a weak verification flow, state that the solution is correct.
2. Provide a new solution:
   - If the initial solution is correct, use the original solution process.
   - If errors or inadequacies are found, revise the solution based on your reflection.

When reliable constructive computation is needed to revise the solution, write a local <tool_flow>...</tool_flow> in the [Reflection] section.

The <tool_flow> should compute a candidate answer or a key mathematical object from the original problem conditions and the verification feedback.

Use ASCII mathematical notation in constructive tool flows:
- x in S
- CartesianProduct(A, B)
- expr mod n = r
- count(V)
- max(V), min(V), sum(V)
- V is nonempty

Tool-flow usage example for constructive solution repair:
<tool_flow>
Let c be a symbolic variable.
Compute T(c) = c + 3*(c + 2) + 5*c.
Return simplify(T(c)).
</tool_flow>

After a tool call succeeds, do not issue an equivalent <tool_flow> again. If it fails, revise the <tool_flow> based on the error before retrying.

After the tool result is returned, continue the reflection and produce the final output in the required format:
[Reflection]
...
[New Solution]
...

Note:
- Always provide a new solution section.

Please strictly output your analysis and revision according to the following format:

[Reflection]
{{Detailed reflection process}}

[New Solution]
{{Complete solution process based on the reflection}}""",

    "input_case_1": "Reflect on the initial solution and provide a new solution.\n\n[Problem]\nEvaluate $\\sqrt{5\\cdot10\\cdot14\\cdot21\\cdot15\\cdot20}$.\n\n[Initial Solution]\n(Remember, feedback is not always be correct.)\nTo evaluate \\(\\sqrt{5 \\cdot 10 \\cdot 14 \\cdot 21 \\cdot 15 \\cdot 20}\\), we start by simplifying the expression inside the square root. We can factor each number into its prime factors:\n\n\\[\n\\begin{align*}\n5 &= 5, \\\n10 &= 2 \\cdot 5, \\\n14 &= 2 \\cdot 7, \\\n21 &= 3 \\cdot 7, \\\n15 &= 3 \\cdot 5, \\\n20 &= 2^2 \\cdot 5.\n\\end{align*}\n\\]\n\nNow, substitute these factorizations into the product:\n\n\\[\n5 \\cdot 10 \\cdot 14 \\cdot 21 \\cdot 15 \\cdot 20 = 5 \\cdot (2 \\cdot 5) \\cdot (2 \\cdot 7) \\cdot (3 \\cdot 7) \\cdot (3 \\cdot 5) \\cdot (2^2 \\cdot 5).\n\\]\n\nCombine all the factors:\n\n\\[\n= 2^3 \\cdot 3^2 \\cdot 5^4 \\cdot 7^2.\n\\]\n\nNow, take the square root of the entire expression:\n\n\\[\n\\sqrt{2^3 \\cdot 3^2 \\cdot 5^4 \\cdot 7^2}.\n\\]\n\nThe square root of a product is the product of the square roots:\n\n\\[\n= \\sqrt{2^3} \\cdot \\sqrt{3^2} \\cdot \\sqrt{5^4} \\cdot \\sqrt{7^2}.\n\\]\n\nSimplify each square root:\n\n\\[\n\\sqrt{2^3} = \\sqrt{2^2 \\cdot 2} = 2 \\sqrt{2},\n\\]\n\\[\n\\sqrt{3^2} = 3,\n\\]\n\\[\n\\sqrt{5^4} = 5^2 = 25,\n\\]\n\\[\n\\sqrt{7^2} = 7.\n\\]\n\nCombine these results:\n\n\\[\n= (2 \\sqrt{2}) \\cdot 3 \\cdot 25 \\cdot 7.\n\\]\n\nCalculate the product of the integers:\n\n\\[\n3 \\cdot 25 = 75,\n\\]\n\\[\n75 \\cdot 7 = 525,\n\\]\n\\[\n2 \\cdot 525 = 1050.\n\\]\n\nThus, the expression simplifies to:\n\n\\[\n1050 \\sqrt{2}.\n\\]\n\nTherefore, the value of \\(\\sqrt{5 \\cdot 10 \\cdot 14 \\cdot 21 \\cdot 15 \\cdot 20}\\) is \\(\\boxed{1050 \\sqrt{2}}\\).\n\n[Feedback]\nThe problem involves evaluating the expression \\(\\sqrt{5 \\cdot 10 \\cdot 14 \\cdot 21 \\cdot 15 \\cdot 20}\\).\n\n1. **Calculate the Product of the Numbers:**\n   - The product of the numbers is calculated as \\(5 \\times 10 \\times 14 \\times 21 \\times 15 \\times 20\\).\n   - Using a computational tool, the product is found to be 4,410,000.\n\n2. **Square the Given Answer:**\n   - The predicted answer is \\(1050\\sqrt{2}\\).\n   - To verify, we square this expression: \\((1050\\sqrt{2})^2 = 1050^2 \\times 2\\).\n   - Using a computational tool, the squared value is calculated to be 2,205,000.\n\n3. **Verification:**\n   - Compare the squared answer (2,205,000) with the calculated product (4,410,000).\n   - Since 2,205,000 does not equal 4,410,000, the verification fails.\n\n**Conclusion:**\nThe predicted answer \\(1050\\sqrt{2}\\) is incorrect because its square does not match the calculated product of the numbers. The verification process returns a result of \"False.\"",
    "output_case_1": r"""[Reflection]
The initial solution attempts to simplify the expression \(\sqrt{5 \cdot 10 \cdot 14 \cdot 21 \cdot 15 \cdot 20}\) by factoring each number into prime factors. The overall method is appropriate, but there is an error in the factor counting and final simplification.

1. The prime factorizations are:
   - \(5 = 5\)
   - \(10 = 2 \cdot 5\)
   - \(14 = 2 \cdot 7\)
   - \(21 = 3 \cdot 7\)
   - \(15 = 3 \cdot 5\)
   - \(20 = 2^2 \cdot 5\)

2. Combining these factors gives:
   \[
   5 \cdot 10 \cdot 14 \cdot 21 \cdot 15 \cdot 20
   = 2^4 \cdot 3^2 \cdot 5^4 \cdot 7^2.
   \]

3. Therefore,
   \[
   \sqrt{5 \cdot 10 \cdot 14 \cdot 21 \cdot 15 \cdot 20}
   = \sqrt{2^4 \cdot 3^2 \cdot 5^4 \cdot 7^2}
   = 2^2 \cdot 3 \cdot 5^2 \cdot 7.
   \]

4. The product is:
   \[
   2^2 \cdot 3 \cdot 5^2 \cdot 7
   = 4 \cdot 3 \cdot 25 \cdot 7
   = 2100.
   \]

The feedback correctly indicates that the proposed answer \(1050\sqrt{2}\) fails verification, since its square does not match the product under the radical. The solution should be revised to the integer value \(2100\).

[New Solution]
To evaluate
\[
\sqrt{5\cdot10\cdot14\cdot21\cdot15\cdot20},
\]
we factor each term into primes:
\[
5 = 5,\quad
10 = 2\cdot5,\quad
14 = 2\cdot7,\quad
21 = 3\cdot7,\quad
15 = 3\cdot5,\quad
20 = 2^2\cdot5.
\]

Thus,
\[
5\cdot10\cdot14\cdot21\cdot15\cdot20
= 2^4\cdot3^2\cdot5^4\cdot7^2.
\]

Taking the square root gives
\[
\sqrt{2^4\cdot3^2\cdot5^4\cdot7^2}
= 2^2\cdot3\cdot5^2\cdot7.
\]

Now compute:
\[
2^2\cdot3\cdot5^2\cdot7
= 4\cdot3\cdot25\cdot7
= 2100.
\]

Therefore,
\[
\sqrt{5\cdot10\cdot14\cdot21\cdot15\cdot20}
= \boxed{2100}.
\]""",
    "input_case_2": "Reflect on the initial solution and provide a new solution.\n\n[Problem]\nWhat is the perimeter of trapezoid $ABCD$? [asy]\nsize(3inch, 1.5inch);\npair a=(0,0), b=(18,24), c=(68,24), d=(75,0), f=(68,0), e=(18,0);\ndraw(a--b--c--d--cycle);\ndraw(b--e);\ndraw(shift(0,2)*e--shift(2,2)*e--shift(2,0)*e);\nlabel(\"30\", (9,12), W);\nlabel(\"50\", (43,24), N);\nlabel(\"25\", (71.5, 12), E);\nlabel(\"24\", (18, 12), E);\nlabel(\"$A$\", a, SW);\nlabel(\"$B$\", b, N);\nlabel(\"$C$\", c, N);\nlabel(\"$D$\", d, SE);\nlabel(\"$E$\", e, S);\n[/asy]\n\n[Initial Solution]\n(Remember, feedback is not always be correct.)\nTo find the perimeter of trapezoid \\(ABCD\\), we need to determine the lengths of all its sides: \\(AB\\), \\(BC\\), \\(CD\\), and \\(DA\\).\n\nGiven:\n- \\(AB = 30\\)\n- \\(BC = 50\\)\n- \\(CD = 25\\)\n- \\(BE = 24\\)\n\nSince \\(BE\\) is perpendicular to \\(AE\\) and \\(ED\\), \\(AE\\) and \\(ED\\) are horizontal segments. We can find \\(AE\\) and \\(ED\\) using the Pythagorean theorem.\n\nFirst, calculate \\(AE\\):\n\\[\nAE = \\sqrt{AB^2 - BE^2} = \\sqrt{30^2 - 24^2} = \\sqrt{900 - 576} = \\sqrt{324} = 18\n\\]\n\nNext, calculate \\(ED\\):\nSince \\(ED\\) is the remaining horizontal segment from \\(E\\) to \\(D\\), and \\(AD = AE + ED\\), we need to find \\(AD\\) first.\n\nThe total horizontal distance from \\(A\\) to \\(D\\) is the sum of \\(AE\\) and \\(ED\\). Since \\(AD\\) is the base of the trapezoid, we can use the fact that \\(AD = AE + ED\\).\n\nNow, calculate \\(AD\\):\n\\[\nAD = \\sqrt{CD^2 - BC^2} = \\sqrt{25^2 - 50^2} = \\sqrt{625 - 2500} = \\sqrt{0} = 0\n\\]\n\nThis calculation seems incorrect because \\(AD\\) cannot be zero. Let's re-evaluate the problem:\n\nSince \\(BE\\) is perpendicular to \\(AE\\) and \\(ED\\), and \\(BE = 24\\), the horizontal distance from \\(A\\) to \\(D\\) is the sum of \\(AE\\) and \\(ED\\).\n\nGiven that \\(AE = 18\\), and the total horizontal distance from \\(A\\) to \\(D\\) is \\(75\\) (as \\(D\\) is directly above \\(A\\) on the x-axis), we have:\n\\[\nED = 75 - 18 = 57\n\\]\n\nNow, calculate the perimeter of trapezoid \\(ABCD\\):\n\\[\n\\text{Perimeter} = AB + BC + CD + DA = 30 + 50 + 25 + 75 = 180\n\\]\n\nThus, the perimeter of trapezoid \\(ABCD\\) is \\(\\boxed{180}\\).",
    "output_case_2": "[Reflection]\nThe initial solution attempts to calculate the perimeter of trapezoid \\(ABCD\\) by summing up its sides. However, there is a fundamental misunderstanding about the dimensions and configuration of the trapezoid in both the solution and the feedback provided.\n\nUpon carefully reviewing the problem, it appears there were some confusions around the naming and use of the lengths. The mistake lies in both interpreting the problem description and illustrating the steps of calculations. Here's a detailed assessment of what went wrong:\n\n1. The initial solution incorrectly sought the length of \\(AD\\) and \\(ED\\) through a misapplied Pythagorean theorem calculation, which led to an error.\n2. There seemed also to be a misunderstanding about the position of some of the sides when delineating the perimeter.\n3. The feedback, although criticizing the solution, still miscalculated the perimeter with the wrong side lenghs.\n\nWe must reassess and correctly identify all lengths, then simply sum them to find the perimeter of trapezoid \\(ABCD\\).\n\n[New Solution]\n1. Identify each side of the trapezoid as clearly described by the problem:\n    - \\(AB = 30\\) (as indicated in the labels)\n    - \\(BC = 50\\)\n    - \\(CD = 25\\)\n    - \\(DA = 75\\) (since \\(AD = 18 + 57 = 75\\) from the corrected understanding of horizontal arrangement.)\n\n2. Sum all sides to calculate the perimeter:\n   \\[\n   \\text{Perimeter} = AB + BC + CD + DA = 30 + 50 + 25 + 75\n   \\]\n   \\[\n   = 180\n   \\]\n\nThus, the correctly determined perimeter of trapezoid \\(ABCD\\) is \\(\\boxed{180}\\).",
    "input_template": "Reflect on the initial solution and provide a new solution.\n\n[Problem]\n{query}\n\n[Initial Solution]\n(Remember, feedback is not always be correct.)\n{old_response}\n\n[Feedback]\n{feedback}\n"
}

class Refiner:
    def __init__(self,model,query,tool_runner=None):
        self.init_messages=[
            {"role":"system","content":refine_dict["system_prompt"]},
            {"role":"user","content":refine_dict["input_case_1"]},
            {"role":"assistant","content":refine_dict["output_case_1"]},
            {"role":"user","content":refine_dict["input_case_2"]},
            {"role":"assistant","content":refine_dict["output_case_2"]},
        ]
        self.template = refine_dict["input_template"]
        self.model = model
        self.query = query
        self.temperature=0
        self.tool_runner = tool_runner

    def extract_new_solution(self,new_response):
        if "[New Solution]" not in new_response:
            return "",new_response
        else:
            return new_response.split("[New Solution]")[0].strip(),new_response.split("[New Solution]")[1].strip()

    def refine(self, old_response,feedback):
        input_context = self.template.format(query=self.query, old_response=old_response,feedback=feedback)
        messages=self.init_messages+[{"role":"user","content":input_context}]

        final_instruction = (
            "Please now produce the final output in the required format:\n\n"
            "[Reflection]\n"
            "{Detailed reflection process}\n\n"
            "[New Solution]\n"
            "{Complete solution process based on the reflection}"
        )

        last_feedback = ""
        last_new_response = ""
        for attempt in range(1, 5):
            response = call_llm_with_tool_flow_loop(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                tool_runner=self.tool_runner,
                caller="P5",
                final_instruction=final_instruction,
            )

            new_response = str(response or "").strip()
            feedback,new_response=self.extract_new_solution(new_response)
            last_feedback = feedback
            last_new_response = new_response
            if new_response.strip():
                return feedback,new_response

            print(f"[Refiner] empty new_response attempt {attempt}/4, model={self.model}", flush=True)
            if attempt < 4:
                time.sleep(3 * attempt)

        return last_feedback,last_new_response

# ------------------------------------------------------------------------------------------------------------------------
# Contrast and Regenerate

# P6 / Agent6: Contrast / Insight Agent
# Compare two responses and extract key solving insights.
contrast_response_dict= {
    "system_prompt": "You are an expert at comparing and extracting key points.\n\nTask: Analyze the differences between two solutions and extract key points\n\nBackground: For the same problem, two solutions have provided different answers. We need to analyze these differences in depth to identify the key aspects of the problem.\n\nSteps:\n1. Carefully read the problem and both solutions\n2. Ignore surface differences in expression, focus on substantial differences in content and method\n3. Compare the core ideas, key steps, and final results of both solutions\n4. Summarize the essence of the problem reflected by these differences and the key points to note when solving\n\nOutput requirements:\n1. Concisely list 1-3 key points\n2. Each point should be specific and helpful for regenerating a better solution\n\nOutput format:\n[Comparative Analysis Process]\n{{Your comparative analysis process}}\n\n[Core Differences in Solutions]\n{{Summarize the differences in solutions based on the comparative analysis process, answer in bullet points}}\n\n[Key Points to Note When Solving the Problem]\n{{Summarize the key points to note when solving the problem based on the differences in solutions, answer in bullet points}}",
    "input_template": "Based on the following information, please analyze the differences and extract key points:\n(The key points should be detailed, not vague.)\n\n[Problem]\n{query}\n\n[Solution 1]\n{response_1}\n\n[Solution 2]\n{response_2}\n\n",
    "input_case_1": "Based on the following information, please analyze the differences and extract key points:\n(The key points should be detailed, not vague.)\n\n[Problem]\nEvaluate $\\sqrt{3\\cdot6\\cdot10\\cdot15\\cdot21\\cdot28}$.\n\n[Solution 1]\nTo evaluate \\(\\sqrt{3 \\cdot 6 \\cdot 10 \\cdot 15 \\cdot 21 \\cdot 28}\\), we start by factoring each of the numbers:\n\n- \\(3\\) is already a prime number.\n- \\(6 = 2 \\cdot 3\\)\n- \\(10 = 2 \\cdot 5\\)\n- \\(15 = 3 \\cdot 5\\)\n- \\(21 = 3 \\cdot 7\\)\n- \\(28 = 2^2 \\cdot 7\\)\n\nNow, let's write the product \\(3 \\cdot 6 \\cdot 10 \\cdot 15 \\cdot 21 \\cdot 28\\) using these factorizations:\n\n\\[\n3 \\cdot (2 \\cdot 3) \\cdot (2 \\cdot 5) \\cdot (3 \\cdot 5) \\cdot (3 \\cdot 7) \\cdot (2^2 \\cdot 7)\n\\]\n\nCombine the factors:\n\n- The factor of \\(2\\) appears \\(1 + 1 + 2 = 4\\) times.\n- The factor of \\(3\\) appears \\(1 + 1 + 1 + 1 = 4\\) times.\n- The factor of \\(5\\) appears \\(1 + 1 = 2\\) times.\n- The factor of \\(7\\) appears \\(1 + 1 = 2\\) times.\n\nThus, the product can be rewritten as:\n\n\\[\n2^4 \\cdot 3^4 \\cdot 5^2 \\cdot 7^2\n\\]\n\nNow, take the square root of the entire expression:\n\n\\[\n\\sqrt{2^4 \\cdot 3^4 \\cdot 5^2 \\cdot 7^2} = \\sqrt{(2^2 \\cdot 3^2 \\cdot 5 \\cdot 7)^2}\n\\]\n\nThe square root of a square is the base itself:\n\n\\[\n2^2 \\cdot 3^2 \\cdot 5 \\cdot 7 = 4 \\cdot 9 \\cdot 5 \\cdot 7\n\\]\n\nCalculate the product:\n\n- \\(4 \\cdot 9 = 36\\)\n- \\(36 \\cdot 5 = 180\\)\n- \\(180 \\cdot 7 = 1260\\)\n\nTherefore, the value of \\(\\sqrt{3 \\cdot 6 \\cdot 10 \\cdot 15 \\cdot 21 \\cdot 28}\\) is \\(\\boxed{1260}\\).\n\n[Solution 2]\nTo evaluate \\(\\sqrt{3 \\cdot 6 \\cdot 10 \\cdot 15 \\cdot 21 \\cdot 28}\\), we start by factoring each number into its prime factors:\n\n\\[\n3 = 3\n\\]\n\n\\[\n6 = 2 \\times 3\n\\]\n\n\\[\n10 = 2 \\times 5\n\\]\n\n\\[\n15 = 3 \\times 5\n\\]\n\n\\[\n21 = 3 \\times 7\n\\]\n\n\\[\n28 = 2^2 \\times 7\n\\]\n\nNow, we multiply all these factorizations together:\n\n\\[\n3 \\cdot 6 \\cdot 10 \\cdot 15 \\cdot 21 \\cdot 28 = 3 \\times (2 \\times 3) \\times (2 \\times 5) \\times (3 \\times 5) \\times (3 \\times 7) \\times (2^2 \\times 7)\n\\]\n\nCombine the factors:\n\n\\[\n= 2^3 \\times 3^4 \\times 5^2 \\times 7^2\n\\]\n\nNow, take the square root of the product:\n\n\\[\n\\sqrt{2^3 \\times 3^4 \\times 5^2 \\times 7^2} = \\sqrt{2^3} \\times \\sqrt{3^4} \\times \\sqrt{5^2} \\times \\sqrt{7^2}\n\\]\n\nCalculate each square root:\n\n\\[\n\\sqrt{2^3} = \\sqrt{8} = 2^{3/2} = 2 \\sqrt{2}\n\\]\n\n\\[\n\\sqrt{3^4} = \\sqrt{81} = 3^2 = 9\n\\]\n\n\\[\n\\sqrt{5^2} = 5\n\\]\n\n\\[\n\\sqrt{7^2} = 7\n\\]\n\nCombine these results:\n\n\\[\n\\sqrt{3 \\cdot 6 \\cdot 10 \\cdot 15 \\cdot 21 \\cdot 28} = (2 \\sqrt{2}) \\times 9 \\times 5 \\times 7\n\\]\n\nCalculate the product of the integers:\n\n\\[\n9 \\times 5 = 45\n\\]\n\n\\[\n45 \\times 7 = 315\n\\]\n\nThus, the expression becomes:\n\n\\[\n2 \\sqrt{2} \\times 315 = 630 \\sqrt{2}\n\\]\n\nTherefore, the value of \\(\\sqrt{3 \\cdot 6 \\cdot 10 \\cdot 15 \\cdot 21 \\cdot 28}\\) is \\(\\boxed{630}\\).",
    "output_case_1": "[Comparative Analysis Process]\nI analyzed both solutions to identify the differences in their approaches and calculations:\n\n1. Both solutions correctly factorized each number into its prime components.\n2. Both solutions attempted to simplify the expression under the square root by combining the prime factors.\n3. The key difference lies in the handling of the square root:\n   - Solution 1 correctly identified that the product under the square root is a perfect square and simplified it directly to an integer.\n   - Solution 2 incorrectly calculated the square root of \\(2^3\\) as \\(2\\sqrt{2}\\) instead of simplifying the entire expression as a perfect square.\n\n[Core Differences in Solutions]\n1. Handling of the square root:\n   - Solution 1: Correctly identified the expression as a perfect square and simplified it to an integer.\n   - Solution 2: Incorrectly handled the square root of \\(2^3\\) and did not recognize the entire expression as a perfect square.\n2. Final result:\n   - Solution 1: Correctly calculated the result as 1260.\n   - Solution 2: Incorrectly calculated the result as \\(630\\sqrt{2}\\).\n\n[Key Points to Note When Solving the Problem]\n1. Verify whether the expression under the square root is a perfect square by checking the exponents of all prime factors.\n2. Simplify the expression under the square root completely before calculating the final result to ensure accuracy.\n3. Recognize that if the product of factors results in a perfect square, the square root should yield an integer, not an expression involving radicals.",
    "input_case_2": "Based on the following information, please analyze the differences and extract key points:\n(The key points should be detailed, not vague.)\n\n[Problem]\nThe trisectors of angles $B$ and $C$ of scalene triangle $ABC$ meet at points $P$ and $Q$, as shown. Angle $A$ measures 39 degrees and angle $QBP$ measures 14 degrees. What is the measure of angle $BPC$?\n\n[asy]unitsize(2cm);\nlabel(\"$B$\",(0,0),W);\n\nlabel(\"$A$\",(1.2,1.5),N);\nlabel(\"$C$\",(1,0),E);\nlabel(\"$Q$\",(.8,.6),N);\nlabel(\"$P$\",(.7,.2),N);\n\ndraw((0,0)--(1.2,1.5)--(1,0)--cycle,linewidth(1));\ndraw((0,0)--(.8,.6)--(1,0),linewidth(1));\ndraw((0,0)--(.7,.2)--(1,0),linewidth(1));\n[/asy]\n\n[Solution 1]\nTo solve for the measure of angle \\( BPC \\), we start by analyzing the given information about the triangle and the angles.\n\nWe know:\n- \\( \\angle A = 39^\\circ \\).\n- \\( \\angle QBP = 14^\\circ \\).\n\nSince \\( \\triangle ABC \\) is a triangle, the sum of its angles is \\( 180^\\circ \\). Therefore, we can find the sum of angles \\( B \\) and \\( C \\):\n\n\\[\n\\angle B + \\angle C = 180^\\circ - \\angle A = 180^\\circ - 39^\\circ = 141^\\circ.\n\\]\n\nThe problem involves trisectors of angles \\( B \\) and \\( C \\). Let the angles at \\( B \\) be trisected into \\( x, x, x \\) and at \\( C \\) into \\( y, y, y \\). Thus, we have:\n\n\\[\n3x = \\angle B \\quad \\text{and} \\quad 3y = \\angle C.\n\\]\n\nFrom the equation \\( \\angle B + \\angle C = 141^\\circ \\), we have:\n\n\\[\n3x + 3y = 141^\\circ \\quad \\Rightarrow \\quad x + y = 47^\\circ.\n\\]\n\nThe angle \\( \\angle QBP = 14^\\circ \\) is given. Since \\( Q \\) and \\( P \\) are points on the trisectors, \\( \\angle QBP \\) is the angle between the first trisector of \\( \\angle B \\) and the second trisector of \\( \\angle C \\). Therefore, we have:\n\n\\[\n\\angle QBP = x + y = 47^\\circ.\n\\]\n\nHowever, this contradicts the given \\( \\angle QBP = 14^\\circ \\). Therefore, we need to reconsider the trisector configuration. The correct interpretation is that \\( \\angle QBP = x + y - 33^\\circ \\) (since \\( x + y = 47^\\circ \\) and \\( \\angle QBP = 14^\\circ \\)).\n\nNow, we need to find \\( \\angle BPC \\). The angle \\( \\angle BPC \\) is the angle between the second trisector of \\( \\angle B \\) and the second trisector of \\( \\angle C \\). Therefore, it is:\n\n\\[\n\\angle BPC = 180^\\circ - (x + y) = 180^\\circ - 47^\\circ = 133^\\circ.\n\\]\n\nThus, the measure of angle \\( BPC \\) is \\(\\boxed{133^\\circ}\\).\n\n[Solution 2]\nTo solve this problem, we need to find the measure of angle \\( BPC \\) in the given triangle configuration. We are given that angle \\( A \\) measures 39 degrees and angle \\( QBP \\) measures 14 degrees. The points \\( P \\) and \\( Q \\) are the intersections of the trisectors of angles \\( B \\) and \\( C \\).\n\nFirst, let's denote the angles of triangle \\( ABC \\) as follows:\n- \\( \\angle A = 39^\\circ \\)\n- Let \\( \\angle B = x \\)\n- Let \\( \\angle C = y \\)\n\nSince the sum of angles in a triangle is 180 degrees, we have:\n\\[\nx + y + 39^\\circ = 180^\\circ\n\\]\n\\[\nx + y = 141^\\circ\n\\]\n\nNext, consider the trisectors of angles \\( B \\) and \\( C \\). Since \\( P \\) and \\( Q \\) are points where the trisectors meet, we can express the angles at \\( B \\) and \\( C \\) in terms of their trisectors:\n- The trisectors of \\( \\angle B \\) divide it into three equal parts, so each part is \\( \\frac{x}{3} \\).\n- The trisectors of \\( \\angle C \\) divide it into three equal parts, so each part is \\( \\frac{y}{3} \\).\n\nThe angle \\( QBP \\) is given as 14 degrees. This angle is formed by one trisector of \\( \\angle B \\) and one trisector of \\( \\angle C \\). Therefore, we can express \\( \\angle QBP \\) as:\n\\[\n\\angle QBP = \\frac{x}{3} + \\frac{y}{3} = 14^\\circ\n\\]\n\nNow, we have two equations:\n1. \\( x + y = 141^\\circ \\)\n2. \\( \\frac{x}{3} + \\frac{y}{3} = 14^\\circ \\)\n\nFrom the second equation, multiply through by 3 to clear the fractions:\n\\[\nx + y = 42^\\circ\n\\]\n\nHowever, this seems inconsistent with the first equation. Let's re-evaluate the setup. The correct interpretation should be:\n- The angle \\( QBP \\) is actually the angle between the trisectors, not the sum of the trisectors.\n\nThus, the correct setup should be:\n\\[\n\\frac{x}{3} + \\frac{y}{3} = 14^\\circ\n\\]\n\nNow, solve the system of equations:\n1. \\( x + y = 141^\\circ \\)\n2. \\( \\frac{x}{3} + \\frac{y}{3} = 14^\\circ \\)\n\nMultiply the second equation by 3:\n\\[\nx + y = 42^\\circ\n\\]\n\nThis is incorrect, so let's correct the interpretation:\n- The angle \\( QBP \\) is the angle between the trisectors, not the sum of the trisectors.\n\nRe-evaluate:\n- The angle \\( BPC \\) is the external angle formed by the trisectors.\n\nThe correct interpretation is:\n- The angle \\( BPC \\) is the external angle to \\( \\angle QBP \\).\n\nThus, the measure of angle \\( BPC \\) is:\n\\[\n180^\\circ - 14^\\circ = 166^\\circ\n\\]\n\nTherefore, the measure of angle \\( BPC \\) is \\(\\boxed{166^\\circ}\\).",
    "output_case_2": "[Comparative Analysis Process]\nI analyzed both solutions to identify the differences in their approaches and calculations:\n\n1. Both solutions correctly identified the given angles and the relationship between the angles in triangle \\(ABC\\).\n2. Both solutions attempted to use the properties of trisectors to find the measure of angle \\(BPC\\).\n3. The key difference lies in the interpretation and calculation of the angles formed by the trisectors:\n   - Solution 1 attempted to use the sum of the trisected angles and derived a contradiction, leading to a reevaluation and a final angle of \\(133^\\circ\\).\n   - Solution 2 initially misinterpreted the relationship between the trisectors and the given angle \\(QBP\\), leading to an incorrect intermediate step, but eventually corrected the interpretation to find the external angle \\(BPC\\) as \\(166^\\circ\\).\n\n[Core Differences in Solutions]\n1. Interpretation of angle \\(QBP\\):\n   - Solution 1: Initially assumed \\(QBP\\) was the sum of trisected angles, leading to a contradiction and reevaluation.\n   - Solution 2: Initially miscalculated the sum of trisected angles, but corrected to find the external angle.\n2. Calculation of angle \\(BPC\\):\n   - Solution 1: Concluded \\(BPC\\) as \\(133^\\circ\\) after reevaluation.\n   - Solution 2: Concluded \\(BPC\\) as \\(166^\\circ\\) by considering it as an external angle.\n\n[Key Points to Note When Solving the Problem]\n1. Carefully interpret the geometric configuration and relationships between angles, especially when dealing with trisectors or other subdivisions.\n2. Verify the consistency of derived equations with given conditions to avoid contradictions.\n3. Consider alternative geometric interpretations, such as external angles, when initial calculations lead to inconsistencies.",
    "input_case_3": "Based on the following information, please analyze the differences and extract key points:\n(The key points should be detailed, not vague.)\n\n[Problem]\nWhich of the following is closest to 67.4?\n\nA.      67.332\nB.      67.473\nC.      67.526\nD.      67.445\nE.      67.346\n\nExpress your answer as A, B, C, D, or E.\n\n[Solution 1]\nB. 67.473\n\n[Solution 2]\nTo determine which option is closest to 67.4, we need to calculate the absolute difference between 67.4 and each of the given options. Let's do that:\n\nA. |67.4 - 67.332| = 0.068\nB. |67.4 - 67.473| = 0.073\nC. |67.4 - 67.526| = 0.126\nD. |67.4 - 67.445| = 0.045\nE. |67.4 - 67.346| = 0.054\n\nNow, let's compare these differences:\n\n- A: 0.068\n- B: 0.073\n- C: 0.126\n- D: 0.045\n- E: 0.054\n\nThe smallest difference is 0.045, which corresponds to option D (67.445).\n\nTherefore, the option closest to 67.4 is D.",
    "output_case_3": "[Comparative Analysis Process]\nI analyzed both solutions to identify the differences in their approaches and calculations:\n\n1. Solution 1 directly selected option B without showing any calculations or reasoning.\n2. Solution 2 provided a detailed calculation of the absolute differences between 67.4 and each of the given options, then identified the smallest difference to determine the closest value.\n\n[Core Differences in Solutions]\n1. Methodology:\n- Solution 1: Provided an answer without any supporting calculations or reasoning.\n- Solution 2: Used a systematic approach by calculating the absolute differences and comparing them to find the closest value.\n2. Accuracy:\n- Solution 1: Incorrectly identified option B as the closest value.\n- Solution 2: Correctly identified option D as the closest value based on the smallest absolute difference.\n\n[Key Points to Note When Solving the Problem]\n1. Always calculate the absolute differences when determining which number is closest to a given value to ensure accuracy.\n2. Compare the calculated differences to identify the smallest one, which indicates the closest value.\n3. Provide clear reasoning and calculations to support the chosen answer, ensuring transparency and correctness.",
    "gen_response_with_insights_template": "Answer the following question:\n\n[Problem]\n{query}\n\n[Key Points to Note When Solving the Problem]\n{insights}\n\n[Solution]\n{{Your solution}}\n"
}

class ContrastResponse:
    def __init__(self,model,query,temperature=0) -> None:
        self.init_messages=[
            {"role":"system","content":contrast_response_dict["system_prompt"]},
            {"role":"user","content":contrast_response_dict["input_case_1"]},
            {"role":"assistant","content":contrast_response_dict["output_case_1"]},
            {"role":"user","content":contrast_response_dict["input_case_2"]},
            {"role":"assistant","content":contrast_response_dict["output_case_2"]},
            {"role":"user","content":contrast_response_dict["input_case_3"]},
            {"role":"assistant","content":contrast_response_dict["output_case_3"]},
        ]
        self.contrast_response_template=contrast_response_dict["input_template"]
        self.gen_response_with_insights_template=contrast_response_dict["gen_response_with_insights_template"]
        self.model=model
        self.temperature=temperature
        self.query=query
        self.memory=[]

    def extract_insights(self,output):
        if "[Key Points to Note When Solving the Problem]" in output:
            insights=output.split("[Key Points to Note When Solving the Problem]")[-1].strip()
        else:
            insights=output
        pattern = r'\d+\.\s*(.*)'
        matches = re.findall(pattern, insights)
        self.memory = []  
        for match in matches:
            self.memory.append(match)

    def contrast_response(self,response_1,response_2):
        responses=[response_1,response_2]
        random.shuffle(responses)
        input_content=self.contrast_response_template.format(query=self.query,response_1=responses[0],response_2=responses[1])
        messages=self.init_messages+[{"role":"user","content":input_content}]
        output=call_llm(model=self.model,messages=messages,temperature=self.temperature)
        self.extract_insights(output)
        return output

    def gen_response_with_insights(self):
        insights_str=""
        for i,insight in enumerate(self.memory):
            insights_str+=f"{i+1}. {insight}\n"
        input_content=self.gen_response_with_insights_template.format(query=self.query,insights=insights_str)
        new_response=call_llm(model=self.model,messages=[{"role":"user","content":input_content}],temperature=self.temperature)
        new_response=new_response.strip()
        return new_response
    
    def __call__(self,response_1,response_2):
        contrast_result=self.contrast_response(response_1,response_2)
        new_response=self.gen_response_with_insights()
        return contrast_result,new_response

# ------------------------------------------------------------------------------------------------------------------------
# Dataset

def read_dataset(start,end,shuffle=False):
    def format_sample(id,query,sample):
        return {"id":id,"query":query,"sample":sample}
    data=read_file("MATH/data/all_random_500.jsonl")
    data=[format_sample(id=dp["id"],query=dp["query"],sample=dp) for dp in data]
    if shuffle:
        random.seed(0)
        random.shuffle(data)
    if end==-1:
        return data[start:]
    else:
        return data[start:end]


# ------------------------------------------------------------------------------------------------------------------------
# ProgCo Inference

def get_first_response(model,query,temperature):
    """Get the initial response"""
    messages=[{"role":"user","content":query}]
    response=call_llm(model=model,messages=messages,temperature=temperature)
    return response

class ProgCoAgent:
    def __init__(self,args):
        self.dataset = args.dataset
        self.model=args.model
        self.max_cur_turn=args.max_cur_turn
        self.temperature=args.temperature
        # Disabled placeholder: the real BranchMath executor is injected by
        # repro/scripts/math_hard_test_parallel_python_tool.py as
        # PythonToolLLMCodeExecutor.
        self.llm_code_executor = PythonToolRequiredExecutor(model=self.model,temperature=self.temperature)
        self.feedbacker = Feebacker(model=args.model,temperature=self.temperature)
        self.logger=Logger(args)
        # Reuse the original response
        self.first_response_data=read_file(f"MATH/logs/{self.model}/all_random_500_first_response.jsonl")
        self.first_response_data={item["id"]: item for item in self.first_response_data}


    def inference(self,sample):
        cur_turn=0
        query=sample["query"]
        tool_runner = getattr(self.llm_code_executor, "run_refiner_with_tool_flow_loop", None)

        self.response_refiner=Refiner(
            model=self.model,
            query=query,
            tool_runner=tool_runner,
        )
        self.code_refiner=CodeRefiner(
            model=self.model,
            query=query,
            temperatue=self.temperature,
            tool_runner=tool_runner,
        )
        self.contraster=ContrastResponse(model=self.model,query=query,temperature=self.temperature)
        old_result=None
        old_response=None
        # Initialize the sample log
        self.logger.init_sample(sample_id=sample["id"],query=query)
        while cur_turn < self.max_cur_turn+1:
            print(f"cur_turn:{cur_turn}")
            print("start get response")
            if cur_turn==0:
                # Get the initial response
                # response=get_first_response(model=self.model,query=query,temperature=self.temperature)
                # Reuse the response from the first round previously (to save the cost of re-inferencing)
                response=self.first_response_data[sample["id"]]['response']
                result=self.first_response_data[sample["id"]]['pred']
                # Generate the verification code
                verify_code_generator=VerifyCodeGenerator(model=self.model,temperature=self.temperature)
                verify_code=verify_code_generator.gen_verify_code(query=query)
            else:
                old_response=response
                old_result=result
                response=new_response
                verify_code=new_verify_code
                result=extract_answer(query,response,model=self.model)
            if cur_turn>0:
                if result!=old_result:
                    # contrast insights
                    contrast_result,response=self.contraster(old_response,new_response)
                    result=extract_answer(query,response,model=self.model)
            # Perform the verification
            print(f"start execute verify code")
            execute_result,execute_content=self.llm_code_executor.execute(query=query,response=response,result=result,validate_response_fuc=verify_code)
            verification_unavailable_reason = str(
                getattr(
                    self.llm_code_executor,
                    "verification_unavailable_reason",
                    "",
                ) or ""
            ).strip()
            if verification_unavailable_reason:
                self.logger.update_turn(
                    cur_turn, verify_code, response, result, None,
                    execute_content,
                    feedback=(
                        "backward verification unavailable; "
                        + verification_unavailable_reason
                    ),
                    reflection="", code_feedback="",
                )
                sample_log = self.logger.logs[-1]
                sample_log["turns"][-1][
                    "verification_unavailable_reason"
                ] = verification_unavailable_reason
                sample_log["verification_unavailable_reason"] = (
                    verification_unavailable_reason
                )
                sample_log["stop_reason"] = verification_unavailable_reason
                sample_log["candidate_retained"] = True
                cur_turn += 1
                break
            if execute_result:  # Verification passed
                feedback="verify passed"
                # log
                self.logger.update_turn(cur_turn,verify_code,response,result,execute_result,execute_content,feedback,reflection="",code_feedback="")
                cur_turn+=1
                break
            else:  # Verification not passed
                # Generate feedback and refine
                print(f"start refine")
                feedback_summary=self.feedbacker(query=query,response=response,execute_code=verify_code,execute_result=execute_result,execute_content=execute_content)
                p2_execute_content = execute_content
                reflection,new_response=self.response_refiner.refine(old_response=response,feedback=feedback_summary)
                # code refine
                code_feedback,new_verify_code=self.code_refiner.refine(response=response,verify_code=verify_code,verify_process=p2_execute_content)

                # Refresh execute_content because P4/P5 may have appended PythonTool events through the shared executor.
                execute_content = getattr(self.llm_code_executor, "execute_content", execute_content)

                # log
                self.logger.update_turn(cur_turn,verify_code,response,result,execute_result,execute_content,feedback_summary,reflection,code_feedback)
                cur_turn+=1
        final_response=response
        final_result=result
        self.logger.update_final_response(final_response,final_result)
        self.logger.update_insights(self.contraster.memory)

# ------------------------------------------------------------------------------------------------------------------------
# Main Fuc

def main_fuc(args):
    agent=ProgCoAgent(args)
    data=read_dataset(start=args.start,end=args.end,shuffle=False)
    print("dataset size: ",len(data))
    for i,sample in enumerate(tqdm(data)):
        print("-"*10)
        print("id:",sample["id"])
        agent.inference(sample)
        if i%10==0:
            agent.logger.save_log()
    agent.logger.save_log()

# ------------------------------------------------------------------------------------------------------------------------
# Param

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default="qdd_GPT4o")
    parser.add_argument('--max_cur_turn', type=int, default=3,help='Maximum self-correction rounds (default: 3)')
    parser.add_argument('--start', type=int, default=0,help='Starting index of the evaluation dataset slice (default: 0)')
    parser.add_argument('--end', type=int, default=-1,help='Ending index of the evaluation dataset slice (default: -1 for entire dataset)')
    parser.add_argument('--dataset', type=str, default="MATH")
    parser.add_argument('--temperature', type=float, default=0.0)
    args = parser.parse_args()
    args.experiment_prefix = f"progco_start-{args.start}_end-{args.end}"
    print(vars(args))  # Print all parameters
    return args
                

if __name__ == "__main__":
    args = parse_args()
    main_fuc(args)
