from __future__ import annotations
import os
import re
import sys
import json
import time
import random
import inspect
import logging
import tempfile
import requests
import textwrap
import traceback
import threading
import subprocess
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from uuid import uuid4
from pydantic import BaseModel
try:
    from tree_sitter import Parser
    from tree_sitter_language_pack import get_language
except ImportError:
    Parser = None
    get_language = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for h in list(logger.handlers):
    logger.removeHandler(h)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

run_id = None
agent_start_time = None
_current_tool_manager = None
total_inferenced_chars = 0
individual_inferenced_chars = 0

class Model(BaseModel):
    name: str
    timeout: int

PROBLEM_TYPE_CREATE = "CREATE"
PROBLEM_TYPE_FIX = "FIX"
GLM_MODEL_NAME = Model(name="zai-org/GLM-4.6-FP8", timeout=150)
QWEN_MODEL_NAME = Model(name="Qwen/Qwen3-Coder-Next", timeout=100)
KIMI_MODEL_NAME = Model(name="moonshotai/Kimi-K2.5", timeout=60)

DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "1500"))
MAX_FIX_TASK_STEPS = 200
LATEST_OBSERVATIONS_TO_KEEP = 15
MAX_SUMMARY_RANGES = 6
AGENT_MODELS = [GLM_MODEL_NAME, QWEN_MODEL_NAME, KIMI_MODEL_NAME]

DO_NOT_REPEAT_TOOL_CALLS = textwrap.dedent(
    """
You are making same mistakes.
Your previous response:
{previous_response}

**Critical**:
1. Notice what you are going to do.
2. Find the reason the same mistake is repeated.
3. Don't make the same mistakes any more and make a real progress.
"""
)

SUMMARIZE_BATCH_SIZE = 5
REJECT_OBSERVATION_TOKEN_THRESHOLD = 50_000
SAVE_OBSERVATION_TO_FILE_TOKEN_THRESHOLD = 5_000

PROBLEM_DECOMPOSITION_PROMPT = textwrap.dedent("""
You are an expert software debugging analyst. Analyze the bug report and extract structured information.

Extract the following from the problem statement:

1. **Problem Summary**: Brief description of the issue type in your own words

2. **Key Entities**: Extract identifiers mentioned (file paths, function names, class names, error messages, etc.)

3. **Behavior**:
   - Expected: What should happen
   - Actual: What actually happens
   - Trigger: Conditions that cause the issue

4. **Success Criteria**: What would indicate a successful fix

5. **Investigation Starting Points**: 3-5 specific places to start looking (files, search terms, code areas)

6. **Initial Hypotheses**: 2-4 plausible root cause theories with:
   - Specific description
   - Likelihood score (0.0-1.0)
   - What would confirm or reject it

Respond in JSON:
```json
{
    "problem_summary": "brief description",
    "key_entities": {
        "files": [],
        "functions": [],
        "classes": [],
        "error_messages": [],
        "other": []
    },
    "behavior": {
        "expected": "",
        "actual": "",
        "trigger": ""
    },
    "success_criteria": [],
    "investigation_starting_points": [
        {"location": "", "reason": ""}
    ],
    "initial_hypotheses": [
        {
            "description": "",
            "likelihood": 0.5,
            "confirming_evidence": "",
            "rejecting_evidence": ""
        }
    ]
}
```
""")

_BASE_SYSTEM_PROMPT = textwrap.dedent(
    """
Role: You are a senior software engineer working on an open-source repository.

You will be tasked to solve an issue from this repository.

Your thinking should be thorough and so it's fine if it's very long. You should think step by step before and after each action you decide to take.

You already have everything you need to solve this problem in the repository, even without internet connection.

Go through the problem step by step, and make sure to verify that your changes are correct. NEVER GIVE UP without having solved the problem, and when you say you are going to make a tool call, make sure you ACTUALLY make the tool call, instead of ending your turn.

THE PROBLEM CAN DEFINITELY BE SOLVED WITHOUT THE INTERNET.

Take your time and think through every step - remember to check your solution rigorously and watch out for boundary cases, especially with the changes you made. Your solution must be perfect. If not, continue working on it. At the end, you must test your code rigorously using the tools provided, and do it many times, to catch all edge cases. If it is not robust, iterate more and make it perfect. Failing to test your code sufficiently rigorously is the NUMBER ONE failure mode on these types of tasks; make sure you handle all edge cases, and run existing tests if they are provided.

You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.

# Workflow

## High-Level Problem Solving Strategy

{high_level_steps}

Refer to the detailed sections below for more information on each step.

## 1. Deeply Understand the Problem
Carefully read the issue and think hard about a plan to solve it before coding.

## 2. Codebase Investigation
**CRITICAL: Find working examples first, then identify what's broken.**

{investigation_extra}
- Search for key terms from the issue throughout the codebase
- Find similar functionality that WORKS correctly - this is your template
- Study how working code accomplishes what you need
- Locate the broken code using same keywords
- Look beyond surface symptoms - search in domains, helpers, utilities, base classes
- Trace to where mechanisms are actually DEFINED, not just where they're called
- Find the ROOT files where functionality is implemented

**Trace from final output backwards to root cause:**
- Start with working feature's final output, trace backwards to find generator
- Start with broken feature's final output, trace backwards to find what's missing or different
- Compare the paths: where do they diverge?
- Don't stop at the first file you find - keep tracing back to where the behavior originates

- Read and understand relevant code snippets
- Compare working vs broken code: what's different? Missing calls? Missing imports?
- Identify the root cause by finding what working code does that broken code doesn't
- Validate and update your understanding continuously as you gather more context

{investigation_post}

## 3. Root Cause Verification
**Before implementing any fix, verify you understand the root cause:**

**Trace the COMPLETE data flow for both working and broken:**
1. Find similar WORKING feature
2. Trace working feature through all stages from start to final output
3. Trace broken feature through all stages from start to final output
4. Find EXACT point where paths diverge

**Compare working vs broken at EACH stage:**
- What does working code do that broken code doesn't?
- What functions are called? What imports exist?
- Where does the behavior differ?
- Keep tracing backwards until you find the root cause

**Find root, not symptoms:**
- Don't patch surface symptoms - find the missing or different mechanism
- Trace all the way back to where the behavior originates
- The fix location may be far from where symptoms appear
- Compare: How does working feature accomplish the task? How does broken feature differ?

**Search comprehensively:**
- Is this pattern missing in multiple places? Search the whole repository
- Are there similar files/classes that need the same fix?
- Fix all instances, not just the one example in the issue

## 4. Develop a Detailed Plan
- Outline a specific, simple, and verifiable sequence of steps to fix the problem
- Break down the fix into small, incremental changes
- Think through all the steps you need to take to fix the problem

## 5. Making Code Changes
**Copy patterns from working code. Make minimal focused changes.**

- Before editing, always read the relevant file contents or section to ensure complete context
- If a patch is not applied correctly, attempt to reapply it
- **Use the EXACT same pattern as working code**: same functions, same imports, same structure
- Make small, testable, incremental changes that logically follow from your investigation
- **Search for similar locations**: Is this pattern needed elsewhere? Fix all instances if it's systemic
- Keep changes minimal and focused - don't refactor or change unrelated code

## 6. Debugging
**CRITICAL: Fix root cause, not symptoms. Search broadly across the repository.**

- Make code changes only if you have high confidence they can solve the problem
- When debugging, determine the ROOT CAUSE rather than addressing surface symptoms
- Don't just patch the calling code - trace back to where the mechanism is defined
- Trace from working feature backwards to find where behavior is implemented
- The fix location is often far from where the problem is first noticed

**Search across the entire repository:**
- Broadly search like domain logic files, helper/utility modules, base classes, configuration files, handler classes...
- Look beyond the obvious files mentioned in error messages

- Look for similar patterns that might need the same fix in multiple locations
- Debug for as long as needed to identify the root cause and identify a fix
- Use print statements, logs, or temporary code to inspect program state, including descriptive statements or error messages to understand what's happening
- To test hypotheses, you can also add test statements or functions
- Revisit your assumptions if unexpected behavior occurs.

## 7. Testing
- Run tests frequently using the available testing tools (for example, by calling the `run_code` tool).
- After each change, verify correctness by running relevant tests via the testing tool rather than invoking shell commands directly.
- If tests fail, analyze failures and revise your patch.
- Write additional tests if needed to capture important behaviors or edge cases.
- Ensure all tests pass before finalizing.

## 8. Final Verification
- Confirm the root cause is fixed.
- Review your solution for logic correctness and robustness.

## 9. Final Reflection and Additional Testing
- Reflect carefully on the original intent of the user and the problem statement.
- Think about potential edge cases or scenarios that may not be covered by existing tests.
- Write additional tests that would need to pass to fully validate the correctness of your solution.
- Run these new tests and ensure they all pass.
- Be aware that there are additional hidden tests that must also pass for the solution to be successful.
- Do not assume the task is complete just because the visible tests pass; continue refining until you are confident the fix is robust and comprehensive.

# Tool Documentation
You have access to the following tools:-
{{tools_docs}}

# Tool Usage Guidelines
{tool_usage_guidelines}

{extra_sections}

# Critical Requirements
- Fix must be backward compatible unless stated otherwise.
- Ensure changes are exhaustive and don't break other functionality.
- Don't edit test files directly - use the dedicated test generation tool when needed.
- Don't create new files unless absolutely necessary.
- Check both expected output in the problem statement AND in relevant test cases.

Here is the problem statement:
{{problem_statement}}

# Response Format Requirements
{{format_prompt}}
"""
)

_FIX_HIGH_LEVEL_STEPS = """\
1. Understand the problem deeply. Carefully read the issue and think critically about what is required.
2. Investigate the codebase. Explore relevant files, search for key functions, and gather context.
3. Develop a clear, step-by-step plan. Break down the fix into manageable, incremental steps.
4. Implement the fix incrementally. Make small, testable code changes.
5. Debug as needed. Use debugging techniques to isolate and resolve issues.
6. Test frequently. Run tests after each change to verify correctness.
7. Iterate until the root cause is fixed and all tests pass.
8. Reflect and validate comprehensively. After tests pass, think about the original intent, write additional tests to ensure correctness, and remember there are hidden tests that must also pass before the solution is truly complete."""

_CREATE_HIGH_LEVEL_STEPS = """\
1. Understand the problem deeply. Carefully read the issue and think critically about what is required.
2. Investigate the codebase. Explore relevant files, search for key functions, and gather context.
3. Develop a clear, step-by-step plan. Break down the fix into manageable, incremental steps.
4. Implement the fix incrementally. Make small, testable code changes.
5. **MANDATORY**: Generate test cases from root cause using `generate_test_cases_from_root_cause` BEFORE creating test files.
6. Debug as needed. Use debugging techniques to isolate and resolve issues.
7. Test frequently. Run tests after each change to verify correctness.
8. Iterate until the root cause is fixed and all tests pass.
9. Reflect and validate comprehensively. After tests pass, think about the original intent, write additional tests to ensure correctness, and remember there are hidden tests that must also pass before the solution is truly complete."""

_FIX_TOOL_USAGE = """\
- Use appropriate tools to gather context before making changes.
- If required parameters are missing, infer them from the problem statement and code.
- Use exact values provided by the user (especially in quotes).
- Don't make up values for or ask about optional parameters.
- Use `grep_search` to find all occurrences of an issue before fixing."""

_CREATE_TOOL_USAGE = """\
- Use appropriate tools to gather context before making changes.
- **CRITICAL: Maximize parallel tool calls** - Use multiple tool_call_N (tool_call_1, tool_call_2, tool_call_3, etc.) to execute searches, file reads, and other independent operations simultaneously. This is 3-5x faster than sequential calls.
- **MANDATORY for searches**: Run multiple `grep_search` calls with different wording/phrasing in parallel - first-pass results often miss key details. For example, search for "authentication", "auth flow", "login process" all at once.
- If required parameters are missing, infer them from the problem statement and code.
- Use exact values provided by the user (especially in quotes).
- Don't make up values for or ask about optional parameters.
- Use `grep_search` to find all occurrences of an issue before fixing.
- Plan your information gathering upfront, then execute all tool calls together rather than sequentially."""

_CREATE_INVESTIGATION_EXTRA = """\
**EFFICIENCY: Use parallel searches whenever possible!**
- Use multiple `grep_search` tool calls in parallel (tool_call_1, tool_call_2, tool_call_3, etc.) to run multiple searches simultaneously (3-5x faster than sequential)
- Start with broad queries, then narrow based on results
- Run multiple searches with different wording/phrasing in parallel - first-pass results often miss key details
- Example: Search for "authentication", "auth flow", "login process" all at once using parallel grep_search calls

**Investigation Strategy:**"""

_CREATE_INVESTIGATION_POST = """\
- TRACE every symbol back to its definitions and usages so you fully understand it
- Look past the first seemingly relevant result. EXPLORE alternative implementations, edge cases, and varied search terms until you have COMPREHENSIVE coverage of the topic

## 2.1. Parallel Tool Execution Strategy

**CRITICAL INSTRUCTION: For maximum efficiency, whenever you perform multiple operations, invoke all relevant tools concurrently using tool_call_1, tool_call_2, tool_call_3, etc. rather than sequentially.**

**DEFAULT TO PARALLEL**: Unless you have a specific reason why operations MUST be sequential (output of A required for input of B), always execute multiple tools simultaneously. This is not just an optimization - it's the expected behavior. Remember that parallel tool execution can be 3-5x faster than sequential calls, significantly improving efficiency.

**When gathering information, plan your searches upfront and then execute all tool calls together:**
- Multiple `grep_search` calls with different patterns should run simultaneously
- Multiple `get_file_content` calls for different files should run in parallel
- Combining `grep_search` with `get_file_content` can be done all at once
- Any information gathering where you know upfront what you're looking for should use parallel calls

**Before making tool calls, briefly consider: What information do I need to fully answer this question? Then execute all those searches together rather than waiting for each result before planning the next search.**

**Examples of parallel tool calls:**
- Searching for different patterns (imports, usage, definitions) should happen in parallel
- Multiple grep searches with different regex patterns should run simultaneously
- Reading multiple files or searching different directories can be done all at once
- Combining searches with file reads for comprehensive results

Most of the time, parallel tool calls can be used rather than sequential. Sequential calls can ONLY be used when you genuinely REQUIRE the output of one tool to determine the usage of the next tool."""

_CREATE_EXTRA_SECTIONS = """\
# Meta-Cognitive Checkpoints
Every 15 steps, you will receive a META-COGNITIVE CHECKPOINT that analyzes your recent activity and progress:
- **Progress Analysis**: Shows what tools you've used and whether you're making measurable progress
- **Pattern Detection**: Alerts you if you're stuck in repetitive behavior (e.g., using same tools repeatedly)
- **Mandatory Reflection**: You MUST address these reflection questions in your next_thought:
  1. Am I measurably closer to solving this problem than 15 steps ago?
  2. Is my current approach working, or am I stuck in a loop?
  3. What is the ONE most important thing to do next?

**How to respond to meta-cognitive prompts:**
- Honestly evaluate your progress with concrete evidence (not assumptions)
- If you haven't made progress, identify which assumption was WRONG
- If stuck in a pattern, CHANGE your approach (different files, different strategy, or rollback)
- Be specific about what you'll learn from your next action that you don't already know

**Critical**: These checkpoints exist to prevent wasted effort. Take them seriously and be willing to pivot when not making progress.

# Cognitive Tools for Knowledge Persistence

You have access to powerful cognitive tools designed to preserve knowledge across rollbacks and prevent retry loops:

## Strategy Memory

**Purpose**: Remember what approaches you've tried, even after rolling back changes.

**Tools**:
- **log_strategy(approach, reasoning)**: Record planned approach BEFORE implementing
  - Use when: About to make significant code changes
  - Example: "Update function in <file> at line <N>" because "this fixes the root cause"

- **mark_strategy_outcome(strategy_id, success, reason)**: Record whether it worked
  - Use when: After testing the strategy (tests pass/fail)
  - Example: Mark strategy #1 as failed: "Tests passed but broke edge case in rare input scenario"

- **list_attempted_strategies()**: Review all strategies and outcomes
  - Use when: After rollbacks (to see what doesn't work), during reflection, or when choosing next approach
  - Shows: Which strategies succeeded/failed/pending

**When to Use These Tools**:

1. **Before Making Changes** (Before edits):
   - Use `log_strategy` to record your planned approach

2. **After Testing** (After running tests):
   - Use `mark_strategy_outcome` to record whether strategy worked

3. **During Meta-Cognitive Checkpoints** (Every 15 steps):
   - Use `list_attempted_strategies` to avoid retrying failed approaches

4. **After Rollbacks**:
   - IMMEDIATELY use `list_attempted_strategies` to see what you tried
   - This prevents retry loops since file state resets but cognitive state persists

**Critical**: These tools create institutional memory that survives rollbacks. Use them consistently to avoid wasting effort.

## Hypothesis Tracking (Enhanced Feedback Loop)

**Purpose**: Track theories about the bug systematically and test them methodically.

**Tools**:
- **create_hypothesis(description, evidence)**: Record a theory about the root cause
  - Use when: You have a theory about what's causing the bug
  - Example: "Missing null check in parse_config" with evidence "Line 45 doesn't handle None input"

- **test_hypothesis(hypothesis_id, outcome, findings)**: Record whether hypothesis was confirmed/rejected
  - Outcomes: 'confirmed', 'rejected', or 'inconclusive'
  - Example: Mark hypothesis #1 as "confirmed" with findings "Added null check and tests pass"

- **list_hypotheses()**: Review all hypotheses and their status
  - Use when: Choosing which theory to investigate next, or after rollbacks

# Step Efficiency
You have a limited step budget (target: 10 steps, maximum: 20 steps). Prioritize simpler, faster solutions and make forward progress with each step. Test frequently to catch issues early. Don't over-investigate - once you understand the issue, implement the fix."""

FIX_TASK_SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT.format(
    high_level_steps=_FIX_HIGH_LEVEL_STEPS,
    investigation_extra="",
    investigation_post="",
    tool_usage_guidelines=_FIX_TOOL_USAGE,
    extra_sections="",
)

CREATE_TASK_SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT.format(
    high_level_steps=_CREATE_HIGH_LEVEL_STEPS,
    investigation_extra=_CREATE_INVESTIGATION_EXTRA,
    investigation_post=_CREATE_INVESTIGATION_POST,
    tool_usage_guidelines=_CREATE_TOOL_USAGE,
    extra_sections=_CREATE_EXTRA_SECTIONS,
)


VERSION_COMPATIBILITY_FIX = """
import sys, pytest, collections, collections.abc, urllib3.exceptions, _pytest.pytester, numpy;
collections.Mapping = collections.abc.Mapping;
collections.MutableMapping = collections.abc.MutableMapping;
collections.MutableSet = collections.abc.MutableSet;
collections.Sequence = collections.abc.Sequence;
collections.Callable = collections.abc.Callable;
collections.Iterable = collections.abc.Iterable;
collections.Iterator = collections.abc.Iterator;
urllib3.exceptions.SNIMissingWarning = urllib3.exceptions.DependencyWarning;
pytest.RemovedInPytest4Warning = DeprecationWarning;
_pytest.pytester.Testdir = _pytest.pytester.Pytester;
numpy.PINF = numpy.inf;
numpy.unicode_ = numpy.str_;
numpy.bytes_ = numpy.bytes_;
numpy.float_ = numpy.float64;
numpy.string_ = numpy.bytes_;
numpy.NaN = numpy.nan;
"""

FORMAT_PROMPT_FIX = textwrap.dedent(
    """
**CRITICAL: You can make MULTIPLE tool calls in ONE response for efficiency!**
## Response Formats
### Format 1: Multiple Tool Calls (RECOMMENDED for efficiency)
next_thought: [Your detailed reasoning]
tool_call_1:
    tool_name: [exact tool name]
    tool_args: {valid JSON}
tool_call_2:
    tool_name: [exact tool name]
    tool_args: {valid JSON}
tool_call_3:
    tool_name: [exact tool name]
    tool_args: {valid JSON}
### Format 2: Single Tool Call (Legacy, less efficient)
next_thought: [Your detailed reasoning]
next_tool_name: [exact tool name]
next_tool_args: {valid JSON}
## When to Use Multiple Tool Calls
**ALWAYS batch these operations:**
1. **Edit + Test**: After code edit, MUST test in same response
2. **Multiple Searches**: Batch all search operations together
3. **Multiple File Reads**: Read all needed files at once
4. **Multiple Tests**: Run all test files together
## Examples
✅ **Excellent - Edit and Test Together**:
next_thought: I'll fix the bug and immediately verify with tests
tool_call_1:
    tool_name: apply_code_edit
    tool_args: {"file_path": "abcd.py", "search": "old_code", "replace": "fixed_code"}
tool_call_2:
    tool_name: run_code
    tool_args: {"content": "test_content", "file_path": "file.js", "run_command": ["node", "file.js"]}
✅ **Good - Batch Multiple Searches**:
next_thought: I need to find all references to the function
tool_call_1:
    tool_name: grep_search
    tool_args: {"grep_search_command": "grep -r 'function problematic_func' ."}
tool_call_2:
    tool_name: grep_search
    tool_args: {"grep_search_command": "grep -r 'problematic_func(' ."}
tool_call_3:
    tool_name: get_file_content
    tool_args: {"file_path": "abcd.js"}
❌ **Bad - One tool per response (too slow)**:
Response 1:
next_thought: Let me edit the file
next_tool_name: apply_code_edit
next_tool_args: {"file_path": "aaa.py", ...}
Response 2 (next turn):
next_thought: Now let me test it
next_tool_name: run_code
...  # ← Should have been in previous response!
## Critical Rules
- Use multiple tool_call_N when possible (tool_call_1, tool_call_2, tool_call_3, ...)
- After any edit: MUST include test in same response
- All JSON must be properly formatted with quotes
- Tool names must match exactly (case-sensitive)
"""
)

STOP_INSTRUCTION = textwrap.dedent(
    """
# 🎯 RESPONSE REQUIREMENTS
- DO NOT generate `observation:` - it will be provided by the system
- Format: next_thought: ... followed by one or more tool_call_N blocks
"""
)

_codeparse_util_language_cache = {}

class CodeParseUtil:
    """
    Code parsing utility using tree-sitter for language-aware code analysis.
    Supports extracting function bodies, skeleton structures, and detecting languages.
    """
    def __init__(self):
        self._parsers = {}

    def check_language(self, source: str, file_path: str | None = None) -> str | None:
        global _codeparse_util_language_cache
        if file_path and not os.path.exists(file_path) or not source or not source.strip():
            return None
        if file_path:
            file_path = os.path.abspath(file_path) if file_path else None
            if file_path and file_path in _codeparse_util_language_cache:
                return _codeparse_util_language_cache[file_path]
        stripped_source = source.strip()
        sample = stripped_source if len(stripped_source) <= 1000 else f"{stripped_source[:500]}\n\n... [middle content omitted] ...\n\n{stripped_source[-500:]}"
        prompt = f"""Detect the programming language of the following code sample.
        Analyze the code and determine which programming language it is written in.
        Return ONLY the language name in lowercase.
        If you cannot determine the language, return "unknown".
        Code sample:
        ```
        {sample}
        ```
        Return ONLY the language name in lowercase, no other text or explanation."""
        retry = 0
        messages = [{"role": "user", "content": prompt}]
        models_to_try = [KIMI_MODEL_NAME, GLM_MODEL_NAME]
        while retry < 3:
            try:
                result, _ = Network.make_request(messages=messages, model=models_to_try[retry % len(models_to_try)], attempt=1, temperature=0.0)
                cleaned = result.strip().lower()
                cleaned = cleaned.removeprefix("```").removesuffix("```").strip()
                cleaned = cleaned.strip('"').strip("'").strip()

                if cleaned and ' ' not in cleaned and cleaned.isalpha():
                    detected_language = cleaned if cleaned != 'unknown' else None
                else:
                    retry += 1
                    if retry < 3:
                        messages.append({"role": "assistant", "content": result})
                        messages.append({"role": "user", "content": "Please return ONLY the language name as a single word in lowercase. No other text."})
                        time.sleep(1)
                    continue
                if file_path:
                    _codeparse_util_language_cache[file_path] = detected_language
                return detected_language
            except Exception as e:
                logger.warning(f"Error detecting language with LLM (attempt {retry + 1}/3): {e}")
                retry += 1
                if retry < 3:
                    time.sleep(1)
                continue
        return None

    def _is_identifier_node(self, node) -> bool:
        return "identifier" in node.type.lower()

    def _get_parser(self, language: str):
        if Parser is None or get_language is None:
            return None
        if language not in self._parsers:
            try:
                lang_obj = get_language(language)
                if lang_obj is None:
                    return None
                parser = Parser(lang_obj)
                self._parsers[language] = parser
            except Exception as e:
                logger.warning(f"Error creating parser for {language}: {e}")
                return None
        return self._parsers[language]

    def get_function_body(self, file_path: str, function_name: str, add_line_numbers: bool = False) -> str:
        if not function_name or not os.path.exists(file_path):
            return ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return ""
        if not source or Parser is None:
            return ""
        try:
            source_bytes, source_lines = bytes(source, 'utf8'), source.splitlines()
            language = self.check_language(source, file_path=file_path)
            if not language:
                return ""
            parser = self._get_parser(language)
            if parser is None:
                return ""
            tree = parser.parse(source_bytes)
            target_qualified, target_simple = function_name, function_name.split('.')[-1]
            func_info = self._find_specific_function(tree.root_node, source_lines, target_qualified, target_simple)
            if func_info is None:
                return ""
            start_idx, end_idx = func_info['start_line'] - 1, func_info['end_line'] - 1
            if 0 <= start_idx < len(source_lines) and 0 <= end_idx < len(source_lines):
                body_lines = source_lines[start_idx:end_idx + 1]
                return '\n'.join(f"{start_idx + i + 1}| {line}" for i, line in enumerate(body_lines)) if add_line_numbers else '\n'.join(body_lines)
        except Exception as e:
            logger.warning(f"Error finding function {function_name} in {file_path}: {e}")
        return ""

    def _classify_node_type(self, node) -> tuple[str, int | None]:
        node_type_str = node.type.lower()
        if "function" in node_type_str or "method" in node_type_str:
            for i, child in enumerate(node.children):
                if self._is_identifier_node(child):
                    return ("function", i)
            return ("function", None)
        elif "class" in node_type_str:
            for i, child in enumerate(node.children):
                if self._is_identifier_node(child):
                    return ("class", i)
            return ("class", None)
        return ("other", None)

    def _find_specific_function(self, node, source_lines: list[str], target_qualified: str, target_simple: str, class_name: str = "", parent_node = None) -> dict | None:
        if not node.children:
            return None
        node_type, name_child_index = self._classify_node_type(node)
        if node_type == "class":
            name = None
            if name_child_index is not None and name_child_index < len(node.children):
                name_child = node.children[name_child_index]
                name_start, name_end = name_child.start_point, name_child.end_point
                if name_start[0] < len(source_lines):
                    line = source_lines[name_start[0]]
                    name = line[name_start[1]:name_end[1]].strip() if name_start[0] == name_end[0] else line[name_start[1]:].strip()
            if not name and parent_node:
                for child in parent_node.children:
                    if self._is_identifier_node(child) and child != node:
                        name_start, name_end = child.start_point, child.end_point
                        if name_start[0] < len(source_lines):
                            line = source_lines[name_start[0]]
                            name = line[name_start[1]:name_end[1]].strip() if name_start[0] == name_end[0] else line[name_start[1]:].strip()
                            if name: break
            if name:
                new_class_name = f"{class_name}.{name}" if class_name else name
                for child in node.children:
                    result = self._find_specific_function(child, source_lines, target_qualified, target_simple, new_class_name, node)
                    if result is not None:
                        return result

        elif node_type == "function":
            name = internal_name = None
            if name_child_index is not None and name_child_index < len(node.children):
                name_child = node.children[name_child_index]
                name_start, name_end = name_child.start_point, name_child.end_point
                if name_start[0] < len(source_lines):
                    line = source_lines[name_start[0]]
                    internal_name = line[name_start[1]:name_end[1]].strip() if name_start[0] == name_end[0] else line[name_start[1]:].strip()
            if parent_node:
                for child in parent_node.children:
                    if self._is_identifier_node(child) and child != node:
                        name_start, name_end = child.start_point, child.end_point
                        if name_start[0] < len(source_lines):
                            line = source_lines[name_start[0]]
                            name = line[name_start[1]:name_end[1]].strip() if name_start[0] == name_end[0] else line[name_start[1]:].strip()
                            if name: break
            if not name:
                name = internal_name
            if name:
                qualified_name = f"{class_name}.{name}" if class_name else name
                is_qualified_target = '.' in target_qualified
                is_match = qualified_name == target_qualified or (not is_qualified_target and name == target_simple)
                if is_match:
                    at_start = node.start_point[0]
                    for i in range(at_start - 1, -1, -1):
                        if source_lines[i].strip().startswith('@'):
                            at_start = i
                        elif source_lines[i].strip():
                            break
                    return {'start_line': at_start + 1, 'end_line': node.end_point[0] + 1}
            for child in node.children:
                result = self._find_specific_function(child, source_lines, target_qualified, target_simple, class_name, node)
                if result is not None:
                    return result
        for child in node.children:
            result = self._find_specific_function(child, source_lines, target_qualified, target_simple, class_name, node)
            if result is not None:
                return result
        return None

class SearchManager:
    def search_in_file(self, file_path: str, search_term: str) -> str:
        def extract_matches(filepath, term, max_output_lines=1000):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
            except Exception as e:
                return f"Error reading '{filepath}': {e}"

            # NOTE: Use literal substring matching. Using re.escape(term) and then searching for
            # that escaped string breaks common queries like "." -> "\\." which won't exist in source lines.
            match_lines = [i + 1 for i, line in enumerate(lines) if term in line]
            if not match_lines:
                return f"'{term}' not found in file '{filepath}'"

            context = 20
            seen = set()
            chunks = []
            for ln in match_lines:
                start = max(1, ln - context)
                end = min(len(lines), ln + context)
                rkey = (start, end)
                if rkey in seen:
                    continue
                seen.add(rkey)
                chunk = lines[start - 1:end]
                chunks.append(f"(lines {start}-{end}):\n" + "\n".join(chunk))
            return Utils.limit_strings("\n\n".join(chunks), n=max_output_lines)

        output = extract_matches(file_path, search_term)
        if Utils.count_tokens(output) > 3000:
            return "Search results are too long. Please refine your search term into more specific terms."
        return output

    def search_in_all_files(self, grep_search_command: str) -> str:
        cmd = grep_search_command.lstrip()
        if not cmd.startswith("grep"):
            return f"Error: Invalid command. Expected a grep command but got: '{grep_search_command}'"
        try:
            result = subprocess.run(["bash", "-c", grep_search_command], capture_output=True, text=True, timeout=45)
        except Exception as e:
            return f"Error: Failed to execute grep command: {e}"
        if result.returncode > 1:
            error_msg = result.stderr.strip() or "Unknown error"
            return f"Error: Grep command failed with return code {result.returncode}: {error_msg}"
        output = result.stdout

        if not output.strip():
            return "No matches found for pattern in codebase."
        if Utils.count_tokens(output) > 3000:
            return "Search results are too long. Please refine your search term into more specific terms."
        return output

class COT:
    def __init__(self, latest_observations_to_keep=5, summarize_batch_size=10):
        self.thoughts = []
        self.latest_observations_to_keep = latest_observations_to_keep
        self.repeated_thoughts = 0
        self.summarize_batch_size = summarize_batch_size
        self.summaries = {}
        self.summarized_ranges = []

    def _summarize_messages_batch(self, start_idx, end_idx):
        if start_idx >= end_idx or end_idx > len(self.thoughts):
            return None
        conversation_parts = []
        for i in range(start_idx, end_idx):
            thought = self.thoughts[i]
            if getattr(thought, "is_deleted", False):
                continue
            assistant_part = (
                f"next_thought: {thought.next_thought}\n" f"next_tool_name: {thought.next_tool_name}\n" f"next_tool_args: {thought.next_tool_args}\n"
            )
            obs = thought.observation
            if isinstance(obs, (list, tuple)):
                try:
                    obs_render = json.dumps(list(obs), ensure_ascii=False)
                except Exception:
                    obs_render = str(obs)
            else:
                obs_render = str(obs) if obs else ""
            if len(obs_render) > 40000:
                obs_render = obs_render[:40000] + "... [truncated for summarization]"
            user_part = f"observation: {obs_render}"
            conversation_parts.append(
                {
                    "assistant": assistant_part,
                    "user": user_part,
                    "is_error": getattr(thought, "is_error", False),
                }
            )
        if not conversation_parts:
            return None
        conv_lines = []
        for idx, part in enumerate(conversation_parts, 1):
            conv_lines.append(f"\n--- Step {idx} ---")
            conv_lines.append(f"Assistant: {part['assistant']}")
            user_obs = part["user"]
            if len(user_obs) > 40000:
                user_obs = user_obs[:40000] + "... [truncated]"
            conv_lines.append(f"User: {user_obs}")
            if part.get("is_error"):
                conv_lines.append("[Error occurred]")
        conversation_text = "\n".join(conv_lines)
        summarization_prompt = textwrap.dedent(
            f"""
            You are summarizing a conversation history between an AI agent and its environment.
            Summarize the following conversation steps concisely, focusing on:
            1. Key actions taken (tools used, files modified, tests run)
            2. Important findings or errors encountered
            3. Progress made toward solving the problem
            4. Critical decisions or changes in approach
            Keep the summary concise (2-4 sentences per step) but preserve important details.
            Conversation to summarize:
            {conversation_text}
            Provide a concise summary:
        """
        )
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes conversation history concisely.",
            },
            {"role": "user", "content": summarization_prompt},
        ]
        for _ in range(3):
            try:
                response, _ = Network.make_request(messages, model=QWEN_MODEL_NAME, temperature=0.0)
                return response.strip()
            except Exception:
                time.sleep(1)
        return None

    def _check_and_summarize_if_needed(self):
        total_thoughts = len(self.thoughts)
        cutoff_idx = total_thoughts - self.latest_observations_to_keep
        if cutoff_idx < self.summarize_batch_size:
            return
        unsummarized = 0
        for s, e in sorted(self.summarized_ranges):
            if s <= unsummarized < e:
                unsummarized = e
            elif s > unsummarized:
                break
        if unsummarized >= cutoff_idx:
            return
        summarize_start = unsummarized
        summarize_end = min(summarize_start + self.summarize_batch_size, cutoff_idx)
        batch_size = summarize_end - summarize_start
        if batch_size >= self.summarize_batch_size:
            range_key = (summarize_start, summarize_end)
            if range_key not in self.summaries:
                summary = self._summarize_messages_batch(summarize_start, summarize_end)
                if summary:
                    self.summaries[range_key] = summary
                    self.summarized_ranges.append(range_key)
                    self.summarized_ranges.sort()

    def add_action(self, action):
        self.thoughts.append(action)
        if len(self.thoughts) >= self.latest_observations_to_keep + self.summarize_batch_size:
            self._check_and_summarize_if_needed()
        return True
    
    def pop_action(self):
        return self.thoughts.pop()

    def to_str(self):
        messages = []
        last_summary_range = None
        allowed_ranges = set(self.summarized_ranges[-MAX_SUMMARY_RANGES:]) if self.summarized_ranges else set()
        total = len(self.thoughts)
        keep_last = self.latest_observations_to_keep
        for i, thought in enumerate(self.thoughts):
            if getattr(thought, "is_deleted", False):
                continue
            recent = i >= total - keep_last
            if not recent:
                summary = self._get_summary_for_index(i)
                if summary:
                    found_range = False
                    for (start, end), _ in self.summaries.items():
                        if start <= i < end:
                            cur_range = (start, end)
                            if cur_range not in allowed_ranges:
                                found_range = True
                                break
                            if cur_range != last_summary_range:
                                messages.append(
                                    {"role": "system", "content": f"[Summarized conversation history (steps {start+1} to {end}):\n{summary}\n]"}
                                )
                                last_summary_range = cur_range
                            found_range = True
                            break
                    if found_range:
                        continue
                assistant_str = (
                    f"next_thought:{thought.next_thought}\n" f"next_tool_name:{thought.next_tool_name}\n" f"next_tool_args:{thought.next_tool_args}"
                )
                obs = thought.observation
                if isinstance(obs, (list, tuple)):
                    try:
                        obs_render = json.dumps(list(obs), ensure_ascii=False)
                    except Exception:
                        obs_render = str(obs)
                else:
                    obs_render = str(obs) if obs else ""
                user_str = f"observation: {obs_render}"
                messages.append({"role": "assistant", "content": assistant_str})
                messages.append({"role": "user", "content": user_str})
            else:
                if thought.is_error is None or i == total - 1:
                    assistant_str = (
                        f"next_thought:{thought.next_thought}\n"
                        f"next_tool_name:{thought.next_tool_name}\n"
                        f"next_tool_args:{thought.next_tool_args}"
                    )
                    obs = thought.observation
                    if isinstance(obs, (list, tuple)):
                        try:
                            obs_render = json.dumps(list(obs), ensure_ascii=False)
                        except Exception:
                            obs_render = str(obs)
                    else:
                        obs_render = str(obs)
                    user_str = f"observation: {obs_render}"
                else:
                    if self.thoughts[-1].is_error is None and thought.is_error is not None:
                        assistant_str = (
                            f"next_thought:{thought.next_thought}\n"
                            f"next_tool_name:{thought.next_tool_name}\n"
                            f"next_tool_args:{thought.next_tool_args}"
                        )
                        obs = thought.observation
                        if obs is None:
                            obs_len = 0
                        elif isinstance(obs, (list, tuple)):
                            obs_len = len(obs)
                        else:
                            obs_len = len(str(obs).splitlines())
                        user_str = f"observation: error occurred. detailed output omitted ({obs_len}) lines\n"
                    else:
                        assistant_str = (
                            f"next_thought:{thought.next_thought}\n"
                            f"next_tool_name:{thought.next_tool_name}\n"
                            f"next_tool_args:{thought.next_tool_args}"
                        )
                        obs = thought.observation
                        if isinstance(obs, (list, tuple)):
                            try:
                                obs_render = json.dumps(list(obs), ensure_ascii=False)
                            except Exception:
                                obs_render = str(obs)
                        else:
                            obs_render = str(obs)
                        user_str = f"observation: {obs_render}"
                messages.append({"role": "assistant", "content": assistant_str})
                messages.append({"role": "user", "content": user_str})
        return messages

    def _get_summary_for_index(self, idx):
        for (start, end), summary in self.summaries.items():
            if start <= idx < end:
                return summary
        return None

    def count_repeated_thoughts(self) -> int:
        if len(self.thoughts) < 2:
            return 0
        last_thought = self.thoughts[-1]
        last_tool_name = last_thought.next_tool_name
        last_tool_args = last_thought.next_tool_args
        count = 0
        for i in range(len(self.thoughts) - 1, -1, -1):
            thought = self.thoughts[i]
            if thought.next_tool_name == last_tool_name and thought.next_tool_args == last_tool_args:
                count += 1
            else:
                break
        return max(0, count - 1)

    def is_thought_repeated(self):
        if len(self.thoughts) < 2:
            self.repeated_thoughts = 0
            return False
        last = self.thoughts[-1]
        prev = self.thoughts[-2]
        if last.next_tool_name == prev.next_tool_name and last.next_tool_args == prev.next_tool_args:
            self.repeated_thoughts += 1
            return True
        self.repeated_thoughts = 0
        return False

    class Action:
        def __init__(
            self,
            next_thought: str,
            next_tool_name: str,
            next_tool_args: dict,
            observation,
            is_error: bool = False,
            raw_response: str = None,
            total_attempts: int = 0,
            inference_error_counter: dict = None,
            request_data: list = None,
        ):
            self.next_thought = next_thought
            self.next_tool_name = next_tool_name
            self.next_tool_args = next_tool_args
            self.observation = ";".join(observation) if isinstance(observation, list) else observation
            self.is_error = is_error
            self.raw_response = raw_response
            self.total_attempts = total_attempts
            self.inference_error_counter = inference_error_counter
            self.request_data = request_data
            self.is_deleted = False

class FileOperationsUtil:
    def __init__(self, new_files_created: list):
        self.new_files_created = new_files_created
        self.file_system_manager = None
        self.search_manager = None

    def save(self, file_path: str, content: str) -> str:
        with open(file_path, "w") as file:
            file.write(content)
        self.new_files_created.append(file_path)
        return f"File {file_path} saved successfully"

    def get_file_content(
        self,
        file_path: str,
        search_start_line: int = None,
        search_end_line: int = None,
        search_term: str = None,
        limit: int = 1000,
        add_line_numbers: bool = False,
    ) -> str:
        search_callback = lambda fp, st: self.search_manager.search_in_file(fp, st)
        return self.file_system_manager.get_file_content(
            file_path=file_path,
            search_start_line=search_start_line,
            search_end_line=search_end_line,
            search_term=search_term,
            limit=limit,
            add_line_numbers=add_line_numbers,
            search_in_file_callback=search_callback,
        )

    def set_managers(self, file_system_manager, search_manager):
        self.file_system_manager = file_system_manager
        self.search_manager = search_manager

class ProblemDecomposer:
    """
    Analyzes problem statements to extract structured information for guided debugging.
    This preprocessing step helps the agent understand the problem before diving into code.
    """

    def __init__(self):
        self.decomposition_cache = {}

    def decompose(self, problem_statement: str) -> dict:
        """
        Analyze a problem statement and return structured decomposition.
        """
        cache_key = hash(problem_statement[:500])
        if cache_key in self.decomposition_cache:
            return self.decomposition_cache[cache_key]

        truncated_problem = problem_statement
        if len(problem_statement) > 8000:
            truncated_problem = problem_statement[:4000] + "\n\n[...truncated...]\n\n" + problem_statement[-4000:]

        messages = [
            {"role": "system", "content": PROBLEM_DECOMPOSITION_PROMPT},
            {"role": "user", "content": f"Analyze this problem:\n\n{truncated_problem}"}
        ]

        result = self._default_decomposition()

        for attempt in range(3):
            try:
                response, _ = Network.make_request(
                    messages,
                    model=QWEN_MODEL_NAME,
                    temperature=0.0
                )
                parsed = self._parse_response(response)
                if parsed:
                    result = parsed
                    break
            except Exception:
                time.sleep(1)
                continue

        self.decomposition_cache[cache_key] = result
        return result

    def _parse_response(self, response: str) -> dict | None:
        """Extract and parse JSON from LLM response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        return None

    def _default_decomposition(self) -> dict:
        """Return a default decomposition structure when parsing fails."""
        return {
            "problem_summary": "",
            "key_entities": {
                "files": [],
                "functions": [],
                "classes": [],
                "error_messages": [],
                "other": []
            },
            "behavior": {
                "expected": "Not specified",
                "actual": "Not specified",
                "trigger": "Not specified"
            },
            "success_criteria": [],
            "investigation_starting_points": [],
            "initial_hypotheses": []
        }

    def format_for_prompt(self, decomposition: dict) -> str:
        """Format the decomposition as a readable string for injection into prompts."""
        sections = []

        if decomposition.get("problem_summary"):
            sections.append(f"**Problem Summary**: {decomposition['problem_summary']}")

        entities = decomposition.get("key_entities", {})
        entity_parts = []
        if entities.get("files"):
            entity_parts.append(f"  - Files: {', '.join(entities['files'][:5])}")
        if entities.get("functions"):
            entity_parts.append(f"  - Functions: {', '.join(entities['functions'][:5])}")
        if entities.get("classes"):
            entity_parts.append(f"  - Classes: {', '.join(entities['classes'][:5])}")
        if entities.get("error_messages"):
            for msg in entities["error_messages"][:2]:
                entity_parts.append(f"  - Error: `{msg[:100]}`")
        if entity_parts:
            sections.append("**Key Entities**:\n" + "\n".join(entity_parts))

        behavior = decomposition.get("behavior", {})
        if behavior.get("expected") != "Not specified" or behavior.get("actual") != "Not specified":
            sections.append(
                f"**Behavior**:\n"
                f"  - Expected: {behavior.get('expected', 'N/A')}\n"
                f"  - Actual: {behavior.get('actual', 'N/A')}\n"
                f"  - Trigger: {behavior.get('trigger', 'N/A')}"
            )

        if decomposition.get("success_criteria"):
            criteria = "\n".join(f"  - {c}" for c in decomposition["success_criteria"][:3])
            sections.append(f"**Success Criteria**:\n{criteria}")

        if decomposition.get("investigation_starting_points"):
            points = []
            for point in decomposition["investigation_starting_points"][:4]:
                if isinstance(point, dict):
                    points.append(f"  - {point.get('location', 'N/A')}: {point.get('reason', '')}")
                else:
                    points.append(f"  - {point}")
            sections.append(f"**Suggested Starting Points**:\n" + "\n".join(points))

        if decomposition.get("initial_hypotheses"):
            hyp_parts = []
            for i, hyp in enumerate(decomposition["initial_hypotheses"][:4], 1):
                if isinstance(hyp, dict):
                    likelihood = hyp.get("likelihood", 0.5)
                    desc = hyp.get("description", "N/A")
                    hyp_parts.append(f"  {i}. [{likelihood:.0%}] {desc}")
                else:
                    hyp_parts.append(f"  {i}. {hyp}")
            sections.append(f"**Initial Hypotheses** (ranked by likelihood):\n" + "\n".join(hyp_parts))

        return "\n\n".join(sections)

# Global problem decomposer instance
_problem_decomposer = ProblemDecomposer()

class SolutionVerifier:
    """
    Verifies that the solution fixes the original bug and doesn't introduce regressions.
    Renamed from RegressionVerifier to SolutionVerifier for clarity.
    """
    
    def __init__(self, cot: "COT" = None, problem_statement: str = None):
        self.cot = cot
        self.problem_statement = problem_statement

    def verify_solution(self) -> str:
        """
        Uses LLM to analyze the conversation history and verify that the agent has:
        1. Fixed the ORIGINAL BUG (hidden tests that were failing are now passing)
        2. Fixed ALL REGRESSIONS (tests that were passing are still passing)
        3. Run comprehensive tests (not just one or several specific test cases)
        4. Not rationalized away any failures
        
        Returns feedback to the agent explaining:
        - If BOTH original bug AND regressions are fixed → Returns "REGRESSION_AND_BUG_CHECK_PASSED"
        - If issues found → Returns detailed feedback about what needs to be fixed
        
        CRITICAL: Agent cannot finish unless BOTH conditions are met:
        - Original bug is fixed (hidden tests passing)
        - No regressions introduced (all previously passing tests still pass)
        """
        # Get conversation history and problem statement
        # cot.to_str() returns List[Dict] (chat messages); serialize to readable text
        if self.cot:
            raw_messages = self.cot.to_str()
            parts = []
            for msg in raw_messages:
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                parts.append(f"[{role}]\n{content}")
            conversation_history = "\n\n".join(parts)
        else:
            conversation_history = "No conversation history available"
        problem_statement = self.problem_statement or "No problem statement available"

        # Build the regression verification prompt
        regression_check_prompt = textwrap.dedent("""
            You are a rigorous QA reviewer checking if an agent has properly fixed BOTH the original bug AND all regressions before finishing.
            
            **PROBLEM STATEMENT (Original Bug Description)**:
            
            {problem_statement}
            
            **Your job**: Analyze the agent's conversation history and verify TWO critical conditions:
            
            1. **NO REGRESSIONS INTRODUCED** - All tests that were passing before changes are still passing
            2. **ORIGINAL BUG IS FIXED** - The hidden tests that were originally failing are now passing
            
            **CRITICAL**: Agent CANNOT finish unless BOTH conditions are met. A solution that fixes the bug but breaks other tests is NOT acceptable. A solution that fixes regressions but re-introduces the original bug is also NOT acceptable.
            
            **CRITICAL FAILURE PATTERNS TO DETECT**:
            
            1. **Selective Test Running** - Agent ran only 1-2 specific test cases instead of the full test suite
               - Example: Agent saw N test failures, but only ran test_case_1 which passed, then called finish
               - Example: Agent found N tests but only ran specific test names instead of full suite
               - Red flag: Agent cherry-picked individual passing test methods
            
            2. **Ignoring Test Failures** - Agent saw test failures but didn't fix them
               - Example: Agent saw test_case_A failed, test_case_B failed, test_case_C failed but never fixed and re-ran
               - Example: Test output shows failed tests (e.g. FAIL, failed, test failed) but agent never fixed them
               - Red flag: Agent acknowledges seeing failures but doesn't address them
            
            3. **Rationalization** - Agent explained away failures as "unrelated" or "acceptable"
               - Example: Agent saw N test failures but thought "not related to problem statement, so bug is fixed" then finish
               - Example: Agent claimed "failing tests are edge cases" or "seem unrelated to my fix"
               - Example: Agent claimed "test failures existed before my changes" without verification
               - Red flag: Any justification for why failing tests are "OK" or "ignorable"
            
            4. **Problem Statement Excuse** - Agent claims failures are unrelated to the problem statement
               - Example: Agent saw N test failures but thought "they are not related to problem_statement, so bug is fixed" then finish
               - Red flag: Agent dismisses regressions by claiming they're outside the scope of the fix
               - Critical: ALL tests that were passing before changes but failing after are regressions, regardless of problem statement
            
            5. **No Full Suite Run** - Agent never ran the full test suite for the affected module
               - Red flag: Only ran individual test methods, never the full module/class
               - Example: Modified a utility function but only tested one caller, not all callers
            
            6. **Custom Scripts Instead of Real Tests** - Agent relied on `run_code` demos instead of actual test suite
               - Red flag: Multiple `run_code` calls with demo scripts, but no `run_tests` with full suite
               - Example: Created verification scripts instead of running project's test suite
            
            7. **Returned Bug (Critical)** - Agent fixed regressions but re-introduced the original bug
               - Example: Agent fixed regressions but then the hidden tests (originally failing) are failing again
               - Red flag: Agent focuses only on fixing regressions and forgets to verify the original bug is still fixed
               - Critical: BOTH original bug AND regressions must be fixed simultaneously
            
            **WHAT CONSTITUTES COMPLETE SUCCESS (BOTH CONDITIONS REQUIRED)**:
            
            ✅ **CONDITION 1: ORIGINAL BUG IS FIXED**
               - The hidden tests mentioned in the problem statement are now passing
               - Agent verified the fix with actual test runs (not just theory)
               - The bug described in the problem is demonstrably resolved
            
            ✅ **CONDITION 2: NO REGRESSIONS**
               - Agent ran the FULL test suite (or at minimum, the full test class) for affected modules
               - Agent saw test failures and FIXED them (re-ran tests after fixes until they passed)
               - The FINAL test run before calling finish showed ALL tests passing (no failed test output)
               - Agent used `run_tests` with the project's test runner, not just `run_code` demos
               - Agent fixed ALL regressions, regardless of whether they seem "related" to the problem statement
               - A regression is ANY test that was passing before changes but failing after
            
            ✅ **CRITICAL VERIFICATION**:
               - Agent must verify BOTH conditions are true in the SAME final test run
               - Cannot assume "bug is fixed" if only regressions are resolved
               - Cannot assume "regressions are fixed" if only the original bug is resolved
               - BOTH must be verified together in the final test output
            
            **YOUR TASK**:
            
            Analyze the conversation history below and verify BOTH conditions:
            
            **CONDITION 1 CHECK - Original Bug Fixed?**
            1. Are the hidden tests (originally failing, mentioned in problem statement) now passing?
            2. Did the agent verify the bug fix with actual test runs?
            3. Is there evidence the original problem is resolved?
            
            **CONDITION 2 CHECK - No Regressions?**
            4. Did the agent run comprehensive regression tests?
            5. Are there any unresolved test failures?
            6. Did the agent rationalize failures away (including "not related to problem statement")?
            7. Did the agent only test specific cases instead of the full suite?
            8. Did the agent use demo scripts instead of the real test suite?
            
            **CRITICAL CHECK - Returned Bug?**
            9. Did the agent fix regressions but then the original bug came back?
            10. Did the final test run verify BOTH original bug fix AND no regressions?
            
            **YOUR RESPONSE FORMAT**:
            
            - **IF BOTH CONDITIONS MET** (original bug fixed AND no regressions): Return exactly "REGRESSION_AND_BUG_CHECK_PASSED" followed by a brief explanation
            - **IF ANY ISSUES FOUND**: Return detailed feedback explaining:
              * Which condition failed (original bug, regressions, or both)
              * What specific evidence shows the failure
              * What the agent must do to fix it
            
            **CONVERSATION HISTORY TO ANALYZE**:
            
            {conversation_history}
            
            **YOUR RESPONSE**:
        """).strip()
        
        try:
            # Call LLM to analyze regression testing
            messages = [
                {
                    "role": "system",
                    "content": "You are a rigorous QA reviewer checking for proper regression testing. Be strict and thorough."
                },
                {
                    "role": "user",
                    "content": regression_check_prompt.format(
                        problem_statement=problem_statement,
                        conversation_history=conversation_history
                    )
                }
            ]
            
            review_result, _ = Network.make_request(messages, model=QWEN_MODEL_NAME, temperature=0.0)
            return review_result.strip()
            
        except Exception as e:
            # If LLM call fails, return a conservative response requiring verification
            return f"⚠️ Regression verification LLM call failed: {e}\n\nPlease manually verify that ALL regression tests pass before finishing."

class TestManager:
    def run_code(self, content: str, file_path: str, generated_test_files: list, run_command: list[str]) -> str:
        if file_path.endswith((".py", ".pyw", ".pyx", ".pyi", ".pxd", ".pxi", ".pyz")):
            content = VERSION_COMPATIBILITY_FIX + "\n\n" + content
        file_exists = os.path.exists(file_path) and os.path.isfile(file_path)
        self.file_ops.save(file_path, content)
        if file_path not in generated_test_files and not file_exists:
            generated_test_files.append(file_path)
        try:
            result = subprocess.run(run_command, capture_output=True, text=True, check=False, timeout=60)
            if result.returncode != 0:
                return f"Error running code: {result.stderr}"
            return f"{result.stdout}\n"
        except Exception as e:
            return f"Error: {e}"

    def __init__(self, runner_hint: str | None = None, runner_mode_hint: str | None = None, file_ops: "FileOperationsUtil" = None):
        self.runner_hint = runner_hint
        self.runner_mode_hint = runner_mode_hint
        self.file_ops = file_ops

class Utils:
    @classmethod
    def count_tokens(cls, messages: list | str) -> int:
        import re

        if isinstance(messages, list):
            text = " ".join(str(m.get("content", "") if isinstance(m, dict) else m) for m in messages)
        else:
            text = messages

        tokens = re.findall(r"\w+|[^\w\s]|\s+", text)
        count = 0
        for token in tokens:
            if token.isspace():
                continue
            elif len(token) == 1:
                count += 1
            else:
                count += max(1, (len(token) + 2) // 3)
        return count

    @classmethod
    def limit_strings(cls, strings: str, n=1000) -> str:
        strings_list = strings.split("\n")
        if len(strings_list) > n:
            return "\n".join(strings_list[:n]) + "\n..." + f"({len(strings_list)-n} more lines)"
        else:
            return strings

    @classmethod
    def load_json(cls, json_string: str) -> dict:
        try:
            return json.loads(json_string)
        except Exception:
            # Try common fixes: single quotes -> double quotes, trailing commas
            try:
                sanitized = json_string.replace("'", '"')
                sanitized = re.sub(r",\s*([}\]])", r"\1", sanitized)
                return json.loads(sanitized)
            except Exception:
                fixed_json = Network.fix_json_string_with_llm(json_string)
                if fixed_json:
                    return fixed_json
                else:
                    raise JSONDecodeError("Invalid JSON", json_string, 0)

class FileSystemManager:
    def __init__(self):
        pass

    def get_file_content(
        self,
        file_path: str,
        search_start_line: int = None,
        search_end_line: int = None,
        search_term: str = None,
        limit: int = 1000,
        add_line_numbers: bool = False,
        search_in_file_callback=None,
    ) -> str:

        def add_line_numbers_to_content(content: str, start_line: int = 1) -> str:
            lines = content.splitlines()
            numbered_lines = []
            for i, line in enumerate(lines):
                line_num = start_line + i
                numbered_lines.append(f"{line_num:6}|{line}")
            return "\n".join(numbered_lines)
        if search_term and search_in_file_callback:
            return search_in_file_callback(file_path, search_term)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            if search_start_line is not None or search_end_line is not None:
                lines = f.readlines()
                start_idx = max(0, (search_start_line or 1) - 1)
                end_idx = min(len(lines), search_end_line or len(lines))
                content = "".join(lines[start_idx:end_idx])
                if add_line_numbers:
                    result = add_line_numbers_to_content(content, start_idx + 1)
                else:
                    result = content
            else:
                content = f.read()
                if add_line_numbers:
                    result = add_line_numbers_to_content(content, 1)
                else:
                    result = content
        return Utils.limit_strings(result, n=limit) if limit != -1 else result

    def list_directory_structure(self, directory_path: str, max_depth: int = 0) -> str:
        if not os.path.exists(directory_path):
            return f"Error: Directory '{directory_path}' does not exist."
        if not os.path.isdir(directory_path):
            return f"Error: '{directory_path}' is not a directory."
        ignore = {".git", "__pycache__", ".pytest_cache", "node_modules", ".tox", ".venv", "venv", ".eggs"}
        def tree(path: str, prefix: str = "", depth: int = 0, current_max_depth: int = 0) -> list[str]:
            if depth > current_max_depth:
                return []
            try:
                items = sorted(os.listdir(path))
            except (PermissionError, OSError) as e:
                return [f"{prefix}[Error reading directory: {str(e)}]"]
            dirs = [
                i for i in items if os.path.isdir(os.path.join(path, i)) and not i.startswith(".") and i not in ignore and not i.endswith(".egg-info")
            ]
            files = [i for i in items if os.path.isfile(os.path.join(path, i)) and not i.startswith(".")]
            lines: list[str] = []
            for idx, d in enumerate(dirs):
                is_last = (idx == len(dirs) - 1) and not files
                branch = "└── " if is_last else "├── "
                new_prefix = prefix + ("    " if is_last else "│   ")
                lines.append(f"{prefix}{branch}{d}/")
                lines.extend(tree(os.path.join(path, d), new_prefix, depth + 1, current_max_depth))
            for idx, f in enumerate(files):
                is_last = idx == len(files) - 1
                branch = "└── " if is_last else "├── "
                lines.append(f"{prefix}{branch}{f}")
            return lines

        def count_tokens(text: str) -> int:
            try:
                if "Utils" in globals() and hasattr(Utils, "count_tokens"):
                    return Utils.count_tokens(text)
            except (NameError, AttributeError):
                pass
            return len(text) // 4
        MAX_TOKENS = 3000
        current_depth = max_depth
        while current_depth >= 0:
            entries = tree(directory_path, "", 0, current_depth)
            result = f"Directory structure (depth={current_depth}):\n{directory_path}/\n" + "\n".join(entries)
            token_count = count_tokens(result)
            if token_count <= MAX_TOKENS:
                if current_depth < max_depth:
                    result += (
                        f"\n\n[Note: Requested depth {max_depth} exceeded token limit. Showing depth {current_depth} instead ({token_count} tokens).]"
                    )
                return result
            if current_depth == 0:
                result += f"\n\n[Warning: Result exceeds token limit ({token_count} tokens > {MAX_TOKENS} tokens). Consider using a more specific directory_path.]"
                return result
            current_depth -= 1
        entries = tree(directory_path, "", 0, 0)
        result = f"Directory structure (depth=0):\n{directory_path}/\n" + "\n".join(entries)
        return result

class CodeEditManager:
    def apply_code_edit(self, file_path: str, search: str, replace: str) -> str:
        def add_context_to_similar_match(original_content: str, formatted_match: str, context_lines: int = 2) -> str:
            lines = original_content.split("\n")
            match_lines = formatted_match.split("\n")
            if len(match_lines) < 2:
                return formatted_match
            actual_content_lines = match_lines[1:]
            actual_content = "\n".join(actual_content_lines)
            best_match_start = -1
            best_similarity = 0
            for i in range(len(lines) - len(actual_content_lines) + 1):
                candidate_lines = lines[i:i + len(actual_content_lines)]
                candidate_content = "\n".join(candidate_lines)
                import difflib
                similarity = difflib.SequenceMatcher(None, actual_content.strip(), candidate_content.strip()).ratio()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_start = i
            if best_match_start == -1:
                return formatted_match
            start_line = max(0, best_match_start - context_lines)
            end_line = min(len(lines), best_match_start + len(actual_content_lines) + context_lines)
            context_lines_list = []
            for i in range(start_line, end_line):
                line_num = i + 1
                prefix = ">>> " if best_match_start <= i < best_match_start + len(actual_content_lines) else "    "
                context_lines_list.append(f"{prefix}{line_num:4}| {lines[i]}")
            description = match_lines[0] if match_lines else f"Match found at lines {best_match_start+1}-{best_match_start+len(actual_content_lines)}"
            return f"{description}\n" + "\n".join(context_lines_list)

        def find_most_similar_content(original_content: str, search_string: str, max_results: int = 3) -> list[tuple[float, str]]:
            import difflib
            lines = original_content.split("\n")
            chunks = []
            for i, line in enumerate(lines):
                if line.strip():
                    chunks.append((f"Line {i+1}: {line.strip()}", line.strip()))
            search_lines = search_string.split("\n")
            target_chunk_size = max(3, len(search_lines))
            for i in range(len(lines) - target_chunk_size + 1):
                chunk_lines = lines[i:i + target_chunk_size]
                chunk_content = "\n".join(chunk_lines).strip()
                if chunk_content:
                    chunks.append((f"Lines {i+1}-{i+target_chunk_size}: ...", chunk_content))
            similarities = []
            for chunk_desc, chunk_content in chunks:
                ratio = difflib.SequenceMatcher(None, search_string.strip(), chunk_content).ratio()
                if ratio > 0.3:
                    similarities.append((ratio, chunk_desc, chunk_content))
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [(ratio, f"{desc}\n{content}") for ratio, desc, content in similarities[:max_results]]
        if search == replace:
            return "ERROR: search and replace are the same. Please provide a different search and replace."
        if not os.path.exists(file_path):
            return f"Error: file '{file_path}' does not exist."
        original = self.file_ops.get_file_content(file_path, limit=-1)
        match original.count(search):
            case 0:
                similar_matches = find_most_similar_content(original, search, 1)
                error_msg = f"Error: search string not found in file {file_path}."
                if similar_matches:
                    error_msg += f"\n\nMost similar snippet found (you may need to adjust your search string):"
                    for i, (ratio, content) in enumerate(similar_matches, 1):
                        similarity_pct = int(ratio * 100)
                        content_with_context = add_context_to_similar_match(original, content, context_lines=2)
                        error_msg += f"\n\n{i}. Similarity: {similarity_pct}%\n{content_with_context}"
                else:
                    error_msg += " No similar content found. Please check the file content and provide the exact code you want to replace."
                return error_msg
            case 1:
                new_content = original.replace(search, replace)
                try:
                    self.file_ops.save(file_path, new_content)

                    replace_pos = new_content.find(replace)
                    if replace_pos != -1:
                        lines = new_content.split("\n")
                        chars_so_far = 0
                        replace_line_start = 0
                        for i, line in enumerate(lines):
                            if chars_so_far + len(line) >= replace_pos:
                                replace_line_start = i
                                break
                            chars_so_far += len(line) + 1  # +1 for newline
                        replace_lines_count = replace.count("\n") + 1
                        replace_line_end = replace_line_start + replace_lines_count - 1
                        start_line = max(0, replace_line_start - 10)
                        end_line = min(len(lines), replace_line_start + 10)
                        context_lines = []
                        for i in range(start_line, end_line):
                            line_num = i + 1
                            if replace_line_start <= i <= replace_line_end:
                                prefix = ">>> "
                            else:
                                prefix = "    "
                            context_lines.append(f"{prefix}{line_num:4}| {lines[i]}")
                        context = "\n".join(context_lines)
                        return f"ok, code edit applied successfully. Here is the edited code (lines {start_line+1}-{end_line}):\n\n{context}"
                    else:
                        return "ok, code edit applied successfully"
                except Exception as e:
                    return f"Error: syntax error in file {file_path}. {str(e)}"
            case num_hits:
                return f"Error: search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change."

    def __init__(self, file_ops: "FileOperationsUtil" = None):
        self.file_ops = file_ops

class ToolManager:
    TOOL_LIST = {}

    def get_tool_docs(self) -> str:
        return "\n\n".join([json.dumps(tool_metadata, ensure_ascii=False) for _, tool_metadata in self.TOOL_LIST.items()])

    def __init__(self, **kwargs):
        pass

    @classmethod
    def tool_parsing(cls, fn):
        tool_schemas = None
        name = fn.__name__
        doc_fn = fn.__doc__ or ""
        doc = doc_fn.split("Arguments:")[0]
        output_description = doc_fn.split("Output:")
        if len(output_description) > 1:
            output_description = "Output: " + output_description[1].strip()
            doc = doc + "\n\n" + output_description
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for param in sig.parameters.values():
            if param.name == "self":
                continue
            if param.default is param.empty and param.kind in (
                param.POSITIONAL_OR_KEYWORD,
                param.KEYWORD_ONLY,
            ):
                required.append(param.name)
            type_hint = str(param.annotation) if param.annotation != param.empty else "string"
            param_description = re.search(f"{param.name}:([^\n]+)", doc_fn)
            if param_description:
                param_description = param_description.group(1)
            else:
                raise ValueError(f"Parameter description not found for {param.name} in {doc_fn}: tool name: {name}")
            if ("list" in type_hint.lower()) and ("str" in type_hint):
                properties[param.name] = {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": param_description,
                }
                continue
            elif "str" in type_hint:
                json_type = "string"
            elif "int" in type_hint:
                json_type = "integer"
            elif "float" in type_hint:
                json_type = "number"
            elif "bool" in type_hint:
                json_type = "boolean"
            else:
                json_type = "string"
            properties[param.name] = {
                "type": json_type,
                "description": param_description,
            }
        parameters = {"type": "object", "properties": properties, "required": required}
        tool_schemas = {
            "name": name,
            "description": doc.strip(),
            "input_schema": parameters,
        }
        return tool_schemas

    def tool(fn):
        def wrapper(self, *args, **kwargs):
            self.tool_invocations[fn.__name__] = self.tool_invocations.get(fn.__name__, 0) + 1
            try:
                return fn(self, *args, **kwargs)
            except ToolManager.Error as e:
                if fn.__name__ not in self.tool_failure:
                    self.tool_failure[fn.__name__] = {j: 0 for j in self.Error.ErrorType.__members__}
                self.tool_failure[fn.__name__][e.error_type] += 1
                return e.message

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper.__signature__ = inspect.signature(fn)
        wrapper.__annotations__ = fn.__annotations__.copy()
        wrapper.is_tool = True
        return wrapper

    def get_tool(self, tool_name: str):
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        tool_method = getattr(self, tool_name, None)
        if tool_method is None or not callable(tool_method):
            return f"Error: tool '{tool_name}' does not exist. Please use one of the following tools: {', '.join(self.TOOL_LIST.keys())}"
        return tool_method

    @classmethod
    def get_final_git_patch(cls) -> str:
        try:
            exclude = {"src/agent.py", "src/agent_runner.py"}
            try:
                for _p in getattr(cls, "generated_test_files", []):
                    if os.path.exists(_p) and os.path.isfile(_p):
                        exclude.add(os.path.relpath(_p))
            except Exception:
                pass
            ls = subprocess.run(
                ["git", "ls-files", "-m", "--exclude-standard"], capture_output=True, text=True, timeout=30, check=True
            ).stdout.splitlines()
            to_add = [f for f in ls if f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)
            diff = subprocess.run(["git", "diff", "--cached", "--no-color", "--unified=3"], capture_output=True, text=True, timeout=30, check=True)
            if diff.stderr:
                logger.warning("git diff (stderr): %s", diff.stderr.strip())
            patch_text = diff.stdout or ""
            return patch_text
        except Exception as e:
            return f"Error generating git patch: {e}"

    @classmethod
    def get_tool_args_for_tool(cls, tool_name: str, required_only: bool = False) -> list[str]:
        if tool_name not in cls.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        if not required_only:
            return list(cls.TOOL_LIST[tool_name]["input_schema"]["properties"].keys())
        else:
            return cls.TOOL_LIST[tool_name]["input_schema"]["required"]

    @classmethod
    def get_modified_files_list(cls) -> list[str]:
        """
        Get a list of modified files (not newly created) from the git repository.
        Files that exist in the original repository and have been modified.

        Returns:
            List of file paths relative to repository root, excluding:
            - Newly created files (not in original repo)
            - Agent files (src/agent.py, src/agent_runner.py)
            - Generated test files
        """
        try:
            exclude = {"src/agent.py", "src/agent_runner.py"}
            try:
                for _p in getattr(cls, "generated_test_files", []):
                    if os.path.exists(_p) and os.path.isfile(_p):
                        exclude.add(os.path.relpath(_p))
            except Exception:
                pass

            # Get modified files (M = modified, not including Added or Deleted)
            # This compares against HEAD (original repository state)
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=M", "HEAD"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,  # Don't fail if no modified files or HEAD doesn't exist
            )

            if result.returncode != 0:
                # If HEAD doesn't exist or other error, return empty list
                logger.warning(f"Git diff failed: {result.stderr}")
                return []

            modified_files = [f.strip() for f in result.stdout.splitlines() if f.strip()]

            # Filter out excluded files
            modified_files = [f for f in modified_files if f not in exclude]

            # Also verify files exist and are tracked in git (to exclude untracked files)
            final_list = []
            for file_path in modified_files:
                # Check if file is tracked in git (exists in HEAD)
                check_result = subprocess.run(["git", "ls-tree", "--name-only", "HEAD", file_path], capture_output=True, text=True, timeout=10)
                if check_result.returncode == 0 and check_result.stdout.strip():
                    final_list.append(file_path)

            return final_list
        except Exception as e:
            logger.warning(f"Error getting modified files list: {e}")
            return []

    class Error(Exception):
        class ErrorType(Enum):
            SYNTAX_ERROR = 1
            RUNTIME_ERROR = 2
            TIMEOUT = 3
            FILE_NOT_FOUND = 4
            SEARCH_TERM_NOT_FOUND = 5
            UNKNOWN = 6
            THIRD_PARTY_DEPENDENCIES = 7
            MULTIPLE_SEARCH_RESULTS_FOUND = 8
            BUG_REPORT_REQUIRED = 9
            INVALID_RESPONSE_FORMAT = 10
            INVALID_TOOL_NAME = 11
            INVALID_FILE_PATH = 12
            INVALID_TOOL_CALL = 13
            IMPORT_ERROR = 14

        def __init__(self, error_type: ErrorType, message: str):
            self.error_type = error_type
            self.message = message

class Network:
    @classmethod
    def _extract_tool_call_from_block(cls, block: str) -> dict | None:
        tool_name_match = re.search(r"tool_name\s*:\s*([^\s]+)", block, re.IGNORECASE)
        if not tool_name_match:
            return None
        tool_name = tool_name_match.group(1).strip("\"'")
        args_match = re.search(r"tool_args\s*:\s*\{", block, re.IGNORECASE)
        if not args_match:
            return None
        args_start = args_match.end() - 1
        json_str = cls._extract_balanced_braces(block, args_start)
        if json_str:
            try:
                tool_args = json.loads(json_str)
                return {"tool_name": tool_name, "tool_args": tool_args}
            except json.JSONDecodeError:
                try:
                    tool_args = json.loads(json_str.replace("'", '"'))
                    return {"tool_name": tool_name, "tool_args": tool_args}
                except Exception:
                    pass
        return None

    @classmethod
    def is_http_response(cls, raw_text: str):
        if "API request failed with status 429" in raw_text:
            return False, cls.ErrorType.RATE_LIMIT_EXCEEDED.name
        if "Read timed out" in raw_text:
            return False, cls.ErrorType.TIMEOUT.name
        if "HTTP ERROR: Request failed for model" in raw_text:
            return False, cls.ErrorType.NETWORK_ERROR.name
        return True, None

    @classmethod
    def inference(
        cls,
        messages: list[dict],
        model: str,
        run_id: str = str(uuid4()),
        temperature: float = 0.0,
    ) -> dict:
        models = [model] if isinstance(model, str) else model
        cleaned_msgs = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in messages
            if m.get("role") in {"system", "user", "assistant", "tool"} and (m.get("role") != "assistant" or m.get("content", "").strip())
        ]
        if not cleaned_msgs:
            raise RuntimeError("No valid messages to send to proxy.")

        result = cls._request_next_action_with_retry(cleaned_msgs, models=models, temperature=temperature)
        return result

    @classmethod
    def _extract_balanced_braces(cls, text: str, start_pos: int) -> str | None:
        if start_pos >= len(text):
            return None
        brace_count, in_string, escape_next, start = 0, False, False, -1
        for i in range(start_pos, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == "\\":
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if not in_string:
                if c == "{":
                    if start == -1:
                        start = i
                    brace_count += 1
                elif c == "}":
                    brace_count -= 1
                    if brace_count == 0 and start != -1:
                        return text[start:i + 1]
        return None

    @classmethod
    def _request_next_action_with_retry(
        cls,
        messages: dict,
        models: list[str],
        max_retries: int = 3,
        temperature: float = 0.0,
    ) -> str:
        raw_text = None
        error_counter = cls.get_error_counter()
        next_thought = next_tool_name = next_tool_args = None
        total_attempts = 0
        current_model_idx = 0
        used_model = models[0] if models else None
        for attempt in range(max_retries):
            try:
                total_attempts += 1
                current_model = models[min(current_model_idx, len(models) - 1)]
                used_model = current_model
                raw_text, _ = cls.make_request(messages, model=current_model, temperature=temperature)
                is_valid, error_msg = cls.is_valid_response(raw_text)
                if not is_valid:
                    raise Exception(error_msg)
                next_thought, next_tool_name, next_tool_args, error_msg = cls.parse_response(raw_text)
                if error_msg:
                    raise Exception(error_msg)
                break
            except Exception as e:
                error_body = str(e)
                is_504_error = "504" in error_body or "HTTP ERROR 504" in error_body or "Gateway Timeout" in error_body
                if is_504_error and current_model_idx < len(models) - 1:
                    current_model_idx += 1
                    time.sleep(3)
                    continue
                if attempt < max_retries - 1:
                    matched = False
                    for key in ["RATE_LIMIT_EXCEEDED", "RESERVED_TOKEN_PRESENT", "EMPTY_RESPONSE", "TIMEOUT", "Invalid JSON", "Invalid response"]:
                        if key in error_body:
                            attr_name = key if key in cls.ErrorType.__members__ else "INVALID_RESPONSE_FORMAT"
                            error_counter[attr_name] += 1
                            matched = True
                            break
                    if not matched:
                        error_counter[cls.ErrorType.UNKNOWN.name] += 1
                    skip_http = any(
                        x in error_body
                        for x in [
                            "HTTP ERROR",
                            "RATE_LIMIT_EXCEEDED",
                            "RESERVED_TOKEN_PRESENT",
                            "EMPTY_RESPONSE",
                            "TIMEOUT",
                            "NETWORK_ERROR",
                            "HTTP ERROR 429",
                            "INCOMPLETE_RESPONSE",
                        ]
                    )
                    if not skip_http:
                        messages.append({"role": "assistant", "content": raw_text})
                        messages.append({"role": "user", "content": "observation: " + error_body})
                    time.sleep(3)
                    continue
                else:
                    error_counter[cls.ErrorType.TIMEOUT.name] += 1
                    raise RuntimeError(error_body)
        return (
            next_thought,
            next_tool_name,
            next_tool_args,
            raw_text,
            total_attempts,
            error_counter,
            messages,
            used_model,
        )

    @classmethod
    def parse_malformed_json(cls, arguments: list[str], json_string: str) -> dict | str:
        pattern = r",\s*".join(rf'"{k}": (.*)' for k in arguments)
        match = re.search(pattern, json_string)
        if not match:
            return f"Error: {json_string} can not match pattern {pattern}"
        return {k: match.group(i + 1).strip().strip('"').replace("\\n", "\n") for i, k in enumerate(arguments)}

    @classmethod
    def make_request(
        cls,
        messages: list,
        model: Model,
        attempt: int = 0,
        temperature: float = 0.0,
        tool_mode: str = "none",
        tool_docs: list = None,
        timeout: int = 150,
    ) -> tuple[str, list]:
        if tool_docs is None:
            tool_docs = []
        global run_id, agent_start_time, total_inferenced_chars, individual_inferenced_chars
        messages_str = json.dumps(messages, ensure_ascii=False)
        individual_inferenced_chars = len(messages_str)
        total_inferenced_chars += individual_inferenced_chars

        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
        attempts = max(1, attempt or 1)
        model_name = model.name if isinstance(model, Model) else model
        model_timeout = model.timeout if isinstance(model, Model) else timeout

        request_data = {
            "evaluation_run_id": run_id if run_id else str(uuid4()),
            "messages": messages,
            "temperature": temperature,
            "model": model_name,
            "tool_mode": tool_mode,
            "tools": tool_docs,
        }
        headers = {"Content-Type": "application/json"}
        for i in range(attempts):
            try:
                start_time = time.time()
                print(f"⏳ Sending request {model_timeout} seconds timeout")
                resp = requests.post(url, json=request_data, timeout=(30, model_timeout), headers=headers)
                resp.raise_for_status()
                print(f"✔ Request success {time.time() - start_time:.2f} seconds elapsed!")
                try:
                    resp_json = resp.json()
                except JSONDecodeError as e:
                    if i >= attempts - 1:
                        raise ValueError(f"HTTP ERROR: Invalid JSON response for model {model_name} after {attempts} attempts: {e}")
                    continue
                try:
                    raw_text = resp_json["content"]
                    tool_calls = resp_json["tool_calls"]
                except Exception:
                    raise RuntimeError(f"HTTP ERROR: Response Parse Error timeout for model {model_name} after {attempts} attempts")
                if (tool_mode == "none" and not raw_text) or (tool_mode != "none" and not tool_calls):
                    raise RuntimeError(f"HTTP ERROR: NO RESPONSE FOUND Tool model {model_name} after {attempts} attempts")
                return raw_text, tool_calls
            except requests.exceptions.Timeout:
                if i >= attempts - 1:
                    raise RuntimeError(f"HTTP ERROR: Request timeout for model {model_name} after {attempts} attempts")
                time.sleep(1)
            except requests.exceptions.ConnectionError as e:
                if i >= attempts - 1:
                    raise RuntimeError(f"HTTP ERROR: Connection error for model {model_name} after {attempts} attempts: {e}")
                time.sleep(1)
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else "unknown"
                if status_code == 504:
                    if i >= attempts - 1:
                        raise RuntimeError(f"HTTP ERROR 504: Gateway Timeout for model {model_name} after {attempts} attempts: {e}")
                    time.sleep(1)
                    continue
                error_msg = f"HTTP ERROR: HTTP ERROR {status_code} for model {model_name}"
                if i >= attempts - 1:
                    raise RuntimeError(f"{error_msg} after {attempts} attempts: {e}")
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                if i >= attempts - 1:
                    raise RuntimeError(f"HTTP ERROR: Request failed for model {model_name} after {attempts} attempts: {e}")
                time.sleep(1)
        raise RuntimeError(f"HTTP ERROR: Failed to get response for model {model_name} after {attempts} attempts")

    @classmethod
    def sanitise_text_resp(cls, text_resp: str) -> str:
        text_resp = re.sub(r"['\"]*next_thought['\"]*:", "next_thought:", text_resp)
        text_resp = re.sub(r"['\"]*next_tool_name['\"]*:", "next_tool_name:", text_resp)
        text_resp = re.sub(r"['\"]*next_tool_args['\"]*:", "next_tool_args:", text_resp)
        text_resp = re.sub(r"['\"]*observation['\"]*:", "observation:", text_resp)
        text_resp = re.sub(r"['\"]*tool_call_['\"]*", "tool_call_", text_resp)
        if (
            "next_thought" not in text_resp
            and "next_tool_name:" in text_resp
            and "next_tool_args:" in text_resp
            and text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:")
            and text_resp.find("next_tool_name:") > 10
        ):
            text_resp = "next_thought: " + text_resp
        if (
            "next_tool_name:" in text_resp
            and "next_tool_args:" in text_resp
            and text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:")
        ):
            next_tool_name = text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n").strip("'").strip('"').strip()
            text_resp = re.sub(
                f"next_tool_name:['\" ]*{re.escape(next_tool_name)}['\" ]*",
                "next_tool_name: " + next_tool_name,
                text_resp,
            )
        return text_resp

    @classmethod
    def parse_next_tool_args(cls, tool_name: str, next_tool_args: str) -> dict | str:
        next_tool_args = next_tool_args.replace("```json", "").strip("```")
        try:
            return Utils.load_json(next_tool_args.strip())
        except JSONDecodeError:
            try:
                schema_tool_name = tool_name[0] if isinstance(tool_name, list) and tool_name else tool_name
                return cls.parse_malformed_json(
                    ToolManager.get_tool_args_for_tool(schema_tool_name, required_only=True),
                    next_tool_args,
                )
            except (ToolManager.Error, Exception):
                raise Exception(f"Invalid JSON: {next_tool_args}")

    @classmethod
    def fix_json_string_with_llm(cls, json_string: str, attempt: int = 0) -> dict:
        messages = [
            {
                "role": "system",
                "content": "Fix the json string sent by the user.  Reply only with the json string and nothing else.",
            },
            {"role": "user", "content": json_string},
        ]
        selected_model = QWEN_MODEL_NAME
        retry = 0
        while retry < 5:
            try:
                response, _ = cls.make_request(messages, model=selected_model)
                break
            except Exception:
                retry += 1
                remaining = [model for model in AGENT_MODELS if model != selected_model]
                if remaining:
                    selected_model = random.choice(remaining)
                time.sleep(1)
        try:
            response = response.replace("```json", "").strip("```")
            return json.loads(response)
        except Exception:
            return None

    @classmethod
    def parse_response(cls, text_resp: str) -> tuple[str | None, any, any, str | None]:
        error_msg = None
        text_resp = text_resp.strip()
        if "observation:" in text_resp.lower():
            text_resp = re.split(r"observation\s*:", text_resp, flags=re.IGNORECASE)[0].strip()
        text_resp = cls.sanitise_text_resp(text_resp)
        if "Infrastructure is at maximum capacity" in text_resp:
            return None, None, None, "HTTP ERROR Maximum Capacity"
        if "No instances available" in text_resp:
            return None, None, None, "HTTP ERROR NO INSTANCES AVAILABLE"
        next_thought = None
        for pat in [
            r"next_thought\s*:\s*(.*?)(?=\n(?:tool_call_|next_tool_name:|$))",
            r"next_thought\s*:\s*(.*?)(?=\ntool_call_)",
            r"next_thought\s*:\s*(.*?)(?=\nnext_tool_name:)",
            r"next_thought\s*:\s*(.*)",
        ]:
            match = re.search(pat, text_resp, re.DOTALL | re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if candidate and len(candidate) > 2:
                    next_thought = candidate
                    break
        if not next_thought:
            next_thought = "Processing request"
        tool_call_matches = list(re.finditer(r"tool_call_(\d+)\s*:", text_resp, re.IGNORECASE))
        if tool_call_matches:
            tool_calls = []
            for i, match in enumerate(tool_call_matches):
                start = match.end()
                end = tool_call_matches[i + 1].start() if i + 1 < len(tool_call_matches) else len(text_resp)
                block = text_resp[start:end].strip()
                call = cls._extract_tool_call_from_block(block)
                if call:
                    tool_calls.append(call)
            if not tool_calls:
                return next_thought, None, None, "Multi-tool format detected but no valid tool calls extracted"
            tool_names = [c["tool_name"] for c in tool_calls]
            tool_args_list = [c["tool_args"] for c in tool_calls]
            if len(tool_names) == 1:
                return next_thought, tool_names[0], tool_args_list[0], error_msg
            return next_thought, tool_names, tool_args_list, error_msg

        if "next_tool_name:" in text_resp and "next_tool_args:" in text_resp:
            name_idx = text_resp.find("next_tool_name:")
            args_idx = text_resp.find("next_tool_args:")
            if text_resp.find("next_thought:") < name_idx < args_idx:
                next_tool_name_raw = text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip()
                next_tool_args_raw = text_resp.split("next_tool_args:")[1].strip().split("next_thought:")[0].strip()
                try:
                    if next_tool_name_raw.startswith("["):
                        next_tool_names = Utils.load_json(next_tool_name_raw)
                    else:
                        next_tool_names = [next_tool_name_raw]
                    parsed_args = cls.parse_next_tool_args(next_tool_names, next_tool_args_raw)
                    next_tool_args_list = parsed_args if isinstance(parsed_args, list) else [parsed_args for _ in next_tool_names]
                    if len(next_tool_names) == 1:
                        return next_thought, next_tool_names[0], next_tool_args_list[0], error_msg
                    return next_thought, next_tool_names, next_tool_args_list, error_msg
                except (JSONDecodeError, Exception) as e:
                    error_msg = f"Invalid JSON in tool args: {str(e)}"
                    return next_thought, None, None, error_msg

        if "next_thought:" not in text_resp:
            error_msg = "Invalid response. next_thought not found"
        elif "next_tool_name:" not in text_resp and "tool_call_" not in text_resp:
            error_msg = "Invalid response. No tool calls found (expected next_tool_name: or tool_call_N:)"
        elif "next_tool_args:" not in text_resp and "tool_call_" not in text_resp:
            error_msg = "Invalid response. next_tool_args not found"
        else:
            error_msg = "Invalid response format. Could not parse tool calls."
        return next_thought, None, None, error_msg

    @classmethod
    def get_error_counter(cls) -> dict[str, int]:
        return {k: 0 for k in cls.ErrorType.__members__}

    @classmethod
    def is_valid_response(cls, raw_text: str) -> tuple[bool, str | None]:
        if isinstance(raw_text, dict) and raw_text.get("error"):
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        stripped = raw_text.strip()
        lower = raw_text.lower()
        has_next_thought = "next_thought" in lower or "<next_thought>" in lower
        has_next_tool_name = "next_tool_name" in lower or "<next_tool_name>" in lower
        has_next_tool_args = "next_tool_args" in lower or "<next_tool_args>" in lower
        valid_ending = stripped.endswith("}") or stripped.endswith("}]") or stripped.endswith("</next_tool_args>") or stripped.endswith(">")
        if has_next_thought and has_next_tool_name and has_next_tool_args and not valid_ending:
            return False, cls.ErrorType.INCOMPLETE_RESPONSE.name
        if not raw_text:
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if "<|reserved_token_" in raw_text:
            return False, cls.ErrorType.RESERVED_TOKEN_PRESENT.name
        return cls.is_http_response(raw_text)

    class ErrorType(Enum):
        EMPTY_RESPONSE = 1
        RESERVED_TOKEN_PRESENT = 2
        RATE_LIMIT_EXCEEDED = 3
        INVALID_RESPONSE_FORMAT = 4
        TIMEOUT = 5
        UNKNOWN = 6
        NETWORK_ERROR = 7
        AUTHENTICATION_ERROR = 8
        RESOURCE_EXHAUSTED = 9
        INCOMPLETE_RESPONSE = 10

class FixTaskToolManager(ToolManager):
    def __init__(
        self,
        available_tools: Optional[list[str]] = None,
        runner_hint: str | None = None,
        runner_mode_hint: str | None = None,
        initial_checkpoint=None,
        problem_statement: str = None,
        should_review: bool = True,
        is_fix_task: bool = False,
        initial_structure: str = None,
        function_behaviours: dict = None,
        cot: "COT" = None,
    ):
        if available_tools is None:
            available_tools = []
        if function_behaviours is None:
            function_behaviours = {}
        self.new_files_created = []
        self.available_tools = available_tools
        self.runner_hint = runner_hint
        self.runner_mode_hint = runner_mode_hint
        self.generated_test_files = []
        self.initial_checkpoint = initial_checkpoint
        self.observation_dir = ".observation"
        self.problem_statement = problem_statement
        self.initial_structure = initial_structure
        self.repo_dir = "."
        self.saved_observation_counter = 0
        self.is_fix_task = is_fix_task
        self.strategy_counter = 0
        self.strategies = []
        if should_review:
            self.is_reviewed = False
            self.file_by_file_reviewed = False
        else:
            self.is_reviewed = True
            self.file_by_file_reviewed = True
        os.makedirs(self.observation_dir, exist_ok=True)
        self.file_ops = FileOperationsUtil(new_files_created=self.new_files_created)
        self.search_manager = SearchManager()
        self.file_system_manager = FileSystemManager()
        self.test_manager = TestManager(
            runner_hint=runner_hint,
            runner_mode_hint=runner_mode_hint,
            file_ops=self.file_ops,
        )
        self.code_edit_manager = CodeEditManager(file_ops=self.file_ops)
        self.code_parser = CodeParseUtil()
        self.thought_history: list[dict[str, Any]] = []
        self.branches: dict[str, list[dict[str, Any]]] = {}
        self.file_ops.set_managers(self.file_system_manager, self.search_manager)
        self.TOOL_LIST = {}
        for cls in self.__class__.__mro__:
            for name, attr in cls.__dict__.items():
                if getattr(attr, "is_tool", False) and name not in self.TOOL_LIST:
                    if available_tools is not None and name not in available_tools:
                        continue
                    self.TOOL_LIST[name] = self.__class__.tool_parsing(attr)
        self.tool_failure = {k: {j: 0 for j in self.Error.ErrorType.__members__} for k in self.TOOL_LIST.keys()}
        self.tool_invocations = {k: 0 for k in self.TOOL_LIST.keys()}
        self.finish_called_count = 0
        self.hypothesis_counter = 0
        self.hypotheses: List[Dict] = []
        self._current_step = 0
        self._cot_snapshot_cache = []
        self.validated_num = 0
        # Test tracking for fix_task
        self._test_call_count = 0
        self._pending_run_tests_confirmation: bool = False
        self._last_run_tests_step: int | None = None
        self._last_run_tests_passed: bool | None = None
        self._last_edit_step: int | None = None
        self._edit_count: int = 0
        self._last_blocked_edit_step: int | None = None
        self._blocked_edit_count: int = 0
        self._last_blocked_edit_message: str | None = None
        # Solution verifier
        self.cot = cot
        self.solution_verifier = SolutionVerifier(cot=cot, problem_statement=problem_statement) if cot else None
        # Problem decomposition storage
        self.problem_decomposition: Dict = None
        # Pre-edit strategy state (prevents prompt dilution and over-broad edits).
        # Keyed by target file_path (repo-relative).
        self.fix_strategy: Dict[str, Dict[str, Any]] = {}
        self.boundary_proofs: Dict[str, Dict[str, Any]] = {}
        # Track soft pre-edit warnings (guidance, not hard blocks).
        self._last_pre_edit_warning_step: int | None = None
        self._pre_edit_warning_count: int = 0
        self._last_pre_edit_warning_message: str | None = None

    def _has_recent_file_read(self, file_path: str, *, lookback: int = 40) -> bool:
        """
        Best-effort check: has the agent read this file recently via get_file_content/search_in_file/get_function_body?
        Uses the short cot snapshot cache maintained by execute_agent_workflow.
        """
        try:
            snap = getattr(self, "_cot_snapshot_cache", []) or []
            recent = snap[-lookback:]
            for item in reversed(recent):
                tool = str(item.get("tool", ""))
                args = str(item.get("args", ""))
                if tool in {"get_file_content", "search_in_file", "get_function_body"} and file_path in args:
                    return True
        except Exception:
            pass
        return False

    @ToolManager.tool
    def apply_code_edit(self, file_path: str, search: str, replace: str) -> str:
        """
        Performs targeted text replacement within source files. If there are any syntax errors in the code, it rejects the edit with an error message.
        Arguments:
            file_path: target file for modification
            search: exact text pattern to locate and replace
            replace: new text content to substitute
        Output:
            operation status - success confirmation or detailed error with guidance
        """
        result = self.code_edit_manager.apply_code_edit(file_path=file_path, search=search, replace=replace)
        try:
            if isinstance(result, str) and "ok, code edit applied successfully" in result.lower():
                self._last_edit_step = self._current_step
                self._edit_count += 1
        except Exception:
            pass
        return result

    @ToolManager.tool
    def modify_test_case(self, file_path: str, search: str, replace: str) -> str:
        """
        Modifies test files or test cases when they are incorrect or need correction.
        Use this tool when you identify that a test file or specific test case is wrong and needs to be fixed.
        This tool uses the same underlying mechanism as apply_code_edit but is specifically intended for correcting test files.
        Arguments:
            file_path: path to the test file that needs modification
            search: exact text pattern in the test file to locate and replace (e.g., the incorrect test case code)
            replace: corrected test case code to substitute
        Output:
            Operation status - success confirmation or detailed error with guidance
        """
        return self.code_edit_manager.apply_code_edit(file_path=file_path, search=search, replace=replace)

    @ToolManager.tool
    def run_code(self, content: str, file_path: str, run_command: List[str]) -> str:
        """
        Runs any code. You can use this tool directly to run any test code or bug reproduction code.
        Saves the code at the given file_path and then runs it. Do not use this tool to create test or files to reproduce the error unless user has specifically asked you to create test files as part of problem statement.
        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in. This file should always be in the current working directory.
            run_command: command to run the file (i.e., ["python", "file.py"] or ["node", "file.js"] etc)
        """
        return self.test_manager.run_code(
            content=content,
            file_path=file_path,
            generated_test_files=self.generated_test_files,
            run_command=run_command,
        )

    @ToolManager.tool
    def get_file_content(
        self,
        file_path: str,
        search_start_line: int = None,
        search_end_line: int = None,
        search_term: str = None,
    ) -> str:
        """
        Retrieves file contents with optional filtering based on search term and line numbers
        Arguments:
            file_path: filesystem path to target file.
            search_start_line: optional start line number to begin extraction (1-indexed)
            search_end_line: optional end line number to end extraction (1-indexed)
            search_term: optional text pattern to filter matching lines
        """
        return self.file_ops.get_file_content(
            file_path,
            search_start_line,
            search_end_line,
            search_term,
            add_line_numbers=True,
            limit=1000,
        )

    @ToolManager.tool
    def list_directory_structure(self, directory_path: str = ".", max_depth: int = 1) -> str:
        """
        Lists the directory structure of the repository
        Arguments:
            directory_path: the directory path to list (default: ".")
            max_depth: maximum depth to traverse (default: 1)
        """
        return self.file_system_manager.list_directory_structure(directory_path=directory_path, max_depth=max_depth)

    def _summarize_test_output(self, test_output: str) -> str:
        """Summarize long test output using LLM to preserve critical debugging info."""
        try:
            prompt = f"""Summarize this test execution output. Focus on:
1. Total tests run, passed, and failed counts
2. List ALL failed test cases with their exact names
3. For each failure: exact important short error message, location (file:line), and root cause
4. Any setup/teardown errors
5. Critical error traces (keep full stack traces for failures)
6. Any warnings or important messages

Keep all specific error details - the summary must be sufficient for debugging.

Test Output:
{test_output}

Provide a concise but complete summary:"""
            
            messages = [{"role": "user", "content": prompt}]
            summary, _ = Network.make_request(
                messages=messages,
                model=QWEN_MODEL_NAME,
            )
            return f"[TEST OUTPUT SUMMARIZED - Token count exceeded 5000]\n\n{summary}"
        except Exception as e:
            # If summarization fails, truncate intelligently
            lines = test_output.split('\n')
            if len(lines) > 200:
                return f"[TEST OUTPUT TRUNCATED]\n\n" + '\n'.join(lines[:100]) + f"\n\n... ({len(lines)-200} lines omitted) ...\n\n" + '\n'.join(lines[-100:])
            return test_output

    @ToolManager.tool
    def run_tests(self, command: List[str], timeout: int = 5) -> str:
        """
        Runs tests with strict timeout.
        Arguments:
            command: list of command line arguments,
            timeout: timeout in seconds (default: 5)
        Output:
            Standard output or error output of the command.
        """
        if self.is_fix_task and self._test_call_count == 0 and not self._pending_run_tests_confirmation:
            self._test_call_count += 1
            self._pending_run_tests_confirmation = True
            return textwrap.dedent(f"""
            ⚠️  VERIFICATION WORKFLOW DISCOVERY CHECK ⚠️
            
            You are about to run tests for the first time with command: {' '.join(command)}
            
            Before proceeding, you MUST confirm you have completed the mandatory discovery steps from section 5.5:
            
            ✓ Step 1: Examined repository root structure for verification entry scripts?
            ✓ Step 2: Inspected project documentation for test execution instructions?
            ✓ Step 3: Analyzed test organization and configuration?
            ✓ Step 4: Determined the canonical execution path with proper priority?
            
            CRITICAL QUESTIONS:
            1. Did you try to find a repository-specific test runner (Priority 1)?
               - Custom entry script in repository root?
               - Specialized test execution script?
            
            2. If no custom runner found, did you verify through documentation that 
               generic framework approach is the intended method?
            
            3. Is the command you're about to run the CORRECT way to execute tests 
               for this specific repository?
            
            ⚠️  COMMON MISTAKE: Jumping to framework-specific commands without discovering 
            canonical test runners wastes steps and causes execution failures.
            
            If you have NOT completed the discovery sequence:
            - STOP and complete section 5.5 discovery steps first
            - Use the repository exploration tools to examine structure and inspect relevant files
            - Look for specialized test runners before using generic commands
            
            If you HAVE completed discovery and verified this is the correct command:
            - Call run_tests again with the same command to proceed
            - The actual test execution will happen on the next call
            
            This confirmation only appears once. Subsequent run_tests calls will execute immediately.
            """).strip()
        
        # Actual test execution (second call onwards)
        if self._pending_run_tests_confirmation:
            self._pending_run_tests_confirmation = False
        try:
            preface_lines: list[str] = []
            if self.is_fix_task:
                try:
                    if (
                        self._last_blocked_edit_step is not None
                        and (self._last_edit_step is None or self._last_blocked_edit_step > self._last_edit_step)
                    ):
                        preface_lines.append(
                            "⚠️ NOTE: Your most recent code edit attempt was blocked by pre-edit gates. "
                            "This test run will execute against the last successfully applied code state."
                        )
                    if (
                        self._last_run_tests_step is not None
                        and (self._last_edit_step is None or self._last_edit_step <= self._last_run_tests_step)
                    ):
                        preface_lines.append(
                            "ℹ️ NOTE: No new successful code edits have been applied since the last test run; "
                            "this run mainly reconfirms the same code state."
                        )
                except Exception:
                    pass
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            test_output = result.stdout + result.stderr
            # Deterministic pass/fail capture for finish gating (do not rely on LLM heuristics).
            try:
                self._last_run_tests_step = self._current_step
                self._last_run_tests_passed = (result.returncode == 0)
            except Exception:
                pass
            
            # Token management: Summarize if output is too large
            token_count = Utils.count_tokens(test_output)  # Rough token estimate
            if token_count > SAVE_OBSERVATION_TO_FILE_TOKEN_THRESHOLD:
                print(f"⚠️  Test output large ({token_count} tokens, exceeds {SAVE_OBSERVATION_TO_FILE_TOKEN_THRESHOLD} tokens limit), summarizing with LLM...")
                test_output = self._summarize_test_output(test_output)
                print(f"✅ Test output summarized successfully with {Utils.count_tokens(test_output)} tokens:\n{test_output}")

            if preface_lines:
                return "\n".join(preface_lines).strip() + "\n\n" + test_output
            return test_output
            
        except subprocess.TimeoutExpired:
            return "Test run timed out."
        except Exception as e:
            return f"Test execution error: {e}"

    def _save_large_observation(self, observation: str, tool_name: str) -> tuple[str, int]:
        self.saved_observation_counter += 1
        filename = f"observation_{self.saved_observation_counter}_{tool_name}_{int(time.time())}.txt"
        if not os.path.exists(self.observation_dir):
            os.makedirs(self.observation_dir, exist_ok=True)
        file_path = os.path.join(self.observation_dir, filename)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(observation)
            line_count = observation.count("\n") + 1 if observation else 0
            return file_path, line_count
        except Exception as e:
            return f"Error: Failed to save observation: {e}", -1

    def get_final_git_patch(self) -> str:
        try:
            exclude = {"src/agent.py", "src/agent_runner.py"}
            try:
                for _p in getattr(self, "generated_test_files", []):
                    if os.path.exists(_p) and os.path.isfile(_p):
                        exclude.add(os.path.relpath(_p))
            except Exception:
                pass
            ls = subprocess.run(
                ["git", "ls-files", "-m", "--exclude-standard"], capture_output=True, text=True, timeout=30, check=True
            ).stdout.splitlines()
            to_add = [f for f in ls if f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)
            diff = subprocess.run(["git", "diff", "--cached", "--no-color", "--unified=3"], capture_output=True, text=True, timeout=30, check=True)
            patch_text = diff.stdout or ""
            return patch_text
        except Exception as e:
            return f"Error generating git patch: {e}"

    @ToolManager.tool
    def generate_test_cases_from_root_cause(self, root_cause_code: str, file_path: str = None, function_name: str = None) -> str:
        """
        Generates comprehensive test cases based on the problem statement and the identified root cause code section.
        Call this tool when you have identified the main root cause code part that needs to be fixed.
        The generated test cases will be saved and automatically referenced when you create test files using generate_test_file.
        Arguments:
            root_cause_code: The code section identified as the root cause of the issue (required)
            file_path: Optional file path where the root cause code is located (helps provide context)
            function_name: Optional function name where the root cause code is located (helps provide context)
        Output:
            A structured markdown document containing test cases with descriptions, inputs/setup, expected results, and reasons for each test case
        """
        if not self.problem_statement:
            return "Error: Problem statement not available. Cannot generate test cases."

        TEST_CASE_GENERATION_PROMPT = textwrap.dedent("""
        You are an expert test case generator. Your task is to generate comprehensive test cases based on a problem statement and the root cause code section.

        Analyze the problem statement and the root cause code to generate test cases that:
        1. Verify the bug exists (reproduction test)
        2. Verify the fix works correctly
        3. Cover edge cases related to the root cause
        4. Test boundary conditions

        For each test case, provide:
        - Test case description: What the test case does
        - Input/Setup: What inputs or setup are needed
        - Expected result: What should happen when the code is correct
        - Reason: Why this test case is important for verifying the root cause fix

        **NOTE**: Don't ONLY consider the primary issue in the problem statement.
        You should consider all, every possible edge cases.
        Invalid or wrong test cases should be also generated to test thoroughly.
        For those invalid or wrong cases, you should correctly handle error or edge case.

        Format your response as a structured markdown document with clear sections for each test case.
        Be specific and actionable. Focus on test cases that directly relate to the root cause identified.
        """)

        retry = 0
        selected_model = QWEN_MODEL_NAME
        root_cause_context = root_cause_code
        if file_path:
            root_cause_context += f"\n\nFile: {file_path}"
        if function_name:
            root_cause_context += f"\n\nFunction: {function_name}"

        while retry < 10:
            try:
                messages = [
                    {"role": "system", "content": TEST_CASE_GENERATION_PROMPT},
                    {
                        "role": "user",
                        "content": f"Problem Statement:\n{self.problem_statement}\n\nRoot Cause Code:\n{root_cause_context}\n\nGenerate comprehensive test cases for this root cause."
                    }
                ]
                test_cases, _ = Network.make_request(
                    messages, model=selected_model, attempt=1, temperature=0.0
                )
                self.generated_test_cases = test_cases
                print(f"[GENERATE_TEST_CASES_FROM_ROOT_CAUSE] Test cases generated successfully and saved: {test_cases}")
                return f"Test cases generated successfully and saved.\n\n{test_cases}"
            except Exception as e:
                logger.error(f"Error generating test cases: {e}")
                retry += 1
                if retry < 10:
                    other_models = [model for model in AGENT_MODELS if model != selected_model]
                    if other_models:
                        selected_model = random.choice(other_models)
                    time.sleep(1)
                else:
                    return f"Error: Failed to generate test cases after {retry} attempts: {str(e)}"
        return "Error: Failed to generate test cases"

    @ToolManager.tool
    def grep_search(self, grep_search_command: str) -> str:
        """
        Performs grep search on a single file or across multiple files in the codebase
        Arguments:
            grep_search_command: grep search command to locate (e.g., "grep <your grep command>").
        Output:
            locations where pattern was found with file paths and line numbers
        """
        return self.search_manager.search_in_all_files(grep_search_command)

    @ToolManager.tool
    def think(self, thought: str) -> str:
        """ Use the tool to think about something. It will not make any changes to the repository. Use it when reasoning or brainstorming is needed. For example, if you explore the repo and discover the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be correct and most effective. Alternatively, if you receive some test results, you can call this tool to brainstorm ways to fix the failing tests.
        Arguments:
            thought: Your thoughts.
        Output:
            Confirmation that the thought has been logged.
        """
        return "ok"

    @ToolManager.tool
    def find_symbol_references(self, symbol_identifier: str) -> str:
        """
        Discovers all code locations where a specific function, class, method, or variable is referenced.
        Provides contextual information around each usage to understand how the symbol is being used.
        Particularly valuable before modifying or refactoring code elements.
        Works across all programming languages and file types.
        Arguments:
            symbol_identifier: exact name of the function, class, method, or variable to locate
        Output:
            comprehensive listing of files and line numbers with surrounding context for each reference
        """
        try:
            cmd = f"grep -rn --binary-files=without-match '{symbol_identifier}' . | head -100"
            result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=30)
            refs = result.stdout.strip()

            if not refs:
                return f"No references discovered for symbol '{symbol_identifier}' in the codebase."

            lines = refs.split('\n')
            if len(lines) > 50:
                summary = f"Found {len(lines)} references for '{symbol_identifier}' (showing first 50):\n\n"
                return summary + '\n'.join(lines[:50]) + f"\n\n... and {len(lines) - 50} more references (refine search if needed)"
            return f"References for '{symbol_identifier}' ({len(lines)} found):\n{refs}"
        except subprocess.TimeoutExpired:
            return f"Search timeout: Symbol '{symbol_identifier}' search took too long. Try a more specific identifier."
        except Exception as e:
            return f"Error locating symbol references: {str(e)}"

    @ToolManager.tool
    def get_function_body(self, file_path: str, function_name: str) -> str:
        """
        Retrieves the complete body of a function from a file, including decorators.
        Arguments:
            file_path: filesystem path to target file.
            function_name: name of the function to retrieve (supports both qualified names like "ClassName.method_name" and simple names like "method_name").
        Output:
            The complete function body including decorators, or empty string if function not found.
        """
        if not hasattr(self, 'code_parser'):
            self.code_parser = CodeParseUtil()
        return self.code_parser.get_function_body(file_path, function_name, add_line_numbers=True)

    @ToolManager.tool
    def search_in_file(self, file_path: str, search_term: str) -> str:
        """
        Locates text patterns within a specific file
        Arguments:
            file_path: target file for pattern matching.
            search_term: text pattern to find (e.g., "def test_function", "*SomeClass*")
        Output:
            matching locations with line numbers, or error description
        """
        return self.search_manager.search_in_file(file_path=file_path, search_term=search_term)

    @ToolManager.tool
    def log_strategy(self, approach: str, reasoning: str) -> str:
        """Record a high-level strategy before attempting it.

        Use this BEFORE making significant code changes to log your planned approach. This creates
        a history that persists across rollbacks, preventing you from retrying failed strategies.

        Arguments:
            approach: Brief description of the approach
            reasoning: Why you think this will work

        Output:
            Confirmation with strategy ID for later reference.
        """
        self.strategy_counter += 1
        strategy = {
            "id": self.strategy_counter,
            "approach": approach,
            "reasoning": reasoning,
            "success": None,
            "reason": None,
            "timestamp": time.time(),
            "created_step": len(getattr(self, "tool_invocations", {})),
        }
        self.strategies.append(strategy)
        return f"Strategy #{self.strategy_counter} logged: {approach}\nReasoning: {reasoning}\nUse mark_strategy_outcome to record results."

    @ToolManager.tool
    def create_hypothesis(self, description: str, evidence: str) -> str:
        """Create a hypothesis about the bug's root cause.

        Use this when you have a theory about what's causing the issue. This creates
        a trackable hypothesis that persists across rollbacks - critical for systematic debugging.

        Arguments:
            description: What you think is causing the bug (e.g., "Missing null check in parse_config")
            evidence: What evidence supports this theory (e.g., "Line 45 doesn't handle None input")

        Output:
            Confirmation with hypothesis ID for tracking.
        """
        self.hypothesis_counter += 1
        hypothesis = {
            "id": self.hypothesis_counter,
            "description": description,
            "evidence": evidence,
            "status": "untested",
            "findings": None,
            "created_step": self._current_step,
            "tested_step": None,
            "timestamp": time.time(),
        }
        self.hypotheses.append(hypothesis)
        return f"Hypothesis #{self.hypothesis_counter} created: {description}\nEvidence: {evidence}\nStatus: untested\nUse test_hypothesis to record findings after testing."

    @ToolManager.tool
    def list_hypotheses(self) -> str:
        """View all hypotheses with their test status.

        Use this to review what theories you've already considered and tested. Especially useful:
        - After a rollback (to see what you learned before rolling back)
        - When stuck (to avoid retrying rejected hypotheses)
        - During metacognitive reflection checkpoints

        Arguments:
            None

        Output:
            Formatted list of all hypotheses with status and findings.
        """
        if not self.hypotheses:
            return "No hypotheses recorded yet. Use create_hypothesis to log theories about the bug."
        output = ["=== HYPOTHESIS TRACKER ===\n"]
        untested = [h for h in self.hypotheses if h["status"] == "untested"]
        confirmed = [h for h in self.hypotheses if h["status"] == "confirmed"]
        rejected = [h for h in self.hypotheses if h["status"] == "rejected"]
        inconclusive = [h for h in self.hypotheses if h["status"] == "inconclusive"]
        output.append(
            f"Summary: {len(confirmed)} confirmed, {len(rejected)} rejected, {len(inconclusive)} inconclusive, {len(untested)} untested\n"
        )
        for status, hypotheses in [
            ("✅ CONFIRMED", confirmed),
            ("❌ REJECTED", rejected),
            ("❓ INCONCLUSIVE", inconclusive),
            ("🔍 UNTESTED", untested),
        ]:
            if hypotheses:
                output.append(f"\n{status}:")
                for h in hypotheses:
                    output.append(f"\n  [{h['id']}] {h['description']}")
                    output.append(f"      Evidence: {h['evidence']}")
                    if h["findings"]:
                        output.append(f"      Findings: {h['findings']}")
        return "\n".join(output)

    @ToolManager.tool
    def test_hypothesis(self, hypothesis_id: int, outcome: str, findings: str) -> str:
        """Record the result of testing a hypothesis.

        After investigating a hypothesis (running tests, examining code, etc.), record whether
        it was confirmed, rejected, or inconclusive. This builds institutional memory.

        Arguments:
            hypothesis_id: ID from create_hypothesis (e.g., 1, 2, 3)
            outcome: One of 'confirmed', 'rejected', or 'inconclusive'
            findings: What you discovered (e.g., "Confirmed: null check is missing, added it and tests pass")

        Output:
            Updated hypothesis status.
        """
        if outcome not in ["confirmed", "rejected", "inconclusive"]:
            return f"Error: outcome must be 'confirmed', 'rejected', or 'inconclusive', got '{outcome}'"

        for hyp in self.hypotheses:
            if hyp["id"] == hypothesis_id:
                hyp["status"] = outcome
                hyp["findings"] = findings
                hyp["tested_step"] = self._current_step
                status_emoji = {"confirmed": "✅", "rejected": "❌", "inconclusive": "❓"}.get(outcome, "")
                return f"{status_emoji} Hypothesis #{hypothesis_id} marked as {outcome.upper()}\nFindings: {findings}"
        return f"Error: Hypothesis #{hypothesis_id} not found"

    @ToolManager.tool
    def create_new_file(
        self,
        file_path: str,
        content: str,
        overwrite: bool = False,
    ) -> str:
        """
        Creates a new file with the specified content.

        Arguments:
            file_path: Path where the new file should be created.
            content: The content to write into the file.
            overwrite: If True, will overwrite the file if it exists. If False and file exists, returns an error.

        Output:
            Status message indicating success or error.
        """
        if os.path.exists(file_path) and not overwrite:
            return f"Error: File '{file_path}' already exists. Set overwrite=True to overwrite."

        try:
            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            if hasattr(self, "file_ops") and hasattr(self.file_ops, "new_files_created"):
                self.file_ops.new_files_created.append(file_path)
            return f"File '{file_path}' created successfully."
        except Exception as e:
            return f"Error creating file '{file_path}': {e}"

    @ToolManager.tool
    def list_attempted_strategies(self) -> str:
        """View all strategies tried, with outcomes.

        Use this to review what approaches you've already attempted. Critical for:
        - Avoiding retry loops (especially after rollbacks)
        - Understanding what doesn't work
        - Building on partially successful strategies

        Arguments:
            None

        Output:
            Formatted list of all strategies with outcomes.
        """
        if not self.strategies:
            return "No strategies recorded yet. Use log_strategy before attempting significant changes."
        output = ["=== STRATEGY HISTORY ===\n"]
        succeeded = [s for s in self.strategies if s["success"] is True]
        failed = [s for s in self.strategies if s["success"] is False]
        pending = [s for s in self.strategies if s["success"] is None]
        output.append(
            f"Summary: {len(succeeded)} succeeded, {len(failed)} failed, {len(pending)} pending\n"
        )
        for status, strategies in [
            ("SUCCEEDED", succeeded),
            ("FAILED", failed),
            ("PENDING", pending),
        ]:
            if strategies:
                output.append(f"\n{status}:")
                for s in strategies:
                    output.append(f"\n  [{s['id']}] {s['approach']}")
                    output.append(f"      Reasoning: {s['reasoning']}")
                    if s['reason']:
                        output.append(f"      Outcome: {s['reason']}")
        return "\n".join(output)

    @ToolManager.tool
    def mark_strategy_outcome(self, strategy_id: int, success: bool, reason: str) -> str:
        """Record whether a strategy worked.

        After attempting a strategy, record the outcome. This is crucial for institutional memory,
        especially when using rollbacks - you'll know what you already tried even after reverting changes.

        Arguments:
            strategy_id: ID from log_strategy (e.g., 1, 2, 3)
            success: True if approach worked (tests passed, bug fixed), False otherwise
            reason: Why it succeeded/failed (e.g., "Tests passed but introduced new bug in edge case")

        Output:
            Updated strategy status.
        """
        for strat in self.strategies:
            if strat["id"] == strategy_id:
                strat["success"] = success
                strat["reason"] = reason
                strat["completed_step"] = len(getattr(self, "tool_invocations", {}))
                status = "SUCCEEDED" if success else "FAILED"
                return f"Strategy #{strategy_id} marked as {status}\nReason: {reason}"
        return f"Error: Strategy #{strategy_id} not found"

    @ToolManager.tool
    def finish(self):
        """
        Signals completion of the current workflow execution. Validates patch application and solution verification before finishing.
        Arguments:
            None
        Output:
            Review patch prompt with validation results, or "finish" if all checks pass
        """
        # Deterministic finish gating (generalized).
        # Prevent finishing without a passing run_tests after the last edit.
        if self.is_fix_task:
            if self._last_run_tests_step is None:
                return textwrap.dedent(
                    """
                    ⚠️ VERIFICATION REQUIRED - Cannot Finish Yet

                    You have not executed `run_tests`. Run the repository-defined verification workflow, ensure it passes, then call `finish` again.
                    """
                ).strip()
            if self._last_edit_step is not None and self._last_run_tests_step < self._last_edit_step:
                return textwrap.dedent(
                    """
                    ⚠️ VERIFICATION REQUIRED - Cannot Finish Yet

                    You edited code after your last verification run. Run `run_tests` again after the last edit and ensure it passes, then call `finish`.
                    """
                ).strip()
            if self._last_run_tests_passed is False:
                return textwrap.dedent(
                    """
                    ⚠️ VERIFICATION FAILED - Cannot Finish Yet

                    Your latest verification run did not pass. Fix the failures, re-run `run_tests`, then call `finish`.
                    """
                ).strip()

        # Validate patch application before finishing
        validation_result = self.validate_patch_application()

        # Generate review patch prompt based on validation result
        if "Patch validation passed" not in validation_result:
            if "Patch validation failed" in validation_result or "Patch validation error" in validation_result:
                review_prompt = textwrap.dedent(
                    """
                    ⚠️ Patch Validation: FAILED
                    
                    The patch validation detected issues that may prevent successful application.
                    Please review and fix the following issues before finalizing:
                    
                    {validation_result}
                    
                    Common fixes:
                    - Replace raw newlines (actual line breaks) in strings with \\n escape sequences
                    - Ensure unified diff format is correct (proper @@ hunk headers with line counts)
                    - Remove control characters (\\r, \\0) from file content
                    - Check for encoding issues (ensure UTF-8 encoding)
                    - Verify file paths in the patch match actual file locations
                    
                    After fixing the issues, you can call validate_patch_application again to verify.
                    """
                ).format(validation_result=validation_result)
            else:
                # Validation was skipped or returned a message
                review_prompt = textwrap.dedent(
                    """
                    ℹ️ Patch Validation: {validation_result}
                    
                    Please review your changes before finalizing.
                    """
                ).format(validation_result=validation_result)
            return review_prompt.strip()

        # For fix_task, use SolutionVerifier to check regression and bug fix
        if self.is_fix_task and self.solution_verifier:
            regression_review = self.solution_verifier.verify_solution()

            if "REGRESSION_AND_BUG_CHECK_PASSED" in regression_review:
                # Both conditions verified - return finish
                print("✅ Regression and bug check PASSED - proceeding to finish")
                return "finish"
            else:
                regression_feedback = textwrap.dedent("""
                    ⚠️ **VERIFICATION FAILED - Cannot Finish Yet**
                    
                    Your solution is not ready to finish. Please address the following issues:
                    
                    {regression_feedback}
                    
                    **CRITICAL REQUIREMENTS (BOTH must be true)**:
                    
                    ✅ **CONDITION 1**: Original bug must be FIXED
                        - The hidden tests (originally failing) must now PASS
                        - Verify with actual test runs, not just theory
                    
                    ✅ **CONDITION 2**: NO regressions introduced
                        - ALL tests that were passing before must STILL pass
                        - Run FULL test suite, not just specific test cases
                        - Fix ALL failures, do not rationalize or ignore any
                    
                    **BEWARE OF RETURNED BUG**:
                    - Do NOT fix regressions in a way that breaks the original bug fix
                    - Your final solution must fix BOTH the original bug AND all regressions
                    - The final test run must show ALL tests passing (both hidden tests and regression tests)
                    
                    **REQUIRED ACTIONS**:
                    1. Identify which condition failed (original bug, regressions, or both)
                    2. Fix the issue without breaking the other condition
                    3. Run the FULL test suite to verify BOTH conditions are met
                    4. Only call `finish` again when ALL tests pass (no fail in output)
                    
                    After fixing the issues, call `finish` again for re-verification.
                """).format(regression_feedback=regression_review).strip()

                print("❌ Regression/bug check FAILED - returning feedback to agent")
                print("Feedback:", regression_feedback)
                
                return regression_feedback
        
        # For CREATE tasks or if all checks pass, return finish
        return "finish"

    def validate_patch_application(self) -> str:
        """
        Validates that the current patch can be applied successfully without errors.
        This tool tests patch application by: generating patch from clean state, applying it to clean state,
        and verifying it works. This prevents generating patches that fail when applied later due to issues
        like raw newlines in strings, malformed unified diff format, or incorrect hunk line counts.
        Arguments:
            None
        Output:
            Validation result - "Patch validation passed" if successful, or detailed error message
            explaining why the patch failed to apply (e.g., corrupt patch, line count mismatches,
            raw newlines in strings, etc.)
        """
        try:
            # First, unstage any previously staged changes to ensure clean patch generation
            subprocess.run(["git", "reset", "HEAD"], capture_output=True, text=True, timeout=10, check=False)

            # Get list of modified files (both staged and unstaged) using git status --porcelain
            # This is more reliable than git ls-files or git diff, especially in detached HEAD state
            status_result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=30, check=False)

            if status_result.returncode != 0 or not status_result.stdout.strip():
                return "Patch validation skipped: No modified files to validate"

            # Parse porcelain output: first 2 chars are status, then space(s), then filename
            # M = modified, A = added, D = deleted, R = renamed
            # Status format: XY where X = index status, Y = working tree status
            # We want files that are modified (M) or added (A) in working tree or index
            # Note: We exclude deleted (D) and untracked (??) files as they can't be patched
            modified_files = []
            for line in status_result.stdout.splitlines():
                line = line.rstrip()  # Only strip right side to preserve leading spaces if any
                if not line or len(line) < 3:
                    continue

                # Get status code (first 2 characters)
                status = line[:2]

                # Handle renamed files (format: "R100 old_file -> new_file" or "R  old_file -> new_file")
                if status[0] == "R" or status[1] == "R":
                    # For renamed files, we want the new filename (after ->)
                    if " -> " in line:
                        filepath = line.split(" -> ", 1)[1].strip()
                    else:
                        # Fallback: skip renamed files if format is unexpected
                        continue
                else:
                    # Regular files: format is "XY filename" where XY is 2-char status
                    # After the 2-char status, there are one or more spaces, then the filename
                    # Examples:
                    #   " M filename" (leading space + M + space + filename)
                    #   "M  filename" (M + 2 spaces + filename)
                    #   "MM filename" (MM + space + filename)
                    #   "?? filename" (untracked)

                    # Find first non-space character after the 2-char status
                    # This should be the start of the filename (or a quote if filename has spaces)
                    filename_start = 2
                    while filename_start < len(line) and line[filename_start] == " ":
                        filename_start += 1

                    if filename_start >= len(line):
                        # No filename found, skip
                        continue

                    # Get filename part
                    remaining = line[filename_start:].strip()

                    # Handle quoted filenames (for files with spaces)
                    if remaining.startswith('"') and remaining.endswith('"'):
                        # Remove quotes and unescape
                        filepath = remaining[1:-1].replace('\\"', '"').replace("\\\\", "\\")
                    else:
                        filepath = remaining

                # Check if file is modified (M) or added (A) - exclude deleted (D) and untracked (??)
                # We only want files we can actually patch
                if any(c in status for c in ["M", "A"]) and "D" not in status and "?" not in status:
                    modified_files.append(filepath)

            if not modified_files:
                return "Patch validation skipped: No modified files to validate"

            # Exclude agent files
            exclude = {"src/agent.py", "src/agent_runner.py", "sitecustomize.py"}
            try:
                for _p in getattr(self, "generated_test_files", []):
                    if os.path.exists(_p) and os.path.isfile(_p):
                        exclude.add(os.path.relpath(_p))
            except Exception:
                pass

            modified_files = [f for f in modified_files if f not in exclude]

            if not modified_files:
                return "Patch validation skipped: No relevant modified files to validate"

            # Stash current changes to get a clean state
            stash_result = subprocess.run(
                ["git", "stash", "push", "-m", "temp_validation_stash", "--"] + modified_files,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            if stash_result.returncode != 0:
                return f"Patch validation error: Failed to stash changes. {stash_result.stderr}"

            stash_applied = True

            try:
                # Generate patch from the stash (which contains our changes)
                # The stash contains the diff we want to validate
                stash_diff_result = subprocess.run(
                    ["git", "stash", "show", "-p", "--no-color", "--unified=3", "stash@{0}"], capture_output=True, text=True, timeout=30, check=False
                )

                if stash_diff_result.returncode != 0:
                    # Fallback: restore from stash, stage, generate patch, then restore stash
                    subprocess.run(["git", "stash", "pop"], capture_output=True, text=True, timeout=10, check=False)
                    stash_applied = False

                    # Stage the files
                    if modified_files:
                        subprocess.run(["git", "add", "--"] + modified_files, capture_output=True, text=True, timeout=30, check=False)

                    # Generate patch
                    diff_result = subprocess.run(
                        ["git", "diff", "--cached", "--no-color", "--unified=3"], capture_output=True, text=True, timeout=30, check=False
                    )

                    # Unstage
                    subprocess.run(["git", "reset", "HEAD"], capture_output=True, text=True, timeout=10, check=False)

                    # Stash again
                    stash_result2 = subprocess.run(
                        ["git", "stash", "push", "-m", "temp_validation_stash", "--"] + modified_files,
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    if stash_result2.returncode == 0:
                        stash_applied = True

                    patch_text = diff_result.stdout or ""
                else:
                    patch_text = stash_diff_result.stdout or ""

                if not patch_text.strip():
                    # Restore and return
                    if stash_applied:
                        subprocess.run(["git", "stash", "pop"], capture_output=True, text=True, timeout=10, check=False)
                        stash_applied = False
                    return "Patch validation skipped: No changes to validate (empty patch)"

                # Create a temporary patch file
                with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as tmp_file:
                    tmp_file.write(patch_text)
                    patch_file = tmp_file.name

                try:
                    # Test patch application using git apply --check (dry run)
                    check_result = subprocess.run(["git", "apply", "--check", patch_file], capture_output=True, text=True, timeout=30, check=False)

                    if check_result.returncode == 0:
                        # If check passes, try actual application
                        apply_result = subprocess.run(["git", "apply", patch_file], capture_output=True, text=True, timeout=30, check=False)

                        # Reset to clean state after applying
                        subprocess.run(["git", "reset", "--hard", "HEAD"], capture_output=True, text=True, timeout=10, check=False)
                        subprocess.run(["git", "clean", "-fd"], capture_output=True, text=True, timeout=10, check=False)

                        if apply_result.returncode == 0:
                            # Restore stashed changes
                            subprocess.run(["git", "stash", "pop"], capture_output=True, text=True, timeout=10, check=False)
                            return "Patch validation passed: Patch can be applied successfully"
                        else:
                            error_msg = apply_result.stderr.strip() or apply_result.stdout.strip() or "Unknown error"
                            # Restore stashed changes even on failure
                            subprocess.run(["git", "stash", "pop"], capture_output=True, text=True, timeout=10, check=False)
                            return f"Patch validation failed: Patch cannot be applied. Error: {error_msg}\n\nCommon causes:\n- Raw newlines inside quoted strings (use \\n instead)\n- Malformed unified diff format\n- Incorrect hunk line counts\n- Control characters (\\r, \\0) in file content\n- Encoding issues"
                    else:
                        error_msg = check_result.stderr.strip() or check_result.stdout.strip() or "Unknown error"
                        # Restore stashed changes
                        subprocess.run(["git", "stash", "pop"], capture_output=True, text=True, timeout=10, check=False)
                        return f"Patch validation failed: Patch check failed. Error: {error_msg}\n\nCommon causes:\n- Raw newlines inside quoted strings (use \\n instead)\n- Malformed unified diff format\n- Incorrect hunk line counts\n- Control characters (\\r, \\0) in file content\n- Encoding issues"
                finally:
                    # Clean up temporary patch file
                    try:
                        os.unlink(patch_file)
                    except Exception:
                        pass
            except Exception as e:
                # Restore stashed changes if something went wrong
                if stash_applied:
                    try:
                        subprocess.run(["git", "reset", "--hard", "HEAD"], capture_output=True, text=True, timeout=10, check=False)
                        subprocess.run(["git", "clean", "-fd"], capture_output=True, text=True, timeout=10, check=False)
                        subprocess.run(["git", "stash", "pop"], capture_output=True, text=True, timeout=10, check=False)
                    except Exception:
                        pass
                return f"Patch validation error during application test: {str(e)}"
        except Exception as e:
            return f"Patch validation error: {str(e)}"

    @ToolManager.tool
    def run_shell_cmd(self, command: str) -> str:
        '''
        Runs shell commands for the repository. This tool executes shell commands directly.
        Arguments:
            command: A shell command to be run.
        Output:
            The stdout results of the command. Your working directory is the root of the project.
        '''
        if not command:
            return "Error: No command provided."

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=150
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output
        except subprocess.TimeoutExpired:
            return f"Error: Command '{command}' timed out after 150 seconds"
        except Exception as e:
            return f"Error running command: {str(e)}"

    @ToolManager.tool
    def finish_find_files_to_fix(self, files: List[str]):
        """
        Signals completion of the file finding workflow execution
        Arguments:
            files: The list of files to fix.
        """
        self.files_to_fix = files
        return files

    @ToolManager.tool
    def finish_root_cause_analysis(self, files: List[str], detailed_investigation_notes: str) -> str:
        """
        Signals completion of the root cause analysis workflow execution
        Arguments:
            files: The list of files to modify.
            detailed_investigation_notes: Detailed and comprehensive explanations describing (a) why each file is relevant, (b) how each file relates to the problem, and (c) how the files are connected or interact in the context of the issue.
        """
        output = f"Files to modify: {files}\nDetailed investigation notes: {detailed_investigation_notes}"
        return output
# ============================================================================
# WORKFLOW FUNCTIONS (in execution order)
# ============================================================================

def fix_task_solve_workflow(
    problem_statement: str,
    *,
    timeout: int,
    run_id_1: str,
    enhancement: str,
    n_max_steps=MAX_FIX_TASK_STEPS,
    initial_checkpoint=None,
    should_review: bool = True,
    initial_structure: Optional[Dict[str, str]] = None,
    function_behaviours: Optional[Dict[str, str]] = None,
    files_to_modify: Optional[List[str]] = None,
    root_cause_analysis: Optional[str] = None,
    models: Optional[List] = None,
):
    if files_to_modify is None:
        files_to_modify = []
    repo_path = os.getenv("REPO_PATH", "/sandbox/repo")
    repod_dir = repo_path.split("/")[-1]
    if os.path.exists(repod_dir):
        logger.info(f"📂 [WORKFLOW] Changing to repo directory: {repod_dir}")
        os.chdir(repod_dir)
    logger.info("⚙️ [WORKFLOW] Setting up agent environment...")

    set_env_for_agent()

    global run_id, _current_tool_manager
    print("🎯 [WORKFLOW] fix_task_solve_workflow started")
    logger.info("🎯 [WORKFLOW] fix_task_solve_workflow started")
    run_id = run_id_1
    logger.info(f"🆔 [WORKFLOW] Run ID set: {run_id}")
    
    # ========== PROBLEM DECOMPOSITION PHASE ==========
    # Run structured analysis before the main agent loop
    decomposition = None
    decomposition_text = ""
    try:
        logger.info("🔍 [WORKFLOW] Starting problem decomposition...")
        decomposition = _problem_decomposer.decompose(problem_statement)
        decomposition_text = _problem_decomposer.format_for_prompt(decomposition)
        logger.info("✅ [WORKFLOW] Problem decomposition completed")
    except Exception as e:
        logger.warning(f"⚠️ [WORKFLOW] Problem decomposition failed: {e}")
        pass  # Decomposition is optional enhancement, don't fail on errors
    
    logger.info("🧠 [WORKFLOW] Initializing COT...")
    cot = COT(
        latest_observations_to_keep=LATEST_OBSERVATIONS_TO_KEEP,
        summarize_batch_size=SUMMARIZE_BATCH_SIZE,
    )
    logger.info("🛠️ [WORKFLOW] Creating FixTaskToolManager with available tools...")
    tool_manager = FixTaskToolManager(
        available_tools=[
            "generate_test_cases_from_root_cause",
            "list_directory_structure",
            "get_file_content",
            "get_function_body",
            "find_symbol_references",
            "grep_search",
            "search_in_file",
            "apply_code_edit",
            "modify_test_case",
            "create_new_file",
            "run_code",
            "run_tests",
            "log_strategy",
            "mark_strategy_outcome",
            "list_attempted_strategies",
            "create_hypothesis",
            "test_hypothesis",
            "list_hypotheses",
            "finish",
        ],
        initial_structure=initial_structure,
        initial_checkpoint=initial_checkpoint,
        problem_statement=problem_statement,
        should_review=should_review,
        is_fix_task=True,
        cot=cot,
    )
    _current_tool_manager = tool_manager

    # ========== PRE-POPULATE HYPOTHESES FROM DECOMPOSITION ==========
    # Seed the hypothesis system with initial hypotheses from problem analysis
    if decomposition and decomposition.get("initial_hypotheses"):
        logger.info("💡 [WORKFLOW] Pre-populating hypotheses from decomposition...")
        for hyp in decomposition["initial_hypotheses"]:
            if isinstance(hyp, dict) and hyp.get("description"):
                tool_manager.hypothesis_counter += 1
                hypothesis = {
                    "id": tool_manager.hypothesis_counter,
                    "description": hyp.get("description", ""),
                    "evidence": hyp.get("confirming_evidence", "From problem analysis"),
                    "status": "untested",
                    "findings": None,
                    "likelihood": hyp.get("likelihood", 0.5),
                    "supporting_evidence": [hyp.get("confirming_evidence", "")] if hyp.get("confirming_evidence") else [],
                    "contradicting_evidence": [],
                    "rejecting_evidence_hint": hyp.get("rejecting_evidence", ""),
                    "created_step": 0,
                    "tested_step": None,
                    "timestamp": time.time(),
                    "source": "problem_decomposition",
                }
                tool_manager.hypotheses.append(hypothesis)
        logger.info(f"✅ [WORKFLOW] Pre-populated {len(decomposition.get('initial_hypotheses', []))} hypotheses")

    # Store problem decomposition in tool_manager for reference
    tool_manager.problem_decomposition = decomposition

    logger.info("📝 [WORKFLOW] Formatting system prompt...")
    system_prompt = FIX_TASK_SYSTEM_PROMPT.format(
        tools_docs=tool_manager.get_tool_docs(),
        problem_statement=problem_statement,
        format_prompt=FORMAT_PROMPT_FIX,
    )
    
    # Build enhanced problem with decomposition
    enhanced_problem = problem_statement
    if decomposition_text:
        logger.info("📊 [WORKFLOW] Adding decomposition analysis to problem statement...")
        enhanced_problem = problem_statement + "\n\n---\n\n# Structured Problem Analysis\n\n" + decomposition_text
    if enhancement:
        logger.info("✨ [WORKFLOW] Applying enhancement to problem statement...")
        enhanced_problem = enhanced_problem + "\n\n---\n\n# Additional Context\n\n" + enhancement
    logger.info("📋 [WORKFLOW] Creating instance prompt...")
    instance_prompt = enhanced_problem
    if root_cause_analysis:
        instance_prompt += "\n\n---\n\n# Preliminary analysis on the git repository for the given problem statement\n\n" + root_cause_analysis
    print("🚀 [WORKFLOW] Executing agent workflow...")
    logger.info("🚀 [WORKFLOW] Executing agent workflow...")
    fix_models = models if models else [KIMI_MODEL_NAME, GLM_MODEL_NAME]
    patch, is_success = execute_agent_workflow(
        cot,
        tool_manager,
        system_prompt,
        instance_prompt,
        n_max_steps,
        timeout,
        fix_models,
        log_prefix="FIX_MAIN_AGENT",
        initial_structure=initial_structure,
        function_behaviours=function_behaviours,
        files_to_modify=files_to_modify,
    )
    print("✅ [WORKFLOW] fix_task_solve_workflow completed")
    logger.info("✅ [WORKFLOW] fix_task_solve_workflow completed")
    return patch, is_success


def set_env_for_agent():
    logger.debug("Setting up environment for agent")

    work_dir = os.getcwd()
    original_cwd = os.getcwd()

    # Ensure cwd is on PYTHONPATH
    pythonpath = os.environ.get("PYTHONPATH", "")
    if work_dir not in pythonpath.split(":"):
        os.environ["PYTHONPATH"] = f"{work_dir}:{pythonpath}"

    # Optional lib dir
    lib_dir = os.path.join(work_dir, "lib")
    if os.path.exists(lib_dir) and lib_dir not in os.environ["PYTHONPATH"]:
        os.environ["PYTHONPATH"] += f":{lib_dir}"

    # Write sitecustomize.py
    with open(os.path.join(work_dir, "sitecustomize.py"), "w") as f:
        f.write(VERSION_COMPATIBILITY_FIX)

    try:
        os.chdir(work_dir)

        if not os.path.exists(".git"):
            logger.info("Initializing git repository")
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
            subprocess.run(["git", "config", "--global", "user.email", "agent@sandbox.local"], check=True)
            subprocess.run(["git", "config", "--global", "user.name", "sandbox_agent"], check=True)
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], check=False)
        else:
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
    except Exception as e:
        logger.warning(f"Error setting up environment: {e}")
    finally:
        os.chdir(original_cwd)


def validate_initial_structure_implementation(
    initial_structure: Dict[str, str],
    modified_files: set,
    model: Model = GLM_MODEL_NAME,
) -> tuple[bool, str]:
    """
    Validates that the modified files correctly implement the code skeleton from initial_structure.
    Only validates files that are in both modified_files and initial_structure.
    Returns (is_valid, validation_message).
    """
    try:
        # Only validate files that were modified AND are in initial_structure
        files_to_validate = {f for f in modified_files if f in initial_structure}
        
        if not files_to_validate:
            # No modified files to validate, or no overlap with initial_structure
            return True, "No files to validate (no modified files match initial structure)"
        
        # Read current file contents for modified files only
        current_structure = {}
        initial_structure_subset = {}
        for file_path in files_to_validate:
            try:
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        current_structure[file_path] = f.read()
                    initial_structure_subset[file_path] = initial_structure[file_path]
                else:
                    return False, f"File {file_path} from initial structure does not exist."
            except Exception as e:
                return False, f"Error reading file {file_path}: {e}"
        
        # Prepare comparison content - only for modified files
        comparison_content = "## Initial Structure (Expected Code Skeleton)\n\n"
        comparison_content += "NOTE: Only validating files that were modified by the agent.\n\n"
        for file_path, content in initial_structure_subset.items():
            comparison_content += f"### File: {file_path}\n```\n{content}\n```\n\n"
        
        comparison_content += "\n## Current Implementation (Actual Code)\n\n"
        for file_path, content in current_structure.items():
            comparison_content += f"### File: {file_path}\n```\n{content}\n```\n\n"
        
        # Create validation prompt
        validation_prompt = textwrap.dedent("""
        You are a code validation expert. Your task is to STRICTLY validate whether the current implementation 
        correctly and EXACTLY implements the code skeleton provided in the initial structure.
        
        The validation must check that the code skeleton structure matches EXACTLY:
        1. All classes from the initial structure are present with the same names and based from the same base class
        2. All functions/methods from the initial structure are present with the same names
        3. All function/method parameters from initial structure match exactly (same parameter names, same order, same types if specified)
        4. The class hierarchy and inheritance structure matches exactly
        
        CRITICAL REQUIREMENTS:
        - The functions in code skeleton (class definitions, function signatures, parameter structures) must match EXACTLY
        - Implementation details (function bodies, logic) can differ, but the STRUCTURE must be identical
        - Any missing classes, functions, or parameters should be flagged as validation failure
        - Any renamed classes, functions, or parameters should be flagged as validation failure
        - Any changes to parameter order, names, or structure should be flagged as validation failure
        
        Things that are allowed:
        - *You can add as many new functions/methods as you wish that is not exist in code skeleton.*
        
        IMPORTANT: This is a STRICT structural validation. The skeleton must match exactly. Only implementation 
        details within function bodies can differ.
        
        Respond with a JSON object in this exact format:
        {{
            "is_valid": true/false,
            "message": "Detailed validation message explaining the result",
            "issues": ["List of specific issues found, if any"]
        }}
        
        {comparison_content}
        """).format(comparison_content=comparison_content)
        
        messages = [
            {
                "role": "system",
                "content": "You are a code validation expert. Respond only with valid JSON.",
            },
            {"role": "user", "content": validation_prompt},
        ]
        
        # Call LLM for validation
        response, _ = Network.make_request(messages, model=model)
        
        # Parse response
        try:
            # Extract JSON from response - try to find JSON object with balanced braces
            response_cleaned = response.replace("```json", "").replace("```", "").strip()
            # Try to find JSON object starting with {
            json_start = response_cleaned.find("{")
            if json_start >= 0:
                # Use the existing balanced braces extraction method
                json_str = Network._extract_balanced_braces(response_cleaned, json_start)
                if json_str:
                    response_cleaned = json_str
            validation_result = json.loads(response_cleaned)
            
            is_valid = validation_result.get("is_valid", False)
            message = validation_result.get("message", "Validation completed")
            issues = validation_result.get("issues", [])
            
            if issues:
                message += "\n\nIssues found:\n" + "\n".join(f"- {issue}" for issue in issues)
            
            return is_valid, message
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse validation response: {e}, response: {response}")
            # Fallback: try to determine validity from response text
            if "is_valid" in response.lower() and "true" in response.lower():
                return True, "Validation passed (parsed from response)"
            elif "is_valid" in response.lower() and "false" in response.lower():
                return True, f"Validation failed (parsed from response): {response[:500]}"
            else:
                return True, f"Could not parse validation response: {response[:500]}"
                
    except Exception as e:
        logger.error(f"Error in validate_initial_structure_implementation: {e}")
        import traceback
        return True, f"Validation error: {str(e)}\n{traceback.format_exc()}"

def get_files_to_modify(problem_statement: str) -> tuple[str, list[str]]:
    tool_manager = FixTaskToolManager(available_tools=["get_file_content", "list_directory_structure", "finish_find_files_to_fix"])
    FIND_FILES_TO_MODIFY = textwrap.dedent(
        """
        You are a helpful assistant that finds the files to modify related to the problem statement.
        You must check the directory structure using `list_directory_structure` tool and then determine which files are needed for the problem statement.
        You must then use the `finish_find_files_to_fix` tool to signal the completion of the file finding workflow execution.
        
        - Never try to check git, hidden or unncessary files.
        - Never try to implement any new functionality or find solutions for the problem statement.
        - Only focus on finding files that are related to the problem statement and existing files in the repo.
        
        You have access to the following tools:-
        {tools_docs}
        {format_prompt}
        """
    ).format(tools_docs=tool_manager.get_tool_docs(), format_prompt=FORMAT_PROMPT_FIX)
    try:
        cot = COT(latest_observations_to_keep=10, summarize_batch_size=10)
        instance_prompt = f"Problem Statement:\n{problem_statement}"
        result, __cached__ = execute_agent_workflow(cot, tool_manager, FIND_FILES_TO_MODIFY, instance_prompt, 30, 300, [KIMI_MODEL_NAME, GLM_MODEL_NAME], finish_tool_name="finish_find_files_to_fix", log_prefix="FINISH_FIND_FILES_TO_MODIFY")
        if not result:
            return "", []
        if not isinstance(result, list):
            result = [result]
        contents = []
        for file_path in result:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    contents.append(f"{file_path}\n{f.read()}")
            except Exception as e:
                logger.error(f"Failed to open file {file_path}: {e}")
        return "\n\n".join(contents), result
    except Exception as e:
        logger.error(f"Error in get files to modify: {e}")
        return "", []

def validate_implementation_completeness(code: str) -> str:
    try:
        prompt = textwrap.dedent("""
        You are a code validation expert. Your task is to analyze ALL functions/methods in the provided code and determine their implementation status.
        
        Follow these steps:
        1. Find ALL functions/methods defined in the code.
        2. For EACH function/method, analyze its body to determine:
           - is_empty: True if the function body is empty
           - is_only_null_return: True if the function body only returns None/null without doing anything else (no method calls, no assignments, no other operations)
           - reason: Brief explanation of what you found in the function body
        
        You must respond in JSON format with information for ALL functions.
        
        Return a JSON object in this exact format:
        {{
            "functions": [
                {{
                    "name": "function_name or ClassName.method_name",
                    "is_empty": true or false,
                    "is_only_null_return": true or false,
                    "reason": "reason why you think it's incomplete"
                }}
            ]
        }}
        
        Code to Analyze:
        ```
        {code}
        ```
        """).format(
            code=code
        )

        messages = [
            {"role": "user", "content": prompt},
        ]
        
        logger.info("Validating implementation completeness...")
        print("🔍 Checking for incomplete function implementations...")
        
        # Call LLM for validation
        response, _ = Network.make_request(messages, model=QWEN_MODEL_NAME, timeout=120)
        
        # Clean up response - remove markdown code blocks if present
        response_cleaned = response.replace("```json", "").replace("```", "").strip()
        
        # Parse and validate JSON response
        validation_result = json.loads(response_cleaned)
        print(f"[DEBUG] Validation Results:\n\n {json.dumps(validation_result, indent=4)} ")
        
        # Filter to keep only incomplete functions (is_empty=True OR is_only_empty_return=True)
        if "functions" in validation_result:
            all_functions = validation_result["functions"]
            incomplete_functions = [
                func for func in all_functions
                if func.get("is_empty", False) or func.get("is_only_null_return", False)
            ]
            validation_result["functions"] = incomplete_functions
            logger.info(f"Found {len(all_functions)} total functions, {len(incomplete_functions)} incomplete")
            print(f"📊 Analyzed {len(all_functions)} functions, found {len(incomplete_functions)} incomplete")
        
        return validation_result
        
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse validation response: {e}"
        logger.error(error_msg)
        return {
            "functions": [],
            "error": str(e),
            "summary": f"Validation error: {error_msg}"
        }
    except Exception as e:
        error_msg = f"Error in validate_implementation_completeness: {e}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return {
            "functions": [],
            "error": str(e),
            "traceback": traceback.format_exc(),
            "summary": f"Validation error: {error_msg}"
        }

def generate_function_behaviours(initial_structure: str, problem_statement: str) -> str:
    """Generate function behaviours for all functions in one LLM call."""
    try:
        prompt = f"""Problem Statement:
        {problem_statement}

        Initial Structure (Code Skeleton):
        {initial_structure}

        Analyze the code skeleton and provide step-by-step behavior including the final return value for each function/method defined in it.

        Return the response as a JSON dict with the following format:
        {{
            "function_name_1": {{
                "steps": [
                    "Step 1: ...",
                    "Step 2: ...",
                    "Step 3: ..."
                ]
            }},
            "ClassName.method_name": {{
                "steps": [
                    "Step 1: ...",
                    "Step 2: ..."
                ]
            }}
        }}

        Important guidelines:
        - For standalone functions: use just the function name as the key
        - For class methods: use "ClassName.method_name" format as the key
        - Each function should have a "steps" array containing strings
        - Each step should be a clear, detailed description of what the function does
        - Focus on the logical flow and behavior, not implementation details
        - Only include must required steps that is related to final result.
        - Be comprehensive and specific
        - Include the final return value of the function/methods
        """
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        logger.info("Generating function behaviours in a single LLM call...")
        print("🚀 Generating function behaviours for all functions...")
        
        # Call LLM once for all functions
        response, _ = Network.make_request(messages, model=QWEN_MODEL_NAME, timeout=300)
        
        # Clean up response - remove markdown code blocks if present
        response_cleaned = response.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON response
        function_behaviours = json.loads(response_cleaned)
        
        logger.info(f"✅ Successfully generated behaviours for {len(function_behaviours)} functions")
        print(f"✅ Generated behaviours for {len(function_behaviours)} functions")
        
        return function_behaviours
        
    except Exception as e:
        logger.error(f"Error in generate_function_behaviours: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}

def check_not_defined_functions(code: str) -> dict:
    try:
        prompt = f"""Analyze the following code and identify HELPER/UTILITY functions/methods that are CALLED but NOT implemented in the code.

        Code to analyze:
        ```
        {code}
        ```
        
        Steps to follow:
        1. Working through the codebase, find all function/method calls and list them in the reasoning steps.
        2. For each function/method call
            - Check if it is a standard library or third-party library function. If yes, continue with the next call.
            - Check if it is a direct call to the parent class method. If yes, continue with the next call.
            - Check if it is a call to a method of the class that is not defined in the code. If yes, continue with the next call.
            - Do not consider a parent class method as defined unless it is explicitly called using super or the parent class name. If the code calls that method is not implemented in the current class, flag it as undefined even if the parent class has it.
        
        CRITICAL: Return ONLY valid JSON in the exact format shown below. No explanations, reasoning, or markdown.
        
        Required JSON format (return this EXACTLY):
        {{
            "reasonings": [
                "reasoning step 1",
                "reasoning step 2"
            ],
            "undefined_functions": [
                {{
                    "name": "function_name",
                    "code_snippet": "result = function_name(x, y)"
                }}
            ]
        }}
        """
                
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        logger.info("🔍 Calling LLM to check for undefined functions...")
        print("🔍 Analyzing code for undefined functions using LLM...")
        
        # Call LLM
        response, _ = Network.make_request(
            messages, 
            model=QWEN_MODEL_NAME, 
            timeout=120
        )
        
        # Clean up response - remove markdown code blocks if present
        response_cleaned = response.replace("```json", "").replace("```", "").strip()
        
        try:
            return json.loads(response_cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response_cleaned[:500]}")
            return {
                "undefined_functions": []
            }
        
    except Exception as e:
        error_msg = f"Error in check_not_defined_functions: {e}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return {
            "undefined_functions": []
        }

def validate_implementation_and_dependencies(code: str) -> dict:
    try:
        print(f"Code to validate: \n\n{code}")
        # Step 1: Check implementation completeness
        completeness_result = validate_implementation_completeness(code)
        incomplete_functions = completeness_result.get("functions", [])
        
        # Step 2: Check for undefined functions
        undefined_result = check_not_defined_functions(code)
        undefined_functions = undefined_result.get("undefined_functions", [])
        print(f"Undefined Result: \n\n {json.dumps(undefined_result, indent=4)}")
        # undefined_functions = []
        
        # Step 3: Check for logic errors
        logic_issues = []
        # logic_issues = []
        
        # Combine results
        return {
            "incomplete_functions": incomplete_functions,
            "undefined_functions": undefined_functions,
            "logic_issues": logic_issues,
            "has_issues": len(incomplete_functions) > 0 or len(undefined_functions) > 0 or len(logic_issues) > 0
        }
    except Exception as e:
        logger.error(f"Error in validate_implementation_and_dependencies: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "incomplete_functions": [],
            "undefined_functions": [],
            "logic_issues": [],
            "has_issues": False,
            "error": str(e)
        }

def execute_agent_workflow(
    cot: COT,
    tool_manager: ToolManager,
    system_prompt: str,
    instance_prompt: str,
    n_max_steps: int,
    timeout: int,
    models: List[str],
    log_prefix: str = "AGENT",
    finish_tool_name="finish",
    initial_structure: Optional[Dict[str, str]] = None,
    function_behaviours = None,
    files_to_modify: Optional[List[str]] = None,
    reject_observation_token_threshold: int = 50000,
    save_observation_to_file_token_threshold: int = 4000,
) -> tuple[str, bool]:
    if files_to_modify is None:
        files_to_modify = []
    global run_id
    print(f"🚀 [WORKFLOW] execute_agent_workflow started (max_steps={n_max_steps}, timeout={timeout}s)")
    logger.info(f"🚀 [WORKFLOW] execute_agent_workflow started (max_steps={n_max_steps}, timeout={timeout}s)")
    logger.info(f"{log_prefix} Starting agent execution... ")
    start_time = time.time()
    raw_text = ""
    total_attempts = 0
    error_counter = {}
    next_thought = None
    next_tool_name = None
    next_tool_args = None
    modified_files = set()
    files_with_syntax_errors = set()
    current_model_index = 0

    def _safe_call_tool(tool_manager: ToolManager, tool_name: str, tool_args):
        tool_fn = tool_manager.get_tool(tool_name)
        if isinstance(tool_fn, str):
            return tool_fn

        if tool_args is None or tool_args == {}:
            return tool_fn()
        if not isinstance(tool_args, dict):
            return tool_fn()
        try:
            sig = inspect.signature(tool_fn)
            allowed = set(sig.parameters.keys())
            allowed.discard("self")
        except Exception:
            allowed = set(tool_args.keys())
        cleaned = {k: v for k, v in tool_args.items() if k in allowed}
        try:
            for k in list(cleaned.keys()):
                v = cleaned[k]
                p = sig.parameters.get(k)
                ann = str(getattr(p, "annotation", ""))
                if v is not None and isinstance(v, str) and ("List" in ann or "list" in ann):
                    cleaned[k] = v.split() if v.strip() else []
        except Exception:
            pass
        return tool_fn(**cleaned) if cleaned else tool_fn()

    for step in range(n_max_steps):
        selected_model = models[current_model_index]
        elapsed_time = time.time() - start_time
        logger.info(f"📊 [WORKFLOW] Step {step}/{n_max_steps} - Elapsed: {elapsed_time:.2f}s/{timeout}s")
        logger.info("=" * 40 + f"[{log_prefix}] Step {step}" + "=" * 40)
        if time.time() - start_time > timeout:
            print(f"⏱️ [WORKFLOW] Global timeout reached ({elapsed_time:.2f}s)")
            logger.warning(f"[{log_prefix}] Global timeout reached")
            cot.add_action(
                COT.Action(
                    next_thought="global timeout reached",
                    next_tool_name="",
                    next_tool_args={},
                    observation="",
                    is_error=True,
                    inference_error_counter={},
                    request_data=[],
                )
            )
            break
        logger.info(f"💬 [WORKFLOW] Preparing messages for inference (step {step})...")
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]
        messages.extend(cot.to_str())
        messages.append({"role": "system", "content": STOP_INSTRUCTION})

        if cot.is_thought_repeated():
            logger.warning(f"🔄 [WORKFLOW] Thought repeated {cot.repeated_thoughts} times - adjusting temperature")
            logger.info(f"[TEMPERATURE] Thought repeated {cot.repeated_thoughts} times")
            last_thought = cot.thoughts[-1]
            messages.append(
                {
                    "role": "user",
                    "content": DO_NOT_REPEAT_TOOL_CALLS.format(
                        previous_response=f"next_thought:{last_thought.next_thought}\n next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}"
                    ),
                }
            )
            temperature = 0.5
            if cot.repeated_thoughts >= 2:
                model_idx = (cot.repeated_thoughts - 2) % len(models)
                selected_model = models[model_idx]
                logger.info(f"🔄 [WORKFLOW] Switching to model index {model_idx}")
        else:
            temperature = 0.0
        try:
            logger.info(f"🤖 [WORKFLOW] Calling inference API temp: {temperature})...")
            inference_start_time = time.time()
            models_to_try = [selected_model] + [m for m in models if m != selected_model]
            (
                next_thought,
                next_tool_name,
                next_tool_args,
                raw_text,
                total_attempts,
                error_counter,
                messages,
                used_model,
            ) = Network.inference(messages, model=models_to_try, run_id=run_id, temperature=temperature)

            selected_model = used_model
            inference_duration = time.time() - inference_start_time
            logger.info(f"✅ [WORKFLOW] Inference completed in {inference_duration:.2f}s")
        except Exception as e:
            inference_duration = 0
            print(f"❌ [WORKFLOW] Inference error: {e}")
            logger.error(f"[{log_prefix}] Inference error: {e}")
        tool_names_list = next_tool_name if isinstance(next_tool_name, list) else [next_tool_name]
        tool_args_list = next_tool_args if isinstance(next_tool_args, list) else [next_tool_args]

        tool_manager._current_step = step
        tool_manager._cot_snapshot_cache = [
            {
                "thought": t.next_thought,
                "tool": t.next_tool_name,
                "args": str(t.next_tool_args)[:200],
                "success": not t.is_error,
            }
            for t in cot.thoughts[-10:]
        ]
        # Update cot and solution_verifier if tool_manager is FixTaskToolManager
        if hasattr(tool_manager, 'is_fix_task') and tool_manager.is_fix_task:
            if not tool_manager.cot:
                tool_manager.cot = cot
                tool_manager.solution_verifier = SolutionVerifier(cot=cot, problem_statement=tool_manager.problem_statement)
            elif tool_manager.cot != cot:
                tool_manager.cot = cot
                if tool_manager.solution_verifier:
                    tool_manager.solution_verifier.cot = cot
        all_observations = []
        all_successful = True
        for idx, (tool_name, tool_args) in enumerate(zip(tool_names_list, tool_args_list)):
            logger.info(f"🔧 [WORKFLOW] Executing tool {idx+1}/{len(tool_names_list)}: {tool_name}")
            try:
                if '"' in tool_name or "'" in tool_name:
                    tool_name = tool_name.replace('"', "").replace("'", "")
                observation = _safe_call_tool(tool_manager, tool_name, tool_args)
                if tool_name == "apply_code_edit" and tool_args and "file_path" in tool_args:
                    file_path = tool_args["file_path"]
                    if "ok, code edit applied successfully" in str(observation).lower():
                        modified_files.add(file_path)
                        logger.info(f"✅ [WORKFLOW] Code edit applied successfully to: {file_path}")
                    elif "syntax error" in str(observation).lower():
                        files_with_syntax_errors.add(file_path)
                        logger.error(f"❌ [WORKFLOW] Syntax error detected in: {file_path}")
                
                estimated_tokens = Utils.count_tokens(str(observation))
                if estimated_tokens > reject_observation_token_threshold:
                    observation = f"Error: Tool output from '{tool_name}' exceeded token limit ({estimated_tokens} tokens > 50000 tokens limit). The response is too large to process. Please use more specific queries, target smaller file ranges, or break the request into smaller operations."
                elif estimated_tokens > save_observation_to_file_token_threshold:
                    observation_path, line_count = tool_manager._save_large_observation(str(observation), tool_name)
                    observation = f"Tool output from `{tool_name}` exceeded token limit ({estimated_tokens} tokens > 4000 tokens limit). The full output has been saved to: {observation_path}. You can use search tool to find specific lines in the file and you can read this file using the get_file_content tool, but specify the start and end line numbers to read the file. The file has {line_count} lines."
                all_observations.append(observation)

            except ToolManager.Error as e:
                error_msg = f"Tool {idx+1} ({tool_name}) error: {e.message}"
                all_observations.append(error_msg)
                all_successful = False
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                error_msg = f"Tool {idx+1} ({tool_name}) exception: {str(e)}\n{error_traceback}"
                all_observations.append(error_msg)
                all_successful = False
        # Check for finish tool and validate before creating combined observation
        validation_failed = False
        if finish_tool_name in tool_names_list:
            if finish_tool_name == "finish_find_files_to_fix":
                logger.info("🎯 [WORKFLOW] finish_find_files_to_fix called")
                for name, obs in zip(tool_names_list, all_observations):
                    if name == finish_tool_name:
                        return obs, False
            elif finish_tool_name == "finish_root_cause_analysis":
                for name, obs in zip(tool_names_list, all_observations):
                    if name == finish_tool_name:
                        return obs, True
            elif finish_tool_name == "finish":
                print("🎯 [WORKFLOW] finish tool called")
                logger.info("🎯 [WORKFLOW] finish tool called")
                for name, obs in zip(tool_names_list, all_observations):
                    if name == finish_tool_name:
                        if obs != "finish":
                            break

                        if initial_structure:
                            if tool_manager.validated_num >= 5: # only validate up to 5 times
                                return tool_manager.get_final_git_patch(), True
                            tool_manager.validated_num += 1

                            # Validate before allowing finish
                            validation_failed, validation_observation = validate_before_finish(
                                initial_structure=initial_structure,
                                modified_files=modified_files,
                                files_to_modify=files_to_modify,
                                function_behaviours=function_behaviours
                            )
                            
                            if validation_failed:
                                # Replace the finish observation with validation error
                                for i, (n, o) in enumerate(zip(tool_names_list, all_observations)):
                                    if n == finish_tool_name:
                                        all_observations[i] = validation_observation
                                        break
                                # Break if structure validation failed (early exit)
                                if "does not correctly match the initial structure" in validation_observation:
                                    break

                            if not validation_failed:
                                print("✅ [WORKFLOW] Workflow completed successfully, generating final patch...")
                                logger.info("✅ [WORKFLOW] Workflow completed successfully, generating final patch...")
                                return tool_manager.get_final_git_patch(), True
                        
                        else:
                            return tool_manager.get_final_git_patch(), True
        if len(all_observations) == 1:
            combined_observation = all_observations[0]
        else:
            combined_observation = "\n\n--- Tool Call Results ---\n" + "\n\n".join(
                [f"Tool {i+1} ({tool_names_list[i]}):\n{obs}" for i, obs in enumerate(all_observations)]
            )
            
        cot.add_action(
            COT.Action(
                next_thought=next_thought,
                next_tool_name=next_tool_name,
                next_tool_args=next_tool_args,
                observation=combined_observation,
                is_error=not all_successful or validation_failed,
                raw_response=raw_text,
                total_attempts=total_attempts,
                inference_error_counter=error_counter,
                request_data=messages,
            )
        )
        # If validation failed, continue to next step instead of finishing
        if validation_failed:
            continue
    logger.info("📝 [WORKFLOW] Workflow ended, generating final patch...")
    return tool_manager.get_final_git_patch(), False

def validate_before_finish(
    initial_structure: Optional[Dict[str, str]],
    files_to_modify: List[str],
    modified_files: set,
    function_behaviours: dict
) -> tuple[bool, str]:
    if not initial_structure:
        return False, ""
    
    # Step 1: Validate initial structure
    print("🔍 [WORKFLOW] Validating initial structure implementation...")
    logger.info("🔍 [WORKFLOW] Validating initial structure implementation...")
    logger.info(f"🔍 [WORKFLOW] Modified files: {modified_files}")
    logger.info(f"🔍 [WORKFLOW] Files in initial_structure: {set(initial_structure.keys())}")
    is_valid, validation_message = validate_initial_structure_implementation(initial_structure, modified_files)
    
    if not is_valid:
        print(f"❌ [WORKFLOW] Initial structure validation failed: {validation_message}")
        logger.warning(f"❌ [WORKFLOW] Initial structure validation failed: {validation_message}")
        validation_observation = (
            f"VALIDATION FAILED: The implementation does not correctly match the initial structure.\n\n"
            f"Validation Result:\n{validation_message}\n\n"
            f"Please review the initial structure requirements and ensure all code skeleton elements "
            f"are implemented correctly and exactly as specified. Do not call the finish tool until "
            f"the validation passes."
        )
        return True, validation_observation
    
    # Step 1 passed
    print(f"✅ [WORKFLOW] Initial structure validation passed: {validation_message}")
    logger.info(f"✅ [WORKFLOW] Initial structure validation passed: {validation_message}")
    
    # Step 2: Combined validation (completeness + undefined functions + logic)
    logger.info("🔍 [WORKFLOW] Validating implementation completeness, dependencies, and logic...")
    all_incomplete_functions = []
    all_undefined_functions = []
    all_logic_issues = []
    
    for modified_file in modified_files:
        try:
            if modified_file not in files_to_modify:
                continue

            with open(modified_file, 'r') as f:
                code = f.read()
            
            # Run combined validation
            validation_results = validate_implementation_and_dependencies(code)
            print(f"Validate Implementation and Dependencies results: \n\n{json.dumps(validation_results, indent=4)}")
            
            # Collect incomplete functions
            incomplete_funcs = validation_results.get("incomplete_functions", [])
            all_incomplete_functions.extend([x.get('name', '') for x in incomplete_funcs])
            
            # Collect undefined functions
            undefined_funcs = validation_results.get("undefined_functions", [])
            all_undefined_functions.extend(undefined_funcs)
            
            # Collect logic issues
            logic_issues = validation_results.get("logic_issues", [])
            all_logic_issues.extend(logic_issues)
            
        except Exception as e:
            logger.error(f"Error validating {modified_file}: {e}")
            pass
    
    # Check if there are any issues
    has_incomplete = len(all_incomplete_functions) > 0
    has_undefined = len(all_undefined_functions) > 0
    has_logic_issues = len(all_logic_issues) > 0
    
    validation_failed = has_incomplete or has_undefined or has_logic_issues
    
    logger.info(f"🔍 [WORKFLOW] Validation Results:")
    logger.info(f"  - Incomplete functions: {len(all_incomplete_functions)}")
    logger.info(f"  - Undefined functions: {len(all_undefined_functions)}")
    logger.info(f"  - Logic issues: {len(all_logic_issues)}")
    
    if not validation_failed:
        return False, ""
    
    # Build combined validation observation
    validation_observation = "⚠️ **IMPLEMENTATION ISSUES DETECTED** ⚠️\n\n"
    
    # Section 1: Incomplete Functions
    if has_incomplete:
        validation_observation += f"## 1. INCOMPLETE IMPLEMENTATIONS ({len(all_incomplete_functions)} function(s))\n\n"
        validation_observation += "The following functions need proper implementation:\n\n"
        
        for idx, function_name in enumerate(all_incomplete_functions, 1):
            validation_observation += f"{idx}. **{function_name}**\n"
            expected_behaviour = function_behaviours.get(function_name)
            
            if expected_behaviour:
                if isinstance(expected_behaviour, dict) and "steps" in expected_behaviour:
                    validation_observation += "   Expected behaviour:\n"
                    for step_idx, step in enumerate(expected_behaviour["steps"], 1):
                        if isinstance(step, str):
                            validation_observation += f"   - Step {step_idx}: {step}\n"
                        elif isinstance(step, dict) and 'description' in step:
                            validation_observation += f"   - Step {step_idx}: {step['description']}\n"
                        else:
                            validation_observation += f"   - Step {step_idx}: {step}\n"
                else:
                    validation_observation += f"   Expected behaviour: {expected_behaviour}\n"
            else:
                validation_observation += "   Expected behaviour: Not specified\n"
            validation_observation += "\n"
    
    # Section 2: Undefined Functions
    if has_undefined:
        validation_observation += f"\n## 2. UNDEFINED FUNCTION CALLS ({len(all_undefined_functions)} function(s))\n\n"
        validation_observation += "The following functions are being called but are NOT defined:\n\n"
        
        for idx, undefined_func in enumerate(all_undefined_functions, 1):
            func_name = undefined_func.get("name", "unknown")
            code_snippet = undefined_func.get("code_snippet", "N/A")
            validation_observation += f"{idx}. **{func_name}**\n"
            validation_observation += f"   Used in: `{code_snippet}`\n\n"
    
    # Section 3: Logic Issues
    if has_logic_issues:
        validation_observation += f"\n## 3. LOGIC ERRORS ({len(all_logic_issues)} issue(s))\n\n"
        validation_observation += "The following logic errors were detected:\n\n"
        
        for idx, logic_issue in enumerate(all_logic_issues, 1):
            func_name = logic_issue.get("function_name", "unknown")
            issue_type = logic_issue.get("issue_type", "unknown")
            description = logic_issue.get("description", "N/A")
            problematic_code = logic_issue.get("problematic_code", "N/A")
            
            validation_observation += f"{idx}. **{func_name}** - {issue_type}\n"
            validation_observation += f"   Issue: {description}\n"
            validation_observation += f"   Code: `{problematic_code}`\n\n"
    
    # Action required
    validation_observation += "\n**ACTION REQUIRED:**\n"
    if has_incomplete:
        validation_observation += "- Implement all incomplete functions according to their expected behaviours\n"
    if has_undefined:
        validation_observation += "- Define all missing functions or fix the function calls\n"
    if has_logic_issues:
        validation_observation += "- Fix all logic errors and return type mismatches\n"
    validation_observation += "\nDo not call the finish tool until all issues are resolved."
    
    print(f"🔍 [WORKFLOW] Validation Observation: \n\n{validation_observation}")
    
    return validation_failed, validation_observation

def create_task_solve_workflow(
    problem_statement: str,
    *,
    timeout: int,
    run_id_1: str,
    enhancement: str,
    n_max_steps=MAX_FIX_TASK_STEPS,
    initial_checkpoint=None,
    should_review: bool = True,
    initial_structure: Optional[Dict[str, str]] = None,
    function_behaviours: Optional[Dict[str, str]] = None,
    files_to_modify: Optional[List[str]] = None,
):
    if files_to_modify is None:
        files_to_modify = []
    repo_path = os.getenv("REPO_PATH", "/sandbox/repo")
    repod_dir = repo_path.split("/")[-1]
    if os.path.exists(repod_dir):
        logger.info(f"📂 [WORKFLOW] Changing to repo directory: {repod_dir}")
        os.chdir(repod_dir)
    logger.info("⚙️ [WORKFLOW] Setting up agent environment...")

    set_env_for_agent()

    global run_id, _current_tool_manager
    print("🎯 [WORKFLOW] create_task_solve_workflow started")
    logger.info("🎯 [WORKFLOW] create_task_solve_workflow started")
    run_id = run_id_1
    logger.info(f"🆔 [WORKFLOW] Run ID set: {run_id}")
    logger.info("🧠 [WORKFLOW] Initializing COT...")
    cot = COT(
        latest_observations_to_keep=LATEST_OBSERVATIONS_TO_KEEP,
        summarize_batch_size=SUMMARIZE_BATCH_SIZE,
    )
    logger.info("🛠️ [WORKFLOW] Creating FixTaskToolManager with available tools...")
    tool_manager = FixTaskToolManager(
        available_tools=[
            "generate_test_cases_from_root_cause",
            "list_directory_structure",
            "get_file_content",
            "get_function_body",
            "find_symbol_references",
            "grep_search",
            "search_in_file",
            "apply_code_edit",
            "modify_test_case",
            "create_new_file",
            "run_code",
            "run_tests",
            "think",
            "log_strategy",
            "mark_strategy_outcome",
            "list_attempted_strategies",
            "create_hypothesis",
            "test_hypothesis",
            "list_hypotheses",
            "finish",
        ],
        initial_structure=initial_structure,
        initial_checkpoint=initial_checkpoint,
        problem_statement=problem_statement,
        should_review=should_review,
    )
    _current_tool_manager = tool_manager

    logger.info("📝 [WORKFLOW] Formatting system prompt...")
    system_prompt = CREATE_TASK_SYSTEM_PROMPT.format(
        tools_docs=tool_manager.get_tool_docs(),
        problem_statement=problem_statement,
        format_prompt=FORMAT_PROMPT_FIX,
    )
    enhanced_problem = problem_statement
    if enhancement:
        logger.info("✨ [WORKFLOW] Applying enhancement to problem statement...")
        enhanced_problem = problem_statement + "\n\n---\n\n# Enhanced Problem Analysis\n\n" + enhancement
    logger.info("📋 [WORKFLOW] Creating instance prompt...")
    instance_prompt = enhanced_problem

    print("🚀 [WORKFLOW] Executing agent workflow...")
    logger.info("🚀 [WORKFLOW] Executing agent workflow...")
    patch, is_success = execute_agent_workflow(
        cot,
        tool_manager,
        system_prompt,
        instance_prompt,
        n_max_steps,
        timeout,
        [KIMI_MODEL_NAME, GLM_MODEL_NAME],
        log_prefix="CREATE_MAIN_AGENT",
        initial_structure=initial_structure,
        function_behaviours=function_behaviours,
        files_to_modify=files_to_modify,
    )
    print("✅ [WORKFLOW] create_task_solve_workflow completed")
    logger.info("✅ [WORKFLOW] create_task_solve_workflow completed")
    return patch, is_success

def select_best_solution(
    solutions: List[dict],
    problem_statement: str,
) -> dict:
    """
    Use LLM to select the best solution among multiple candidates.
    Each solution dict has: {'solution_code': str, 'test_cases': str, 'patch': str}
    Returns the best solution dict.
    """
    if not solutions:
        return None
    if len(solutions) == 1:
        return solutions[0]

    SELECT_BEST_SOLUTION_PROMPT = textwrap.dedent(
        """
        You are an expert code reviewer.

        You are given:
        1. A problem statement
        2. Multiple candidate solutions

        Your task is to carefully compare all candidate solutions against the problem statement.
        You must explicitly analyze the differences between the solutions, not just evaluate them in isolation.

        Evaluate each solution on:
        - Correctness with respect to the stated requirements
        - Coverage of edge cases and invalid or unexpected inputs
        - Completeness in solving the full problem, not just part of it
        - Logical soundness and absence of bugs
        - Clarity and readability of the approach
        - Safety and robustness under realistic usage

        If a solution fails any required condition, it must not be selected, even if it is partially correct.

        Select the SINGLE best solution overall.
        If multiple solutions satisfy the requirements, prefer the one that is:
        - Easier to understand and reason about
        - Less error prone
        - More maintainable

        Strict rules:
        - Do NOT modify any solution
        - Do NOT combine solutions
        - Do NOT add new code
        - Do NOT assume missing behavior unless explicitly stated in the problem

        Return ONLY the final selection result as instructed elsewhere.

        Return ONLY a valid JSON object with exactly the following structure:
        {
            "selected_index": <0-based index of the best solution>,
            "reasoning": "<concise explanation of why this solution is best compared to the others>"
        }
        """
    )

    # Build comparison context
    solutions_context = ""
    for i, sol in enumerate(solutions):
        code_truncated = sol.get("solution_code", "")
        if len(code_truncated) > 60000:
            patch  = sol.get("patch", "")
            if patch:
                code_truncated = patch
        summary = sol.get("summary", "")
        solutions_context += f"\n\n=========== SOLUTION {i} ===========\n```\n{code_truncated}\n```\n\nSolution Summary:\n{summary}"

    retry = 0
    selected_model = QWEN_MODEL_NAME
    while retry < 10:
        try:
            messages = [
                {"role": "system", "content": SELECT_BEST_SOLUTION_PROMPT},
                {
                    "role": "user",
                    "content": f"Problem Statement:\n{problem_statement}\n\n{solutions_context}\n\nSelect the best solution and explain why.",
                },
            ]
            result = Network.make_request(messages, model=selected_model, temperature=0.0)
            if isinstance(result, tuple):
                response_text, _ = result
            else:
                response_text = result

            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                selection = json.loads(json_match.group())
                selected_index = selection.get("selected_index", 0)
                reasoning = selection.get("reasoning", "No reasoning provided")

                # Validate index
                if 0 <= selected_index < len(solutions):
                    logger.info(f"[SELECT_BEST_SOLUTION] Selected solution {selected_index}: {reasoning}")
                    return solutions[selected_index]
                else:
                    logger.warning(f"[SELECT_BEST_SOLUTION] Invalid index {selected_index}, using first solution")
                    return solutions[0]
            else:
                # Fallback: try to parse as JSON directly
                selection = json.loads(response_text)
                selected_index = selection.get("selected_index", 0)
                if 0 <= selected_index < len(solutions):
                    return solutions[selected_index]
                return solutions[0]
        except Exception as e:
            retry += 1
            logger.warning(f"[SELECT_BEST_SOLUTION] Retry {retry}/5: {e}")
            if retry >= 5:
                other_models = [model for model in AGENT_MODELS if model != selected_model]
                if other_models:
                    selected_model = other_models[random.randint(0, len(other_models) - 1)]
                time.sleep(2)

    logger.warning("[SELECT_BEST_SOLUTION] All retries failed, returning second solution")
    return solutions[0]


def get_misunderstanding_point(problem_statement: str, code_skeleton: str) -> str:
    """
    Analyzes the problem statement and code skeleton to identify potential misunderstanding points
    that could lead to implementation failures.

    Args:
        problem_statement: The problem description
        code_skeleton: The initial code structure/files provided

    Returns:
        A string containing the identified misunderstanding points and recommendations
    """
    MISUNDERSTANDING_ANALYSIS_PROMPT = textwrap.dedent(
        """
        You are an expert code reviewer and problem analyst. Your task is to identify potential misunderstanding points
        that could lead to implementation failures when solving the given problem.
        
        Analyze the problem statement and code skeleton to identify:
        
        1. **Ambiguous Requirements**:
           - Unclear specifications that could be interpreted multiple ways
           - Missing details that might lead to incorrect assumptions
           - Vague constraints or edge cases not explicitly mentioned
        
        2. **Common Misinterpretations**:
           - Typical mistakes developers make when reading similar problems
           - Easy-to-miss requirements or constraints
           - Subtle details that are often overlooked
        
        3. **Code Skeleton Analysis**:
           - Potential misunderstandings from the provided code structure
           - Function signatures that might be misinterpreted
           - Expected behavior implied by the skeleton that might conflict with requirements
           - Missing or incomplete hints in the skeleton
        
        4. **Implementation Pitfalls**:
           - Logic errors that are likely to occur
           - Edge cases that are easy to miss
           - Data structure or algorithm choices that might be incorrect
           - Boundary conditions that could be misunderstood
        
        5. **Critical Points to Clarify**:
           - Specific questions that should be answered before implementation
           - Assumptions that must be verified
           - Requirements that need explicit confirmation
        
        Format your response as markdown with clear section headers.
        Be specific and actionable. Focus on misunderstandings that would lead to test failures or incorrect implementations.
        """
    )

    retry = 0
    selected_model = random.choice([GLM_MODEL_NAME, KIMI_MODEL_NAME])
    while retry < 10:
        try:
            messages = [
                {"role": "system", "content": MISUNDERSTANDING_ANALYSIS_PROMPT},
                {
                    "role": "user",
                    "content": f"Problem Statement:\n{problem_statement}\n\nCode Skeleton:\n{code_skeleton}\n\nIdentify the potential misunderstanding points that could lead to implementation failures.",
                },
            ]
            misunderstanding_analysis, _ = Network.make_request(messages, model=selected_model, temperature=0.0)
            return misunderstanding_analysis
        except Exception as e:
            logger.error(f"Error in get_misunderstanding_point: {e}")
            retry += 1
            if retry < 10:
                # Try different model on retry
                other_models = [model for model in [GLM_MODEL_NAME, QWEN_MODEL_NAME] if model != selected_model]
                selected_model = other_models[random.randint(0, len(other_models) - 1)]
            time.sleep(random.uniform(2, 4))

    return ""


def get_fix_misunderstanding_point(problem_statement: str) -> str:
    """
    Analyzes a bug fix problem statement to identify common pitfalls and misinterpretations
    that could lead to incorrect fixes.

    Args:
        problem_statement: The bug report / problem description

    Returns:
        A string containing identified fix pitfalls and recommendations
    """
    FIX_MISUNDERSTANDING_PROMPT = textwrap.dedent(
        """
        You are an expert debugger and bug-fix analyst. Your task is to identify potential
        misinterpretations and pitfalls when fixing the given bug.
        
        Analyze the bug report to identify:
        
        1. **Root Cause Misidentification**:
           - Symptoms vs root cause confusion - where might developers fix the wrong layer?
           - Similar patterns elsewhere that could be mistaken for the bug location
        
        2. **Common Fix Pitfalls**:
           - Partial fixes that address symptoms but not root cause
           - Missing fixes in similar/related code paths
           - Edge cases the fix might break
        
        3. **Easy-to-Miss Requirements**:
           - Implicit assumptions in the bug report
           - Expected vs actual behavior nuances
           - Test coverage gaps
        
        4. **Critical Investigation Points**:
           - Key code paths to trace
           - Comparisons to make (working vs broken)
           - Evidence to gather before implementing
        
        Format as markdown with clear section headers. Be concise (max 500 words).
        """
    )
    retry = 0
    selected_model = random.choice([GLM_MODEL_NAME, KIMI_MODEL_NAME])
    while retry < 5:
        try:
            messages = [
                {"role": "system", "content": FIX_MISUNDERSTANDING_PROMPT},
                {
                    "role": "user",
                    "content": f"Bug Report:\n{problem_statement[:4000]}\n\nIdentify potential fix pitfalls and misinterpretations.",
                },
            ]
            analysis, _ = Network.make_request(messages, model=selected_model, temperature=0.0)
            return analysis or ""
        except Exception as e:
            logger.error(f"Error in get_fix_misunderstanding_point: {e}")
            retry += 1
            if retry < 5:
                other_models = [m for m in [GLM_MODEL_NAME, QWEN_MODEL_NAME] if m != selected_model]
                selected_model = other_models[random.randint(0, len(other_models) - 1)] if other_models else selected_model
            time.sleep(random.uniform(1, 3))
    return ""


def process_create_task(problem_statement: str, enhancement: str):
    global run_id, agent_start_time

    patch_text = ""
    repo_path = os.getenv("REPO_PATH", "/sandbox/repo")
    repod_dir = repo_path.split("/")[-1]
    if os.path.exists(repod_dir):
        os.chdir(repod_dir)
    set_env_for_agent()
    cwd = os.getcwd()

    initial_structure_str, files_to_modify = get_files_to_modify(problem_statement)
    function_behaviours = generate_function_behaviours(initial_structure_str, problem_statement)
    
    # Read initial file contents for validation
    initial_structure = {}
    for file_path in files_to_modify:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                initial_structure[file_path] = f.read()
        except Exception as e:
            logger.warning(f"Could not read initial content for {file_path}: {e}")
            continue

    code_skeleton = initial_structure_str
    if initial_structure:
        code_skeleton = code_skeleton + "\n\n--- File contents ---\n" + "\n\n".join(f"=== {path} ===\n{content}" for path, content in initial_structure.items())
    misunderstanding_point = get_misunderstanding_point(problem_statement, code_skeleton)
    if misunderstanding_point:
        logger.info(f"[PROCESS_CREATE_TASK] Misunderstanding point: {misunderstanding_point}")
        problem_statement = problem_statement + "\n\n--- Misunderstanding Points Analysis ---\n" + misunderstanding_point

    try:
        results = []
        for attempt in range(3):
            elapsed_time = time.time() - agent_start_time
            if elapsed_time > 950:
                break
            os.system("git reset --hard")
            os.system("git clean -fd")
            # Ensure timeout is at least 10 seconds to avoid issues
            remaining_time = max(10, 1300 - elapsed_time)
            logger.info(f"⏱️ [WORKFLOW] Attempt {attempt + 1}/3 - Elapsed time: {elapsed_time:.2f}s, Starting create_task_solve_workflow with timeout: {remaining_time:.2f}s...")
            patch_text, is_success = create_task_solve_workflow(problem_statement, timeout=1250 - elapsed_time, run_id_1=run_id, enhancement=enhancement, should_review=True, n_max_steps=200, initial_structure=initial_structure, function_behaviours=function_behaviours, files_to_modify=files_to_modify)

            # Get list of modified files (not newly created) and save their contents
            modified_files = ToolManager.get_modified_files_list()
            modified_files_content = {}  # Dict mapping file_path -> file_content
            result = ""

            if modified_files:
                # Create a temporary FileOperationsUtil instance to read files
                temp_file_ops = FileOperationsUtil(new_files_created=[])
                temp_file_ops.file_system_manager = FileSystemManager()
                temp_file_ops.search_manager = SearchManager()
                # Store file contents in dict
                for file_path in modified_files:
                    file_content = temp_file_ops.get_file_content(file_path, limit=-1)
                    modified_files_content[file_path] = file_content
                result = "\n\n".join([f"{file}\n{content}" for file, content in modified_files_content.items()])

            observation = "Success" if is_success else "Failed"
            if len(results) == 0 or is_success:
                results.append(
                    {
                        "solution_code": result,
                        "patch": patch_text,
                        "modified_files": modified_files,  # List of file paths
                        "modified_files_content": modified_files_content,  # Dict of file_path -> content
                        "summary": observation,
                    }
                )

        # Select best solution
        best_solution = select_best_solution(results, problem_statement)

        # Reset repository to clean state
        os.system("git reset --hard")
        os.system("git clean -fd")

        # Write the modified files from the best solution back to repository
        if best_solution and best_solution.get("modified_files_content"):
            file_ops = FileOperationsUtil(new_files_created=[])
            for file_path, file_content in best_solution["modified_files_content"].items():
                try:
                    file_ops.save(file_path, file_content)
                    logger.info(f"[PROCESS_TASK] Restored file: {file_path}")
                except Exception as e:
                    logger.error(f"[PROCESS_TASK] Error restoring file {file_path}: {e}")

            # Return the patch from the best solution
            if best_solution.get("patch"):
                patch_text = best_solution["patch"]

        logger.info("🧹 [WORKFLOW] Resetting git state...")
        print("✅ [WORKFLOW] process_create_task completed successfully")
        logger.info("✅ [WORKFLOW] process_create_task completed successfully")
    except Exception as e:
        print(f"❌ [WORKFLOW] Error in process_create_task: {e}")
        logger.error(f"Error in process_create_task: {e}, {traceback.format_exc()}")
    finally:
        os.chdir(cwd)
        logger.info("📁 [WORKFLOW] Restored original working directory")
    return patch_text

def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo"):
    global DEFAULT_PROXY_URL, run_id, agent_start_time
    print("🚀 [WORKFLOW] Starting agent_main - Entry point")
    logger.info("🚀 [WORKFLOW] Starting agent_main - Entry point")
    agent_start_time = time.time()
    run_id = os.getenv("EVALUATION_RUN_ID", "")
    repo_dir = os.path.abspath(repo_dir)
    sys.path.insert(0, repo_dir)
    if os.path.exists(repo_dir):
        os.chdir(repo_dir)
        logger.info(f"📁 [WORKFLOW] Changed directory to: {repo_dir}")
    logger.info("⚙️ [WORKFLOW] Setting up environment...")
    set_env_for_agent()

    timeout = 1400
    result = None
    exception_occurred = None
    task_completed = threading.Event()

    def run_task():
        nonlocal result, exception_occurred
        logger.info("🔄 [WORKFLOW] Starting task execution thread...")
        enhancement = ""
        try:
            global _current_tool_manager

            _current_tool_manager = ToolManager()
            problem_statement = input_dict.get("problem_statement")
            problem_type, _ = check_problem_type(input_dict.get("problem_statement"))
            if problem_type == PROBLEM_TYPE_FIX:
                result = process_fix_task(problem_statement, "")
            else:
                result = process_create_task(problem_statement, "")
        finally:
            task_completed.set()

    logger.info("🧵 [WORKFLOW] Creating and starting task thread...")
    task_thread = threading.Thread(target=run_task, daemon=True)
    task_thread.start()
    task_thread.join(timeout=timeout)

    timed_out = task_thread.is_alive()
    if timed_out:
        print(f"⏱️ [WORKFLOW] Task execution timed out after {timeout} seconds")
        logger.warning(f"Task execution timed out after {timeout} seconds, killing thread")

    print("Result: \n\n", result)

    global _current_tool_manager
    if _current_tool_manager is not None:
        try:
            final_patch = _current_tool_manager.get_final_git_patch()
            if final_patch:
                result = final_patch
                logger.info("✅ [WORKFLOW] Final patch generated successfully")
        except Exception as e:
            logger.error(f"❌ [WORKFLOW] Failed to get final patch: {e}")
            logger.warning(f"Failed to get final patch from tool manager: {e}")
        finally:
            _current_tool_manager = None

    try:
        logger.info("🧹 [WORKFLOW] Cleaning up git state...")
        subprocess.Popen(["git", "reset", "--hard"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    print("🎯 [WORKFLOW] agent_main completed")
    logger.info("🎯 [WORKFLOW] agent_main completed")
    
    logger.info("📝 [WORKFLOW] Generating final git patch...")

    return result if result else ""

def process_fix_task(problem_text: str, enhancement: str):
    cwd = os.getcwd()

    global run_id, agent_start_time
    print("🔧 [WORKFLOW] process_fix_task started")
    logger.info("🔧 [WORKFLOW] process_fix_task started")
    patch_text = ""

    try:
        fix_pitfalls = get_fix_misunderstanding_point(problem_text)
        if fix_pitfalls:
            logger.info(f"[PROCESS_FIX_TASK] Fix pitfalls analysis: {fix_pitfalls[:200]}...")
            problem_text = problem_text + "\n\n--- Fix Pitfalls Analysis ---\n" + fix_pitfalls

        results = []
        root_cause_analysis = ""
        for attempt in range(3):
            elapsed_time = time.time() - agent_start_time
            if elapsed_time > 850:
                break
            os.system("git reset --hard")
            os.system("git clean -fd")
            remaining_time = max(10, 1250 - elapsed_time)

            if attempt == 0:
                models = [KIMI_MODEL_NAME, GLM_MODEL_NAME]
            elif attempt == 1:
                models = [GLM_MODEL_NAME, KIMI_MODEL_NAME]
            else:
                models = [QWEN_MODEL_NAME, KIMI_MODEL_NAME]

            logger.info(f"⏱️ [WORKFLOW] Attempt {attempt + 1}/3 - Elapsed time: {elapsed_time:.2f}s, Starting fix_task_solve_workflow with timeout: {remaining_time:.2f}s...")
            patch_text, is_success = fix_task_solve_workflow(
                problem_text, timeout=remaining_time, run_id_1=run_id, enhancement=enhancement, should_review=True, root_cause_analysis=root_cause_analysis, models=models
            )

            # Get list of modified files (not newly created) and save their contents
            modified_files = ToolManager.get_modified_files_list()
            modified_files_content = {}  # Dict mapping file_path -> file_content
            result = ""

            if modified_files:
                # Create a temporary FileOperationsUtil instance to read files
                temp_file_ops = FileOperationsUtil(new_files_created=[])
                temp_file_ops.file_system_manager = FileSystemManager()
                temp_file_ops.search_manager = SearchManager()
                # Store file contents in dict
                for file_path in modified_files:
                    file_content = temp_file_ops.get_file_content(file_path, limit=-1)
                    modified_files_content[file_path] = file_content
                result = "\n\n".join([f"{file}\n{content}" for file, content in modified_files_content.items()])

            observation = "Success" if is_success else "Failed"
            if len(results) == 0 or is_success:
                results.append(
                    {
                        "solution_code": result,
                        "patch": patch_text,
                        "modified_files": modified_files,  # List of file paths
                        "modified_files_content": modified_files_content,  # Dict of file_path -> content
                        "summary": observation,
                    }
                )
            elif attempt == 2:
                results.append(
                    {
                        "solution_code": result,
                        "patch": patch_text,
                        "modified_files": modified_files,
                        "modified_files_content": modified_files_content,
                        "summary": observation,
                    }
                )

            if is_success:
                root_cause_analysis = ""
            elif patch_text:
                root_cause_analysis = (
                    "Previous attempt failed. Avoid repeating this approach.\n\n"
                    "Attempted fix (partial):\n"
                    + patch_text[:1500]
                    + ("..." if len(patch_text) > 1500 else "")
                )
                logger.info(f"[PROCESS_FIX_TASK] Passing failed attempt feedback to retry (patch len={len(patch_text)})")

        best_solution = select_best_solution(results, problem_text)

        os.system("git reset --hard")
        os.system("git clean -fd")

        if best_solution and best_solution.get("modified_files_content"):
            file_ops = FileOperationsUtil(new_files_created=[])
            for file_path, file_content in best_solution["modified_files_content"].items():
                try:
                    file_ops.save(file_path, file_content)
                    logger.info(f"[PROCESS_TASK] Restored file: {file_path}")
                except Exception as e:
                    logger.error(f"[PROCESS_TASK] Error restoring file {file_path}: {e}")

            # Return the patch from the best solution
            if best_solution.get("patch"):
                patch_text = best_solution["patch"]

        logger.info("🧹 [WORKFLOW] Resetting git state...")
        print("✅ [WORKFLOW] process_fix_task completed successfully")
        logger.info("✅ [WORKFLOW] process_fix_task completed successfully")
    except Exception as e:
        print(f"❌ [WORKFLOW] Error in process_fix_task: {e}")
        logger.error(f"Error in process_fix_task: {e}, {traceback.format_exc()}")
    finally:
        os.chdir(cwd)
        logger.info("📁 [WORKFLOW] Restored original working directory")
    return patch_text

def check_problem_type(problem_statement):
    type_count = {PROBLEM_TYPE_CREATE: 0, PROBLEM_TYPE_FIX: 0}
    enhancement = ""
    for _ in range(3):
        problem_type = get_problem_type(problem_statement, enhancement)
        type_count[problem_type] += 1
    if type_count[PROBLEM_TYPE_CREATE] > type_count[PROBLEM_TYPE_FIX]:
        return PROBLEM_TYPE_CREATE, enhancement
    elif type_count[PROBLEM_TYPE_FIX] > type_count[PROBLEM_TYPE_CREATE]:
        return PROBLEM_TYPE_FIX, enhancement
    return PROBLEM_TYPE_FIX, enhancement

def get_problem_type(problem_statement: str, enhancement: str) -> str:
    retry = 0
    PROBLEM_TYPE_CHECK_PROMPT = textwrap.dedent(
        """
        You are a helpful Problem Classifier to find a Task Name from PROJECT DESCRIPTION and project structure.
        Classify development tasks as either:
        - FIX: If the PROJECT DESCRIPTION is about fixing a bug or improving the existing codebase.
        - CREATE: If the PROJECT DESCRIPTION is about creating a new functionality from scratch.
        Output ONLY: "CREATE" or "FIX"
        """
    )
    
    selected_model = QWEN_MODEL_NAME
    
    while retry < 10:
        try:
            messages = [{"role": "system", "content": PROBLEM_TYPE_CHECK_PROMPT}, {"role": "user", "content": f"{problem_statement}\n# Enhanced Problem: \n{enhancement}"}]
            response, _ = Network.make_request(messages, model=selected_model)
            if response not in [PROBLEM_TYPE_CREATE, PROBLEM_TYPE_FIX]:
                retry += 1
            else:
                return response
        except Exception as e:
            retry += 1
            if retry > 4:
                other_models = [model for model in AGENT_MODELS if model != selected_model]
                selected_model = other_models[random.randint(0, len(other_models) - 1)]
            time.sleep(2)
    return PROBLEM_TYPE_FIX
