#!/bin/bash
# PostToolUse hook: validate pipeline files after edits
# Receives JSON on stdin with tool_name and tool_input

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Only check pipeline Python files
if [[ "$FILE_PATH" != *pipeline/*.py ]] || [[ "$FILE_PATH" == *config.py ]]; then
  exit 0
fi

# Check for inline threshold constants (should be in config.py)
if grep -Pn '(threshold|THRESHOLD|weight|WEIGHT)\s*=\s*\d' "$FILE_PATH" 2>/dev/null | grep -v 'config\.' | grep -v '^\s*#' | head -5 > /tmp/inline_check.txt; then
  if [ -s /tmp/inline_check.txt ]; then
    echo '{"hookSpecificOutput":{"message":"Warning: possible inline threshold detected. Constants belong in pipeline/config.py"}}'
    exit 0
  fi
fi

# Check for static word lists (project anti-pattern: use POS tagging instead)
if grep -Pn '(COMMON_WORDS|EXCLUDE_WORDS|STOP_WORDS|word_list|blacklist)\s*=' "$FILE_PATH" 2>/dev/null | grep -v '^\s*#' | head -5 > /tmp/wordlist_check.txt; then
  if [ -s /tmp/wordlist_check.txt ]; then
    echo '{"hookSpecificOutput":{"message":"Warning: static word list detected. Use spaCy POS tagging (token.pos_) instead — see rules/01-global-scaling.md"}}'
    exit 0
  fi
fi

# Check for hardcoded model names
if grep -Pn '"(en_core_web_|all-MiniLM-|paraphrase-)' "$FILE_PATH" 2>/dev/null | grep -v 'config\.' | head -5 > /tmp/model_check.txt; then
  if [ -s /tmp/model_check.txt ]; then
    echo '{"hookSpecificOutput":{"message":"Warning: hardcoded model name detected. Use config.SPACY_MODEL or config.CONTEXT_MODEL_NAME"}}'
    exit 0
  fi
fi

exit 0
