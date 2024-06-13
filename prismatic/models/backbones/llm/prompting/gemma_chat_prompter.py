"""
gemma_chat_prompter.py

# No System Prompts for Gemma => 
# Following prompting instructions @ https://ai.google.dev/gemma/docs/formatting
"""

from typing import Optional

from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder




class GemmaChatPromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)

        # Gemma2 specific
        self.bos, self.eos = "<bos>", "<eos>"

        # Get role-specific "wrap" functions
        self.wrap_human = lambda msg: f"<start_of_turn>user\n{msg}<end_of_turn>\n"
        self.wrap_gpt = lambda msg: f"<start_of_turn>model\n{msg if msg != '' else ' '}<end_of_turn>{self.eos}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            wrapped_message = self.wrap_human(message)
        elif (self.turn_count % 2) == 0:
            wrapped_message = self.wrap_human(message)
        else:
            wrapped_message = self.wrap_gpt(message)

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> None:
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            sys_message = self.wrap_human(message)
            prompt_copy += sys_message

        else:
            human_message = self.wrap_human(message)
            prompt_copy += human_message

        return prompt_copy.rstrip()

    def get_prompt(self) -> str:
        # We need the auto-inseerted <bos> for better fine-tuning convergence. Ref: https://unsloth.ai/blog/gemma-bugs
        return self.prompt.rstrip()