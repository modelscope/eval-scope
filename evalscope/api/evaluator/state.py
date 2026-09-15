from dataclasses import dataclass
from random import Random
from typing import Any, Dict, List, Optional, Sequence, Union, overload

from evalscope.api.agent import AgentTrace
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessage, ChatMessageUser, messages_pretty_str, messages_to_markdown
from evalscope.api.model import ModelOutput


class Target(Sequence[str]):
    """Target for scoring against the current TaskState.

    Target is a sequence of one or more strings. Use the
    `text` property to access the value as a single string.
    """

    def __init__(self, target: Union[str, List[str]]) -> None:
        self.target = target if isinstance(target, list) else [target]

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[str]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[str, Sequence[str]]:
        return self.target[index]

    def __len__(self) -> int:
        return len(self.target)

    @property
    def text(self) -> str:
        return ''.join(self.target)


@dataclass
class Choice:
    """
    A `Choice` represents a single choice in a multiple choice question.

    It is only relevant for the `multiple_choice` solver and corresponding
    `choice` scorer.
    """

    value: str
    """The original value of the choice from the `Sample`."""

    correct: Optional[bool]
    """Did the model think this choice satisfies the question? `None`
    indicates this has not been set yet"""

    original_position: int
    """Choices may be re-ordered during processing, this represents the
    original position in the sample's list of choices"""


class Choices(Sequence[Choice]):
    """
    Wrapper class for a list of `Choice` objects.

    Primarily simply to abstract away implementations of choice-specific
    functionality from the already-big `TaskState` class.
    """

    def __init__(self, choices: Union[List[str], List[Choice]]) -> None:
        """
        Setter for choices, intended to only be used with the `multiple_choice` scorer.

        Choices come from a list of choices for the sample, specifically used by
        the `multiple_choice` scorer.

        For example, if the sample was a multiple choice question like "What is
        the capital of France? A) Paris B) London C) Berlin", we would store the
        possible answers here.
        """
        self._choices: List[Choice] = []

        for i, choice in enumerate(choices):
            if isinstance(choice, str):
                self._choices.append(Choice(value=choice, correct=None, original_position=i))
            elif isinstance(choice, Choice):
                self._choices.append(choice)

    @overload
    def __getitem__(self, index: int) -> Choice: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Choice]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Choice, Sequence[Choice]]:
        return self._choices[index]

    def __len__(self) -> int:
        return len(self._choices)

    def mark_choice(self, index: int, correct: bool) -> None:
        """Set the value of a specific choice"""
        self._choices[index].correct = correct

    def shuffle(self, rand: Random = Random()) -> None:
        """
        Shuffle the choice order, setting the `original_position` so they can be mapped back to their original order.

        Some evals will shuffle the choices from the original sample to try to
        avoid the model answering correctly due to fine-tuning (or similar) on
        specific datasets.
        """
        shuffled_positions = list(range(len(self._choices)))
        rand.shuffle(shuffled_positions)

        shuffled_choices = [Choice('notachoice', None, -1)] * len(self._choices)

        for i, shuffled_position in enumerate(shuffled_positions):
            shuffled_choices[i] = self._choices[shuffled_position]
            shuffled_choices[i].original_position = shuffled_position

        self._choices = shuffled_choices


class TaskState:
    """
    The `TaskState` represents the internal state of the `Task` being run for a single `Sample`.

    The `TaskState` is passed to and returned from each solver during a sample's
    evaluation. It allows us to maintain the manipulated message history, the tools
    available to the model, the final output of the model, and whether the task
    is completed or has hit a limit.
    """

    def __init__(
        self,
        model: str,
        sample: Sample,
        messages: Optional[List[ChatMessage]] = None,
        output: Optional[ModelOutput] = None,
        completed: bool = False,
    ) -> None:
        self._model = model
        self._sample = sample
        self._sample_id = sample.id
        self._group_id = sample.group_id
        self._input = sample.input
        self._target = Target(sample.target)
        self._metadata = sample.metadata
        self._messages: List[ChatMessage] = messages if messages is not None else []
        self._output = output if output else ModelOutput(model=str(model))
        self._completed = completed
        self._agent_trace: Optional[AgentTrace] = None
        if sample.choices:
            self._choices = Choices(sample.choices)
        else:
            self._choices = Choices([])

    @property
    def model(self) -> str:
        """Name of model being evaluated."""
        return self._model

    @property
    def sample_id(self) -> int:
        """Unique id for sample."""
        return self._sample_id

    @property
    def group_id(self) -> int:
        """Group id for sample."""
        return self._group_id

    @property
    def input(self) -> Union[str, List[ChatMessage]]:
        """Input from the `Sample`, should be considered immutable."""
        return self._input

    @property
    def input_text(self) -> str:
        """
        Convenience function for accessing the initial input from the `Sample` as a string.

        If the `input` is a `List[ChatMessage]`, this will return the text from
        the last chat message
        """
        if isinstance(self._input, str):
            return self._input
        else:
            return messages_pretty_str(self._input)

    @property
    def input_markdown(self) -> str:
        """Get the input text as markdown.

        For multi-modal content, images will be represented in markdown format.
        """
        if isinstance(self._input, str):
            return self._input
        else:
            return messages_to_markdown(self._input)

    @property
    def choices(self) -> Choices:
        """Choices for the sample, if applicable."""
        return self._choices

    @property
    def user_prompt(self) -> ChatMessageUser:
        """User prompt for this state.

        Tasks are very general and can have may types of inputs.
        However, in many cases solvers assume they can interact with
        the state as a "chat" in a predictable fashion (e.g. prompt
        engineering solvers). This property enables easy read and
        write access to the user chat prompt. Raises an
        exception if there is no user prompt
        """
        prompt = next((m for m in reversed(self.messages) if m.role == 'user'), None)
        if prompt:
            return prompt
        else:
            raise ValueError('user_prompt requested from TaskState but none available')

    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata from the `Sample` for this `TaskState`"""
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Dict[str, Any]) -> None:
        self._metadata = metadata

    @property
    def messages(self) -> List[ChatMessage]:
        """
        Chat conversation history for sample.

        This will generally get appended to every time a `generate` call is made
        to the model. Useful for both debug and for solvers/scorers to assess
        model performance or choose the next step.
        """
        return self._messages

    @messages.setter
    def messages(self, messages: List[ChatMessage]) -> None:
        self._messages = messages

    @property
    def messages_markdown(self) -> str:
        """Get the messages as markdown.

        For multi-modal content, images will be represented in markdown format.
        """
        return messages_to_markdown(self._messages)

    @property
    def output(self) -> ModelOutput:
        """
        The 'final' model output once we've completed all solving.

        For simple evals this may just be the last `message` from the
        conversation history, but more complex solvers may set this directly.
        """
        return self._output

    @output.setter
    def output(self, output: ModelOutput) -> None:
        self._output = output

    @property
    def completed(self) -> bool:
        """Is the task completed."""
        return self._completed

    @completed.setter
    def completed(self, completed: bool) -> None:
        """Set the completed status."""
        self._completed = completed

    @property
    def target(self) -> str:
        """The scoring target for this `Sample`."""
        return self._target.text

    @property
    def target_values(self) -> List[str]:
        """Accepted target answers for multi-reference scoring."""
        return list(self._target)

    @target.setter
    def target(self, text: str) -> None:
        """Set the target for review purposes."""
        self._target = Target(text)

    @property
    def agent_trace(self) -> Optional[AgentTrace]:
        """Agent loop trajectory for this sample (None for non-agent runs)."""
        return self._agent_trace

    @agent_trace.setter
    def agent_trace(self, trace: Optional[AgentTrace]) -> None:
        self._agent_trace = trace
