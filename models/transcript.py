from dataclasses import dataclass, field


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    segments: list[Segment] = field(default_factory=list)
    full_text: str = ""
    language: str = ""


@dataclass
class TopicSection:
    title: str
    start_time: str
    end_time: str
    content: str
    action_steps: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    summary: str
    sections: list[TopicSection] = field(default_factory=list)
