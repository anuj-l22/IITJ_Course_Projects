from __future__ import annotations
from typing import List, Optional, Literal, Any, Dict
from pydantic import BaseModel, Field

Role = Literal["user", "agent", "system"]

class Citation(BaseModel):
    source: str
    snippet: Optional[str] = None

class Artifact(BaseModel):
    type: Literal["text", "url", "pdf", "table"]
    value: str

class PlanStep(BaseModel):
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None

class Message(BaseModel):
    role: Role
    agent: Optional[str] = None
    content: str = ""
    plan: List[PlanStep] = Field(default_factory=list)
    artifacts: List[Artifact] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)

class AgentResult(BaseModel):
    messages: List[Message]
    final: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
