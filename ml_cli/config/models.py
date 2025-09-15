# ml_cli/config/models.py
from pydantic import BaseModel, Field
from typing import Optional, Literal

class DataConfig(BaseModel):
    data_path: str = Field(..., description="Path to data file")
    target_column: Optional[str] = None
    
class TaskConfig(BaseModel):
    type: Literal["classification", "regression", "clustering"]
    
class MLConfig(BaseModel):
    data: DataConfig
    task: TaskConfig
    output_dir: str = Field(default="output")
    tpot: dict = Field(default_factory=lambda: {"generations": 4})