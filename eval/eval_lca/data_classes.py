from dataclasses import dataclass, field
from typing import List, Any, Tuple

@dataclass
class ModelConfig:
    model: str
    device: str
    tokenizer: str
    max_seq_len: int
    max_context_len: int
    max_completion_len: int
    seed: int
    stopping_criteria: Any = field(default=None)

    def __post_init__(self):
        if self.max_context_len is None:
            # max_len_model = model.config.max_position_embeddings
            self.max_context_len = self.max_seq_len - self.max_completion_len

@dataclass
class DatasetConfig:
    config_name: str
    with_context_files: bool
    sep_symbol: str = field(default=None)
    do_filename_comment: bool = field(default=None)
    do_body_comment: bool = field(default=None)
    context_preprocessing: str = field(default=None)
    context_selection: str = field(default=None)
    context_file_ext: Tuple[str] = field(default=None)
    line_types: List[str] = field(default=None)

    def __post_init__(self):
        if self.with_context_files:
            self.sep_symbol = '\n[SEP]\n'
            self.do_filename_comment = True
            self.do_body_comment = False

            self.context_preprocessing = ""
            if self.do_filename_comment:
                self.context_preprocessing += "FC"
            else:
                self.context_preprocessing += "FN"

            if self.do_body_comment:
                self.context_preprocessing += "-BC"
            else:
                self.context_preprocessing += "-BN"

            if not self.sep_symbol.strip():
                self.context_preprocessing += "-SN"
            elif self.sep_symbol.strip()[1] == "S":
                self.context_preprocessing += "-SS"
            elif self.sep_symbol.strip()[1] == "M":
                self.context_preprocessing += "-SM"

            self.context_selection = "BUF-1-R"
            self.context_file_ext = (".py", ".txt", ".md")
            self.line_types = ["inproject", "infile"]