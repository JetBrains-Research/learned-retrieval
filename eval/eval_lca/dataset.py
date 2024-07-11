from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from eval_lca.data_classes import DatasetConfig

class LcaPythonCompletionDataset(Dataset):
    dataset_name = 'JetBrains-Research/lca-project-level-code-completion'

    def __init__(self, dataset_config: DatasetConfig) -> None:
        self.dataset_config = dataset_config
        self.ds = load_dataset(self.dataset_name, dataset_config.config_name)['test']

        print('Prepare data >>')
        self.prepare_data(dataset_config.with_context_files)
                        
    def prepare_data(self, with_context_files):
        self.data = []
        self.repo_snapshot_lens = []

        for s in tqdm(self.ds):
            completion_filename = s['completion_file']['filename']
            completion_content = s['completion_file']['content'].split('\n')

            if self.dataset_config.line_types is None:
                self.dataset_config.line_types = s['completion_lines']

            for line_type in s['completion_lines']:
                if line_type in self.dataset_config.line_types:
                    for line in s['completion_lines'][line_type]:
                        if line >= len(completion_content):
                            continue

                        completion = '\n'.join(completion_content[:line]) + '\n'
                        if len(completion) == 0:
                            continue

                        gt = completion_content[line]

                        sample = {'completion': completion, 'gt': gt}

                        if with_context_files:
                            num_of_context_files = 0

                            for context_filename, context_content in zip(s['repo_snapshot']['filename'], s['repo_snapshot']['content']):
                                if context_filename.lower().endswith(self.dataset_config.context_file_ext):    
                                    num_of_context_files += 1

                                    model_input = self._prepare_model_input(completion_filename, 
                                                                            completion, 
                                                                            context_filename, 
                                                                            context_content)
                                    
                                    self.data.append({
                                        'sample': sample,
                                        'completion_file': s['completion_file'],
                                        'completion_line': line,
                                        'completion_line_type': line_type,
                                        'context_files': [{'filename': context_filename, 'content': context_content}],
                                        'model_input': model_input,
                                        })
                                    
                            self.repo_snapshot_lens.append(num_of_context_files)

                        else:
                            model_input = completion_filename + self.dataset_config.sep_symbol + completion                        
                            self.repo_snapshot_lens.append(1)

                            self.data.append({
                                    'sample': sample,
                                    'completion_file': s['completion_file'],
                                    'completion_line': line,
                                    'completion_line_type': line_type,
                                    'model_input': model_input,
                                    })
    
    def _prepare_model_input(self, completion_filename, completion, context_filename, context_content):
        if self.dataset_config.do_filename_comment:
            context_filename = f"# {context_filename}"
            completion_filename = f"# {completion_filename}"

        if self.dataset_config.do_body_comment:
            context_filename = context_filename.split('\n')
            context_filename = "# " + '\n# '.join(context_filename) + '\n'
            
        context_model_input = context_filename + self.dataset_config.sep_symbol + context_content
        completion_model_input = completion_filename + self.dataset_config.sep_symbol + completion

        model_input = context_model_input + self.dataset_config.sep_symbol + completion_model_input

        return model_input
        
                    
    def __len__(self) -> int:
        return len(self.data)
    
    def get_limited_len(self, limit_samples):
        return sum(self.repo_snapshot_lens[:limit_samples])

    def get_repo_snapshot_lens(self, limit_samples):
        return self.repo_snapshot_lens[:limit_samples]

    def __getitem__(self, idx) -> dict[str, str]:
        return self.data[idx]