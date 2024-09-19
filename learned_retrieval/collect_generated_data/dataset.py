from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from learned_retrieval.collect_generated_data.data_classes import DatasetConfig
from learned_retrieval.collect_generated_data.utils import sort_filepathes

class LcaPythonCompletionDataset(Dataset):
    dataset_name = 'JetBrains-Research/lca-project-level-code-completion'
    # dataset_name = 'JetBrains-Research/lca-codegen-train'

    def __init__(self, dataset_config: DatasetConfig) -> None:
        self.dataset_config = dataset_config
        self.ds = load_dataset(self.dataset_name, dataset_config.config_name)['test']
        # self.ds = load_dataset(self.dataset_name, data_dir='train', data_files=['chunk_0061.parquet', 'chunk_0071.parquet', 'chunk_0081.parquet', 'chunk_0091.parquet', 'chunk_0101.parquet', 'chunk_0111.parquet'], split='train')

        print('Prepare data >>')
        self.prepare_data()

    def prepare_data(self):
        self.data = []

        for s in tqdm(self.ds):
            completion_filename = s['completion_file']['filename']
            completion_content = s['completion_file']['content'].split('\n')

            filtered_context = self._filter_context(completion_filename, s['repo_snapshot'])

            for line_type in s['completion_lines']:
                if self.dataset_config.line_types is None or line_type in self.dataset_config.line_types:
                    for line in s['completion_lines'][line_type]:
                        if line >= len(completion_content):
                            continue

                        completion = '\n'.join(completion_content[:line]) + '\n'
                        if len(completion) == 0:
                            continue

                        gt = completion_content[line]

                        model_input = self._merge_context([(completion_filename, completion)])
                        # model_input = self._prepare_model_input(completion_filename, completion)

                        context_files = [{'filename': '', 'content': ''}]

                        self.data.append({
                            # 'repo': s['repo'],
                            'completion_content': completion,
                            'ground_truth': gt,
                            'completion_filename': completion_filename,
                            'completion_line': line,
                            'completion_line_type': line_type,
                            'context_files': context_files,
                            'model_inputs': model_input,
                        })

                        if self.dataset_config.with_context_files:
                            if self.dataset_config.num_of_contexts is None:
                                completion_context = filtered_context + [(completion_filename, completion)]
                                model_input = self._merge_context(completion_context)

                                context_files = [{'filename': filename, 'content': content} for filename, content in filtered_context]

                                self.data.append({
                                        # 'repo': s['repo'],
                                        'completion_content': completion,
                                        'ground_truth': gt,
                                        'completion_filename': completion_filename,
                                        'completion_line': line,
                                        'completion_line_type': line_type,
                                        'context_files': context_files,
                                        'model_inputs': model_input,
                                    })
                            elif self.dataset_config.num_of_contexts == 1:
                                for context_filename, context_content in filtered_context:
                                    model_input = self._prepare_model_input(completion_filename, 
                                                                            completion, 
                                                                            context_filename, 
                                                                            context_content)

                                    context_files = [{'filename': context_filename, 'content': context_content}]

                                    self.data.append({
                                        # 'repo': s['repo'],
                                        'completion_content': completion,
                                        'ground_truth': gt,
                                        'completion_filename': completion_filename,
                                        'completion_line': line,
                                        'completion_line_type': line_type,
                                        'context_files': context_files,
                                        'model_inputs': model_input,
                                    })

    def _filter_context(self, completion_filename, repo_snapshot):
        filtered_context = []

        for context_filename, context_content in zip(repo_snapshot['filename'], repo_snapshot['content']):
            if self.dataset_config.context_file_ext is None or context_filename.lower().endswith(self.dataset_config.context_file_ext):    
                filtered_context.append((context_filename, context_content))
            
        if self.dataset_config.composer == "path_distance":
            sorted_pathes = sort_filepathes(completion_filename, filtered_context)
            if self.dataset_config.num_of_contexts is None:
                filtered_context = sorted_pathes[::-1] # decreasing dist
            elif self.dataset_config.num_of_contexts == 1:
                filtered_context = [sorted_pathes[0]]

        return filtered_context

    def _prepare_model_input(self, completion_filename, completion, context_filename=None, context_content=None):
        if self.dataset_config.do_filename_comment:
            context_filename = f"# {context_filename}"
            completion_filename = f"# {completion_filename}"

        if self.dataset_config.do_body_comment:
            context_content = context_content.split('\n')
            completion = "# " + '\n# '.join(completion) + '\n'

        if context_filename is not None and context_content is not None:
            context_model_input = context_filename + self.dataset_config.sep_symbol + context_content
            completion_model_input = completion_filename + self.dataset_config.sep_symbol + completion

            model_input = context_model_input + self.dataset_config.sep_symbol + completion_model_input
        else:
            model_input = completion_filename + self.dataset_config.sep_symbol + completion

        return model_input

    def _merge_context(self, contexts) -> str:
        context_lines = []
        for context_filename, context_content in contexts:
            if (context_content is None) or (context_content.strip() == ""):
                context_content = "# empty file"
            context_lines.extend(
                [context_filename, "", context_content, ""]
            )  # empty string is for additional new-line

        return "\n".join(context_lines)

    def __len__(self) -> int:

        return len(self.data)

    def __getitem__(self, idx) -> dict[str, str]:
        return self.data[idx]
