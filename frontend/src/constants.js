// Benchmark data state (mock initial, replaced on run)
export const INITIAL_RESULTS = null;

export const MODEL_META = {
  'gpt-4o-mini':  { label: 'GPT-4o-mini',   color: '#4f8ef7', cls: 'gpt'    },
  'claude-haiku': { label: 'Claude Haiku',   color: '#4ecdc4', cls: 'claude'  },
  'gemini-flash': { label: 'Gemini Flash',   color: '#ff6b4a', cls: 'gemini'  },
  'llama-3-8b':   { label: 'Llama 3 8B',     color: '#a78bfa', cls: 'llama'   },
  'mistral-7b':   { label: 'Mistral 7B',     color: '#f59e0b', cls: 'mistral' },
};

export const DATASETS = ['sst2', 'mmlu', 'gsm8k'];

export const ATTACKS = [
  { id: 'textbugger',  label: 'TextBugger'  },
  { id: 'deepwordbug', label: 'DeepWordBug' },
  { id: 'textfooler',  label: 'TextFooler'  },
  { id: 'checklist',   label: 'CheckList'   },
  { id: 'stresstest',  label: 'StressTest'  },
];

export const COVERAGE_ITEMS = [
  { id: 'character',   label: 'Character',   desc: 'typos, char swap',        status: 'active'  },
  { id: 'word',        label: 'Word',         desc: 'synonym, split/merge',    status: 'active'  },
  { id: 'sentence',    label: 'Sentence',     desc: 'shuffle, distract',       status: 'active'  },
  { id: 'semantic',    label: 'Semantic',     desc: 'paraphrase, rewrite',     status: 'active'  },

];

export function scoreClass(val) {
  if (val === null || val === undefined) return '';
  if (val >= 85) return 'score-high';
  if (val >= 65) return 'score-mid';
  return 'score-low';
}

export function now() {
  return new Date().toLocaleTimeString('en-GB', { hour12: false });
}
